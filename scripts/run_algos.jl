using Arpack
using CSV
using Dates
using LinearAlgebra
using Logging
using SparseArrays

BLAS.set_num_threads(1)

include("../src/problems/standardLP.jl")
include("../src/algorithms/utils/exitcriterion.jl")
include("../src/algorithms/utils/results.jl")
include("../src/algorithms/utils/helper.jl")
include("../src/algorithms/iclr_lazy.jl")
include("../src/problems/dro/utils/libsvm_parser.jl")
include("../src/problems/dro/wasserstein.jl")

include("../src/algorithms/iclr_lazy_restart.jl")
include("../src/algorithms/pdhg_restart.jl")
include("../src/algorithms/purecd_restart.jl")
include("../src/algorithms/spdhg_restart.jl")


DATASET_INFO = Dict([
    ("a1a", (123, 1605)),
    ("a9a", (123, 32561)),
    ("gisette", (5000, 6000)),
    ("news20", (1355191, 19996)),
    ("rcv1", (47236, 20242)),
])

# Dataset parameters
outputdir = "./run_results/"
dataset = ARGS[1]
if !haskey(DATASET_INFO, dataset)
    throw(ArgumentError("Invalid dataset name supplied."))
end
dim_dataset, num_dataset = DATASET_INFO[dataset]
filepath = "./data/$(dataset).txt"

# Problem instance parameters
κ = 0.1
ρ = 10.

# Problem instance instantiation
yX_T = read_libsvm_into_yXT_sparse(filepath, dim_dataset, num_dataset)
A_T, b, c = droreformuation_wmetric_hinge_standardformnormalized(yX_T, κ, ρ)
problem = StandardLinearProgram(A_T, b, c)
L = svds(A_T, nsv = 1)[1].S[1]

# Exit criterion
maxiter = 1e12
maxtime = 3600 * 12
targetaccuracy = 1e-7
loggingfreq = 5
exitcriterion = ExitCriterion(maxiter, maxtime, targetaccuracy, loggingfreq)

# Common algo parameters
blocksize = 50
R = sqrt(blocksize)
γ = parse(Float64, ARGS[2])  # TODO: Use ArgParse
restartfreq = Inf  # For restart when metric halves, set restartfreq=Inf 

timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS-sss")
loggingfilename = "$(outputdir)/$(dataset)-$(join(ARGS[3:end], "_"))-execution_log-$(timestamp).txt"
io = open(loggingfilename, "w+")
logger = SimpleLogger(io)

println("timestamp = $(timestamp)")
println("Completed initialization.")

with_logger(logger) do

    @info "Running on $(dataset) dataset."
    @info "--------------------------------------------------"
    @info "κ = $(κ)"
    @info "ρ = $(ρ)"
    @info "--------------------------------------------------"
    @info "A_T has size: $(size(A_T))"
    @info "A_T has nnz: $(size(findnz(A_T)[1])[1]))"
    @info "nnz ratio: $(size(findnz(A_T)[1])[1] / (size(A_T)[1] * size(A_T)[2])))"
    @info "L = $(L)"
    @info "--------------------------------------------------"
    @info "maxiter = $(maxiter)"
    @info "maxtime = $(maxtime)"
    @info "targetaccuracy = $(targetaccuracy)"
    @info "loggingfreq = $(loggingfreq)"
    @info "--------------------------------------------------"
    @info "blocksize = $(blocksize)"
    @info "R = $(R)"
    @info "γ = $(γ)"
    @info "restartfreq = $(restartfreq)"

    if "1" in ARGS[3:end]  # TODO: Use algo names instead
        println("========================================")
        println("Running iclr_lazy_restart_x_y.")

        iclr_R_multiplier = 1.0
        println("iclr_R_multiplier = $(iclr_R_multiplier)")

        r_iclr_lazy_restart = iclr_lazy_restart_x_y(
            problem,
            exitcriterion;
            blocksize=blocksize,
            R=R * iclr_R_multiplier,
            γ=γ,
            restartfreq=restartfreq,
            io=io,
        )

        export_filename = "$(outputdir)/$(dataset)-iclr_lazy_restart_x_y-$(timestamp).csv"
        exportresultstoCSV(r_iclr_lazy_restart, export_filename)

        println("========================================")
    end


    if "2" in ARGS[3:end]  # TODO: Use algo names instead
        println("========================================")
        println("Running pdhg_restart_x_y.")

        pdhg_L_multiplier = 1.0
        println("pdhg_L_multiplier = $(pdhg_L_multiplier)")

        r_pdhg_restart = pdhg_restart_x_y(
            problem,
            exitcriterion;
            L=L * pdhg_L_multiplier,
            γ=γ,
            restartfreq=restartfreq,
            io=io,
        )

        export_filename = "$(outputdir)/$(dataset)-pdhg_restart_x_y-$(timestamp).csv"
        exportresultstoCSV(r_pdhg_restart, export_filename)

        println("========================================")
    end


    if "3" in ARGS[3:end]  # TODO: Use algo names instead
        println("========================================")
        println("Running spdhg_restart_x_y.")

        spdhg_R_multiplier = 1.0
        println("spdhg_R_multiplier = $(spdhg_R_multiplier)")

        r_spdhg_restart = spdhg_restart_x_y(
            problem,
            exitcriterion;
            blocksize=blocksize,
            R=R * spdhg_R_multiplier,
            γ=γ,
            restartfreq=restartfreq,
            io=io,
        )

        export_filename = "$(outputdir)/$(dataset)-spdhg_restart_x_y-$(timestamp).csv"
        exportresultstoCSV(r_spdhg_restart, export_filename)

        println("========================================")
    end


    if "4" in ARGS[3:end]  # TODO: Use algo names instead
        println("========================================")
        println("Running purecd_restart_x_y.")

        purecd_R_multiplier = 5.0
        println("purecd_R_multiplier = $(purecd_R_multiplier)")

        r_purecd_restart = purecd_restart_x_y(
            problem,
            exitcriterion;
            blocksize=blocksize,
            R=R * purecd_R_multiplier,
            γ=γ,
            restartfreq=restartfreq,
            io=io,
        )

        export_filename = "$(outputdir)/$(dataset)-purecd_restart_x_y-$(timestamp).csv"
        exportresultstoCSV(r_purecd_restart, export_filename)

        println("========================================")
    end
end
