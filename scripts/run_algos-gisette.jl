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
include("../src/algorithms/pure_cd_restart.jl")
include("../src/algorithms/pdhg_restart.jl")
include("../src/algorithms/spdhg_restart.jl")


outputdir = "./run_results/"  # TODO: As input argument
filepath = "./data/gisette.txt"    ####################################################
dataset = "gisette"     ####################################################
dim_dataset = 5000     ####################################################
num_dataset = 6000       ####################################################

timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
loggingfilename = "$(outputdir)/$(timestamp)-$(dataset)-execution_log.txt"
io = open(loggingfilename, "w+")
logger = SimpleLogger(io)

with_logger(logger) do

    yX_T = read_libsvm_into_yXT_sparse(filepath, dim_dataset, num_dataset)
    A_T, b, c = droreformuation_wmetric_hinge_standardformnormalized(yX_T, 1.0, 0.1)

    problem = StandardLinearProgram(A_T, b, c)
    exitcriterion = ExitCriterion(1e12, 3600., 1e-5, 5)
    L = 93.9  # TODO: This is a hard constant    ####################################################

    @info "A_T has size: $(size(A_T))"
    @info "A_T has nnz: $(size(findnz(A_T)[1])[1]))"
    @info "nnz ratio: $(size(findnz(A_T)[1])[1] / (size(A_T)[1] * size(A_T)[2])))"


    if "1" in ARGS  # TODO: Use algo names instead
        println("========================================")
        println("Running iclr_lazy_restart_x_y.")

        # ICLR Lazy with Restarts
        iclr_blocksize = 5
        iclr_R = sqrt(iclr_blocksize)

        r_iclr_lazy_restart = iclr_lazy_restart_x_y(problem, exitcriterion; blocksize=iclr_blocksize, R=sqrt(iclr_R))
        export_filename = "$(outputdir)/$(timestamp)-$(dataset)-iclr_lazy_restart_x_y-blocksize$(iclr_blocksize)-R$(iclr_blocksize).csv"
        exportresultstoCSV(r_iclr_lazy_restart, export_filename)

        println("========================================")
    end


    if "2" in ARGS
        println("========================================")
        println("Running pdhg_restart_x_y.")

        # PDHG with Restarts
        pdhg_L_multipler = 1.5

        r_pdhg_restart = pdhg_restart_x_y(problem, exitcriterion; L=pdhg_L_multipler * L)
        export_filename = "$(outputdir)/$(timestamp)-$(dataset)-pdhg_restart_x_y-L$(pdhg_L_multipler * L).csv"
        exportresultstoCSV(r_pdhg_restart, export_filename)

        println("========================================")
    end


    if "3" in ARGS
        println("========================================")
        println("Running spdhg_restart_x_y.")

        # SPDHG with Restarts
        spdhg_blocksize = 100
        spdhg_R = sqrt(spdhg_blocksize)

        r_spdhg_restart = spdhg_restart_x_y(problem, exitcriterion; blocksize=spdhg_blocksize, R=spdhg_R)
        export_filename = "$(outputdir)/$(timestamp)-$(dataset)-spdhg_restart_x_y-blocksize$(spdhg_blocksize)-R$(spdhg_R).csv"
        exportresultstoCSV(r_spdhg_restart, export_filename)
        
        println("========================================")
    end


    if "4" in ARGS
        println("========================================")
        println("Running pure_cd_restart_x_y.")

        # PURE_CD with Restarts
        purecd_blocksize = 5
        purecd_R = sqrt(purecd_blocksize)

        r_pure_cd_restart = pure_cd_restart_x_y(problem, exitcriterion; blocksize=purecd_blocksize, R=purecd_R)
        export_filename = "$(outputdir)/$(timestamp)-$(dataset)-pure_cd_restart_x_y-blocksize$(purecd_blocksize)-R(purecd_R).csv"
        exportresultstoCSV(r_pure_cd_restart, export_filename)

        println("========================================")
    end

end
