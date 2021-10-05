using CSV
using Dates
using LinearAlgebra
using SparseArrays

include("../src/problems/standardLP.jl")
include("../src/algorithms/utils/exitcriterion.jl")
include("../src/algorithms/utils/results.jl")
include("../src/algorithms/utils/helper.jl")
include("../src/algorithms/iclr_lazy.jl")
include("../src/problems/dro/utils/libsvm_parser.jl")
include("../src/problems/dro/wasserstein.jl")
include("../src/algorithms/iclr_lazy_restart.jl")
include("../src/algorithms/pdhg.jl")
include("../src/algorithms/spdhg.jl")


outputdir = "./run_results/"
filepath = "./data/a9a.txt"
dataset = "a9a"
dim_dataset = 123
num_dataset = 32561

timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")

yX_T = read_libsvm_into_yXT_sparse(filepath, dim_dataset, num_dataset)
A_T, b, c = droreformuation_wmetric_hinge_standardformnormalized(yX_T, 1.0, 0.1)

problem = StandardLinearProgram(A_T, b, c)
exitcriterion = ExitCriterion(1e12, 3600., 1e0, 5)  ###################  TODO: CHANGE ACCURACY

println("A_T has size: ", size(A_T))
println("A_T has nnz: ", size(findnz(A_T)[1])[1])
println("nnz ratio: ", size(findnz(A_T)[1])[1] / (size(A_T)[1] * size(A_T)[2]))


# ICLR Lazy with Restarts
r_iclr_lazy_restart = iclr_lazy_restart_x_y(problem, exitcriterion; blocksize=5, R=2.23)
export_filename = "$(outputdir)/$(timestamp)-$(dataset)-iclr_lazy_restart_x_y-blocksize5-R2_23.csv"
exportresultstoCSV(r_iclr_lazy_restart, export_filename)


println("========================================")
println("========================================")


