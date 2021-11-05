# Script for executing Gurobi on selected datasets on the DRO problem with Wasserstein metric
# based ambiguity sets. Command line usage:
# julia run_solver.jl <dataset> <method>
# method: -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex.

using Gurobi
using JuMP
using LinearAlgebra
using SparseArrays

BLAS.set_num_threads(1)

include("../src/problems/standardLP.jl")
include("../src/algorithms/utils/exitcriterion.jl")
include("../src/algorithms/utils/results.jl")
include("../src/algorithms/utils/helper.jl")
include("../src/algorithms/clvr_lazy.jl")
include("../src/problems/dro/utils/libsvm_parser.jl")
include("../src/problems/dro/wasserstein.jl")


DATASET_INFO = Dict([
    ("a1a", (123, 1605)),
    ("a9a", (123, 32561)),
    ("gisette", (5000, 6000)),
    ("news20", (1355191, 19996)),
    ("rcv1", (47236, 20242)),
])

# ARGS
dataset = ARGS[1]
gurobimethod = parse(Int, ARGS[2])

# Dataset parameters
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


# Gurobi
println("Setting up LP")
starttime = time()
vector_model = Model(Gurobi.Optimizer)
set_optimizer_attribute(vector_model, "Threads", 1)
set_optimizer_attribute(vector_model, "Method", gurobimethod)
@variable(vector_model, x[1:size(A_T)[1]] >= 0)
@constraint(vector_model, equality, A_T' * x .== b)
@objective(vector_model, Min, c' * x)
endtime = time()
println("=====> Setting up: $(endtime - starttime)")


println("Solving LP")
starttime = time()
optimize!(vector_model)
endtime = time()
println("=====> Solve time: $(endtime - starttime)")


println(objective_value(vector_model))
println(termination_status(vector_model))
println(primal_status(vector_model))
