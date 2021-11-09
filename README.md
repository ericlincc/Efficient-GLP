# Efficient-GLP

This repository contains an implementation of the Coordinate Linear Variance Reduction (CLVR) algorithm for Generalized Linear Programming (GLP) problems, and the support framework for conducting various numerical experiments. Please refer to [our paper](https://arxiv.org/abs/2111.01842) for details.


## Installing the Julia packages to run experiments

```
julia> import Pkg
julia> Pkg.add(["Arpack", "CSV", "Dates", "LinearAlgebra", "Logging", "SparseArrays"])
```


## A brief overview

The bulk of the code for running the experiments is located within `src`, and all implemented algorithms including CLVR are located under `src/algorithms`. We use Julia scripts to conduct our experiments and they are located within `scripts`.


## A basic example


All problem instances must be wrapped into a problem instance before passing into algorithms. For example, in the case of standard form linear programs considerd in the paper where we want to 

$$
\max_{x \in \mathbb{R}^d} c^T x \quad \mathrm{s.t.} \quad A x = b, x \ge 0.
$$

Translating this problem instance into code, we have
```
include("./src/problems/standardLP.jl")

A_T = sparse([1 2; 3 4])  # Transpose of A is used due to Julia's Sparse CSC Matrix
b = Array{Float64}([1; 2])
c = Array{Float64}([-1; 1])

problem = StandardLinearProgram(A_T, b, c)
```

Next, we need to specify some exit criteria.

```
include("../src/algorithms/utils/exitcriterion.jl")

maxiter = 1e5
maxtime = 60
targetaccuracy = 1e-7
loggingfreq = 1
exitcriterion = ExitCriterion(maxiter, maxtime, targetaccuracy, loggingfreq)
```

Then to run the clvr algorithm with restart, we can simply set the algorithm parameters and run
```
# Common algo parameters
blocksize = 1
R = sqrt(blocksize)
γ = 0.1
restartfreq = Inf  # For restart when metric halves, set restartfreq=Inf 

run_results = clvr_lazy_restart_x_y(
    problem,
    exitcriterion;
    blocksize=blocksize,
    R=R * clvr_R_multiplier,
    γ=γ,
    restartfreq=restartfreq,
)
```
