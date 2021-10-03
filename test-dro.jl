using SparseArrays
using LinearAlgebra


include("src/algorithms/iclr_lazy.jl")
include("src/problems/dro/utils/libsvm_parser.jl")
include("src/problems/dro/wasserstein.jl")


# the problem to optimize
#  \min_{\vx\in\gX\subset\sR^d} \max_{\vy\in\sR^n} \Big\{ L(\vx,\vy) = \vc^T\vx + r(\vx)   + \vy^T\mA\vx   - \vy^T\vb\Big\}.
# op_X_r: (I + tau_n ∂G)^{-1}
# γ: strong convexity
# σ, τ, θ: sequence for update
# L: Lipschitz constant
# to be consistent with the sparse column format, we consider
# D: R^{d×n}
function test_iclr_lazy(D::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, c::Vector{Float64})

    d, n = size(D)
#     D = (randn(d, n) +  ones(d, n)) /√n
#     D = sparse(D)
#     # b = randn(n)
#     x_star = ones(d)
#     b = (x_star' * D)'

#     c = ones(d)  # TODO: Need to get from reformulation
    R = 10
    γ = 1
    σ = 0.0
    K = 100000
    x0 = zeros(d)
    y0 = zeros(n)
    blocks = []
    C = []
    bs = 10  # block size
    len_b = n ÷ bs

    for i = 1:len_b
        push!(blocks, (1 + (i-1) * bs): (i * bs))
        row_set = []
        for j = 1:length(blocks[i])
            row_set = union(row_set, rowvals(D[:, blocks[i][j]]))
        end
        push!(C, row_set)
        # @info row_set
    end
    op_X_r(x, τ) = max.(x, 0.0)

    # for i = 1:100
    #     x0, y0 = iclr_lazy_restart_x_y(D, b, c, op_X_r, x0, y0, n, d, γ, σ, R, blocks, C, K * len_b)
    # end

    iclr_lazy(D, b, c, op_X_r, x0, y0, n, d, γ, σ, R, blocks, C, K * len_b)
end

yX_T = read_libsvm_into_yXT_sparse("a1a.txt", 123, 1605)
A_T, b, c = droreformuation_wmetric_hinge_standardformnormalized(yX_T, 0.1, 0.1)

test_iclr_lazy(A_T, b, c)
