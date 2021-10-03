using LinearAlgebra
# the problem to optimize
#  \min_{\vx\in\gX\subset\sR^d} \max_{\vy\in\sR^n} \Big\{ L(\vx,\vy) = \vc^T\vx + r(\vx)   + \vy^T\mA\vx   - \vy^T\vb\Big\}.
# op_X_r: (I + tau_n ∂G)^{-1}
# γ: strong convexity
# σ, τ, θ: sequence for update
# L: Lipschitz constant

# to be consistent with the sparse column format, we consider
# D: R^{d×n}

# function test_pdhg()

#     n, d = 1000, 500
#     γ = 0.0
#     D = (randn(d, n) +  ones(d, n)) /√n
#     # b = randn(n)
#     x_star = ones(d)
#     b = (x_star' * D)'
#     c = ones(d)
#     L = 1
#     τ, σ = 1.0/L, 1.0/L
#     K = 100000
#     x0 = zeros(d)
#     y0 = zeros(n)
#     op_X_r(x, τ) = max.(x, 0.0)
#     pdhg(D, b, c, op_X_r, x0, y0, n, d, γ, σ, τ, L, K)
# end


function pdhg(D, b, c, op_X_r, x0,  y0, n, d, γ, σ, τ, L, K)
    x_bar = x0
    y = deepcopy(y0)
    x = deepcopy(x0)
    x_tilde = zero(x0)
    modulor = 10
    for k = 1:K
        y[:] = y[:] + σ * ((x_bar' * D)' - b)
        x_pre = deepcopy(x)
        x[:] = x[:]  - τ * (D * y[:] + c[:])
        x = op_X_r(x, τ)
        θ = 1/sqrt(1+2*γ*τ)
        τ = θ * τ
        σ = σ / θ
        x_bar[:] = x_bar[:] + θ * (x[:] -  x_pre[:])
        # compute loss and save
        x_tilde[:] = x_tilde[:] + x[:]
        # @info (x_tilde/k)' * D
        if k % modulor == 0
            norm_const = norm(((x_tilde/k)' * D)' - b)
            @info "k:$k, PDHG constraint norm: $norm_const"
            modulor *= 10
        end
    end
end
