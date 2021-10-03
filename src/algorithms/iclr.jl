
# σ: strong convexity constant
# γ: balance factor for ||x-x_0|| and ||y -0 y_0||
# A: R^{d\times n}
#



# function test_iclr()
#     n, d = 1000, 500
#     γ = 0.0
#     D = (randn(d, n) +  ones(d, n)) /√n
#     # b = randn(n)
#     x_star = ones(d)
#     b = (x_star' * D)'
#     c = ones(d)
#     R = 1.0
#     γ = 50
#     σ = 0.0

#     K = 100000
#     x0 = zeros(d)
#     y0 = zeros(n)
#     blocks = []
#     bs = 1
#     len_b = n ÷ bs
#     for i = 1:len_b
#         push!(blocks, (1 + (i-1) * bs): (i * bs))
#     end
#     op_X_r(x, τ) = max.(x, 0.0)
#     iclr(D, b, c, op_X_r, x0, y0, n, d, R, σ, γ, blocks, K * len_b)
# end


function iclr(D, b, c, op_X_r, x0, y0, n, d, R, σ, γ, blocks, K)
    m = length(blocks)
    a = 1/(R * m)
    A = a
    pre_a, pre_A = 0.0, 0.0
    q = a * c
    idx_seq = 1:m
    x = deepcopy(x0)
    y = deepcopy(y0)
    x_tilde = zero(x0)
    z = zero(x0)
    modulor = 10
    for k = 1:K
        x[:] = x0[:] - 1.0/γ * q[:]
        x = op_X_r(x, 1.0/γ * A)
        j = rand(idx_seq)
        Delta_y = γ * m * a * ((x'*D[:, blocks[j]])' - b[blocks[j]])
        y[blocks[j]] = y[blocks[j]] + Delta_y
        pre_a, pre_A = a, A
        a = sqrt(1 + σ * A / γ)/(R * m)
        A = A + a
        Delta_Delta_y = D[:, blocks[j]] * Delta_y
        z[:] = z[:] + Delta_Delta_y
        q[:] = q[:] + a * (z + c) + m * pre_a * Delta_Delta_y

        #
        x_tilde += pre_a * x
        if k % modulor == 0
            norm_const = norm(((x_tilde/pre_A)' * D)' - b)
            @info "k:$(k), ICLR constraint norm: $norm_const"
            modulor *= 10
        end
    end
    return x_tilde/A
end


test_iclr()
