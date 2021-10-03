# using LinearAlgebra
# using SparseArrays


# the problem to optimize
#  \min_{\vx\in\gX\subset\sR^d} \max_{\vy\in\sR^n} \Big\{ L(\vx,\vy) = \vc^T\vx + r(\vx)   + \vy^T\mA\vx   - \vy^T\vb\Big\}.
# op_X_r: (I + tau_n ∂G)^{-1}
# γ: strong convexity
# σ, τ, θ: sequence for update
# L: Lipschitz constant
# to be consistent with the sparse column format, we consider
# A_T: R^{d×n}


# function test_iclr_lazy()

#     n, d = 1000, 500
#     A_T = (randn(d, n) +  ones(d, n)) /√n
#     A_T = sparse(A_T)
#     # b = randn(n)
#     x_star = ones(d)
#     b = (x_star' * A_T)'
#     c = ones(d)
#     R = 2
#     γ = 1
#     σ = 0.0
#     K = 100000
#     x0 = zeros(d)
#     y0 = zeros(n)
#     blocks = []
#     C = []
#     bs = 10  # block size
#     len_b = n ÷ bs

#     for i = 1:len_b
#         push!(blocks, (1 + (i-1) * bs): (i * bs))
#         row_set = []
#         for j = 1:length(blocks[i])
#             row_set = union(row_set, rowvals(A_T[:, blocks[i][j]]))
#         end
#         push!(C, row_set)
#         # @info row_set
#     end
#     op_X_r(x, τ) = max.(x, 0.0)

#     # for i = 1:100
#     #     x0, y0 = iclr_lazy_restart_x_y(A_T, b, c, op_X_r, x0, y0, n, d, γ, σ, R, blocks, C, K * len_b)
#     # end

#     iclr_lazy(A_T, b, c, op_X_r, x0, y0, n, d, γ, σ, R, blocks, C, K * len_b)
# end

# S: block separation
# C:
# theta:
# implement all these things


function iclr_lazy(problem::StandardLinearProgram, exitcriterion::ExitCriterion, γ=1.0, σ=0.0, R=10, blocksize=10)  # TODO: What is γ and σ?
    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox

    d, n = size(A_T)

    x0 = zeros(d)
    y0 = zeros(n)

    blocks = []
    C = []
    len_b = n ÷ blocksize

    for i = 1:len_b
        push!(blocks, (1 + (i-1) * blocksize): (i * blocksize))
        row_set = []
        for j = 1:length(blocks[i])
            row_set = union(row_set, rowvals(A_T[:, blocks[i][j]]))
        end
        push!(C, row_set)
    end

    ##### Start of iclr_lazy

    K = exitcriterion.maxiter  #

    m = length(blocks)
    a = 1/(R * m)
    A = zeros(K+2)
    A[2] = a
    pre_a, pre_A = 0.0, 0.0
    q = zero(x0)
    idx_seq = 1:m
    x = deepcopy(x0)
    y = deepcopy(y0)
    x_tilde = zero(x0)
    z = zero(x0)
    θ = ones(Int64, length(x0))

    # Log initial measure
    starttime = time()
    results = Results()
    norm_const = norm(((x_tilde/a)' * A_T)' - b)
    logresult!(results, 1, 0.0, norm_const)

    k = 2
    exitflag = false

    while !(exitflag)
        j = rand(idx_seq)

        q_hat = q[C[j]] + (A[k] .- A[θ[C[j]]]) .* (z[C[j]] + c[C[j]])

        x_hat = prox(x0[C[j]] - 1/γ * q_hat, 1/γ * A[k])  # TODO: Check why not used
        Delta_y = γ * m * a * ((x[C[j]]' * A_T[C[j], blocks[j]])' - b[blocks[j]])
        y[blocks[j]] = y[blocks[j]] + Delta_y
        pre_a = a
        a = sqrt(1 + σ * A[k] / γ)/(R * m)

        A[k+1] = A[k] + a
        Delta_Delta_y = A_T[C[j], blocks[j]] * Delta_y

        q[C[j]] = q[C[j]] + (A[k+1] .- A[θ[C[j]] .+ 1]) .* (z[C[j]] + c[C[j]]) + (a + m * pre_a) * Delta_Delta_y

        z[C[j]] = z[C[j]] + Delta_Delta_y

        x[C[j]] =  prox(x0[C[j]] - 1/γ * q[C[j]], 1/γ * A[k])
        θ[C[j]] .= k

        x_tilde += pre_a * x

        # Record progress here: Constraint norm, time, #iteration.
        
        if k % exitcriterion.loggingfreg == 0

            norm_const = norm(((x_tilde/A[k])' * A_T)' - b)
            @info "k:$(k), ICLR constraint norm: $norm_const"
            println("k:$(k), ICLR constraint norm: $norm_const")

            elapsedtime = time() - starttime
            logresult!(results, k, elapsedtime, norm_const)

            exitflag = checkexitcondition(exitcriterion, k, elapsedtime, norm_const)
        end

        k += 1
    end

    return results
end


# function iclr_lazy(A_T, b, c, op_X_r, x0, y0, n, d, γ, σ, R, S, C, K)

#     m = length(S)
#     a = 1/(R * m)
#     A = zeros(K+2)
#     A[2] = a
#     pre_a, pre_A = 0.0, 0.0
#     q = zero(x0)
#     idx_seq = 1:m
#     x = deepcopy(x0)
#     y = deepcopy(y0)
#     x_tilde = zero(x0)
#     z = zero(x0)
#     θ = ones(Int64, length(x0))

    
#     for k = 2:K

#         j = rand(idx_seq)

#         q_hat = q[C[j]] + (A[k] .- A[θ[C[j]]]) .* (z[C[j]] + c[C[j]])

#         x_hat = op_X_r(x0[C[j]] - 1/γ * q_hat, 1/γ * A[k])  # TODO: Check why not used
#         Delta_y = γ * m * a * ((x[C[j]]' * A_T[C[j], S[j]])' - b[S[j]])
#         y[S[j]] = y[S[j]] + Delta_y
#         pre_a = a
#         a = sqrt(1 + σ * A[k] / γ)/(R * m)

#         A[k+1] = A[k] + a
#         Delta_Delta_y = A_T[C[j], S[j]] * Delta_y

#         q[C[j]] = q[C[j]] + (A[k+1] .- A[θ[C[j]] .+ 1]) .* (z[C[j]] + c[C[j]]) + (a + m * pre_a) * Delta_Delta_y

#         z[C[j]] = z[C[j]] + Delta_Delta_y

#         x[C[j]] =  op_X_r(x0[C[j]] - 1/γ * q[C[j]], 1/γ * A[k])
#         θ[C[j]] .= k

#         x_tilde += pre_a * x

#         # Record progress here: Constraint norm, time, #iteration.
        
#         if k % modulor == 0
#             # x_tilde[:] = x_tilde[:] + (A[k] .- A[θ[:]]) .* x[:]
#             norm_const = norm(((x_tilde/A[k])' * A_T)' - b)
#             @info "k:$(k), ICLR constraint norm: $norm_const"
#             modulor *= 10
#         end
#     end
#     # x_tilde[:] = x_tilde[:] + (A[K+1] .- A[θ[:]]) .* x[:]
#     return 0
# end





function iclr_lazy_restart_x(A_T, b, c, op_X_r, x0, y0, n, d, γ, σ, R, S, C, K)

    init_norm = norm((x0' * A_T)' - b)

    m = length(S)
    a = 1/(R * m)
    A = zeros(K+2)
    A[2] = a
    pre_a, pre_A = 0.0, 0.0
    q = zero(x0)
    idx_seq = 1:m
    x = deepcopy(x0)
    y = deepcopy(y0)
    x_tilde = zero(x0)
    z = zero(x0)
    θ = ones(Int64, length(x0))
    modulor = 10
    for k = 2:K

        j = rand(idx_seq)
        # @info "test 1:" q[C[j]]
        q_hat = q[C[j]] + (A[k] .- A[θ[C[j]]]) .* (z[C[j]] + c[C[j]])
        # @info q_hat
        x_hat = op_X_r(x0[C[j]] - 1/γ * q_hat, 1/γ * A[k])
        Delta_y = γ * m * a * ((x[C[j]]' * A_T[C[j], S[j]])' - b[S[j]])
        y[S[j]] = y[S[j]] + Delta_y
        pre_a = a
        a = sqrt(1 + σ * A[k] / γ)/(R * m)
        # @info "A[k]: " A[k]
        # @info "a: " a
        A[k+1] = A[k] + a
        Delta_Delta_y = A_T[C[j], S[j]] * Delta_y
        # @info "test 2:" z[C[j]]
        # @info "test 2 1:" c[C[j]]
        # @info "test 2 2:" Delta_Delta_y
        # @info "test 2 2:" A[k+1]
        q[C[j]] = q[C[j]] + (A[k+1] .- A[θ[C[j]] .+ 1]) .* (z[C[j]] + c[C[j]]) + (a + m * pre_a) * Delta_Delta_y
        x_tilde[C[j]] = x_tilde[C[j]] + (A[k] .- A[θ[C[j]]]) .* x[C[j]]
        z[C[j]] = z[C[j]] + Delta_Delta_y
        # @info "test 3:" q[C[j]]
        x[C[j]] =  op_X_r(x0[C[j]] - 1/γ * q[C[j]], 1/γ * A[k])
        θ[C[j]] .= k

        # x_tilde += pre_a * x
        # if k % modulor == 0
        #     x_tilde[:] = x_tilde[:] + (A[k] .- A[θ[:]]) .* x[:]
        #     norm_const = norm(((x_tilde/A[k])' * A_T)' - b)
        #     @info "k:$(k), ICLR constraint norm: $norm_const"
        #     modulor *= 10
        # end

        if k % (2 * m) == 0
            x_tilde[:] = x_tilde[:] + (A[k] .- A[θ[:]]) .* x[:]
            norm_const = norm(((x_tilde/A[k])' * A_T)' - b)
            if norm_const < 0.5 * init_norm
                @info "k % m: " k ÷ m
                @info "norm_const: " norm_const
                return x_tilde/A[k]
            end
        end
    end
    # x_tilde[:] = x_tilde[:] + (A[K+1] .- A[θ[:]]) .* x[:]
    return 0
end

function iclr_lazy_restart_x_y(A_T, b, c, op_X_r, x0, y0, n, d, γ, σ, R, S, C, K)

    init_norm = norm((x0' * A_T)' - b)
    m = length(S)
    a = 1/(R * m)
    A = zeros(K+2)
    A[2] = a
    pre_a, pre_A = 0.0, 0.0
    q = zero(x0)
    idx_seq = 1:m
    x = deepcopy(x0)
    y = deepcopy(y0)
    x_tilde = zero(x0)
    y_tilde = zero(y0)
    # z = zero(x0)
    z = A_T * y
    θ = ones(Int64, length(x0))
    modulor = 10
    for k = 2:K

        j = rand(idx_seq)
        # @info "test 1:" q[C[j]]
        q_hat = q[C[j]] + (A[k] .- A[θ[C[j]]]) .* (z[C[j]] + c[C[j]])
        # @info q_hat
        x_hat = op_X_r(x0[C[j]] - 1/γ * q_hat, 1/γ * A[k])
        Delta_y = γ * m * a * ((x[C[j]]' * A_T[C[j], S[j]])' - b[S[j]])
        y[S[j]] = y[S[j]] + Delta_y

        y_tilde[:] = y_tilde[:] + a * y[:]
        y_tilde[S[j]] = y_tilde[S[j]] + (m-1) * a * Delta_y[:]

        pre_a = a
        a = sqrt(1 + σ * A[k] / γ)/(R * m)
        # @info "A[k]: " A[k]
        # @info "a: " a
        A[k+1] = A[k] + a
        Delta_Delta_y = A_T[C[j], S[j]] * Delta_y
        # @info "test 2:" z[C[j]]
        # @info "test 2 1:" c[C[j]]
        # @info "test 2 2:" Delta_Delta_y
        # @info "test 2 2:" A[k+1]
        q[C[j]] = q[C[j]] + (A[k+1] .- A[θ[C[j]] .+ 1]) .* (z[C[j]] + c[C[j]]) + (a + m * pre_a) * Delta_Delta_y
        x_tilde[C[j]] = x_tilde[C[j]] + (A[k] .- A[θ[C[j]]]) .* x[C[j]]
        z[C[j]] = z[C[j]] + Delta_Delta_y
        # @info "test 3:" q[C[j]]
        x[C[j]] =  op_X_r(x0[C[j]] - 1/γ * q[C[j]], 1/γ * A[k])
        θ[C[j]] .= k

        # x_tilde += pre_a * x
        # if k % modulor == 0
        #     x_tilde[:] = x_tilde[:] + (A[k] .- A[θ[:]]) .* x[:]
        #     norm_const = norm(((x_tilde/A[k])' * A_T)' - b)
        #     @info "k:$(k), ICLR constraint norm: $norm_const"
        #     modulor *= 10
        # end

        if k % (2 * m) == 0
            x_tilde_tmp = x_tilde[:] + (A[k] .- A[θ[:]]) .* x[:]
            norm_const = norm(((x_tilde_tmp/A[k])' * A_T)' - b)
            func_value = c' * (x_tilde_tmp/A[k])
            if norm_const < 0.5 * init_norm
                @info "k ÷ m: $(k ÷ m)"
                @info "norm_const: $(norm_const)"
                @info "func_value: $(func_value-500)"
                return x_tilde_tmp/A[k], y_tilde/A[k]
            end
        end
    end
    # x_tilde[:] = x_tilde[:] + (A[K+1] .- A[θ[:]]) .* x[:]
    return 0
end


function PD_norm(A_T, b, c, x, y)
    v1 = max.(c - A_T * y, 0.0)
    v2 = (x' * A_T)' - b
    # print(vcat(v1, v2))
    # print(typeof(vcat(v1, v2)))
    return norm(vcat(v1, v2))
end


function iclr_lazy_restart_x_y_v2(A_T, b, c, op_X_r, x0, y0, n, d, γ, σ, R, S, C, K)

    init_norm = PD_norm(A_T, b, c, x0, y0)
    m = length(S)
    a = 1/(R * m)
    A = zeros(K+2)
    A[2] = a
    pre_a, pre_A = 0.0, 0.0
    q = zero(x0)
    idx_seq = 1:m
    x = deepcopy(x0)
    y = deepcopy(y0)
    x_tilde = zero(x0)
    y_tilde = zero(y0)
    # z = zero(x0)
    z = A_T * y
    θ = ones(Int64, length(x0))
    modulor = 10
    for k = 2:K

        j = rand(idx_seq)
        # @info "test 1:" q[C[j]]
        q_hat = q[C[j]] + (A[k] .- A[θ[C[j]]]) .* (z[C[j]] + c[C[j]])
        # @info q_hat
        x_hat = op_X_r(x0[C[j]] - 1/γ * q_hat, 1/γ * A[k])
        Delta_y = γ * m * a * ((x[C[j]]' * A_T[C[j], S[j]])' - b[S[j]])
        y[S[j]] = y[S[j]] + Delta_y

        y_tilde[:] = y_tilde[:] + a * y[:]
        y_tilde[S[j]] = y_tilde[S[j]] + (m-1) * a * Delta_y[:]

        pre_a = a
        a = sqrt(1 + σ * A[k] / γ)/(R * m)
        # @info "A[k]: " A[k]
        # @info "a: " a
        A[k+1] = A[k] + a
        Delta_Delta_y = A_T[C[j], S[j]] * Delta_y
        # @info "test 2:" z[C[j]]
        # @info "test 2 1:" c[C[j]]
        # @info "test 2 2:" Delta_Delta_y
        # @info "test 2 2:" A[k+1]
        q[C[j]] = q[C[j]] + (A[k+1] .- A[θ[C[j]] .+ 1]) .* (z[C[j]] + c[C[j]]) + (a + m * pre_a) * Delta_Delta_y
        x_tilde[C[j]] = x_tilde[C[j]] + (A[k] .- A[θ[C[j]]]) .* x[C[j]]
        z[C[j]] = z[C[j]] + Delta_Delta_y
        # @info "test 3:" q[C[j]]
        x[C[j]] =  op_X_r(x0[C[j]] - 1/γ * q[C[j]], 1/γ * A[k])
        θ[C[j]] .= k

        # x_tilde += pre_a * x
        # if k % modulor == 0
        #     x_tilde[:] = x_tilde[:] + (A[k] .- A[θ[:]]) .* x[:]
        #     norm_const = norm(((x_tilde/A[k])' * A_T)' - b)
        #     @info "k:$(k), ICLR constraint norm: $norm_const"
        #     modulor *= 10
        # end

        if k % (2 * m) == 0
            x_tilde_tmp = x_tilde[:] + (A[k] .- A[θ[:]]) .* x[:]
            # norm_const = norm(((x_tilde/A[k])' * A_T)' - b)
            norm_const = PD_norm(A_T, b, c, x_tilde_tmp/A[k], y_tilde/A[k])

            if norm_const < 0.5 * init_norm
                @info "k % m: " k ÷ m
                @info "norm_const: " norm_const
                return x_tilde_tmp/A[k], y_tilde/A[k]
            end
        end
    end
    # x_tilde[:] = x_tilde[:] + (A[K+1] .- A[θ[:]]) .* x[:]
    return 0
end
