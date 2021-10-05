
# using LinearAlgebra
# # the problem to optimize
# #  \min_{\vx\in\gX\subset\sR^d} \max_{\vy\in\sR^n} \Big\{ L(\vx,\vy) = \vc^T\vx + r(\vx)   + \vy^T\mA\vx   - \vy^T\vb\Big\}.
# # op_X_r: (I + tau_n ∂G)^{-1}
# # γ: strong convexity
# # σ, τ, θ: sequence for update
# # L: Lipschitz constant

# # to be consistent with the sparse column format, we consider
# # D: R^{d×n}
# function test_spdhg()

#     n, d = 1000, 500
#     γ = 0.0
#     D = (randn(d, n) +  ones(d, n)) /√n
#     # b = randn(n)
#     x_star = ones(d)
#     b = (x_star' * D)'
#     c = ones(d)
#     R = 1.0

#     K = 100000
#     x0 = zeros(d)
#     y0 = zeros(n)
#     blocks = []
#     bs = 1
#     len_b = n ÷ bs
#     for i = 1:len_b
#         push!(blocks, (1 + (i-1) * bs): (i * bs))
#     end
#     τ, σ =  1.0 / (R * len_b), 1.0 / R
#     op_X_r(x, τ) = max.(x, 0.0)
#     spdhg(D, b, c, op_X_r, x0, y0, n, d, γ, σ, τ, R, blocks, K * len_b)
# end

# # blocks: the separation of dual variables
# # R: row norm

function spdhg(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    R=1.0, blocksize=1)

    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox

    d, n = size(A_T)
    τ, σ =  1.0 / (R * (n ÷ blocksize)), 1.0 / R

    _time1 = time()
    # Precomputing blocks
    blocks, _ = compute_nzrows_for_blocks(A_T, blocksize)
    _time2 = time()
    @info ("Initialization time = ", _time2 - _time1)   

    x0 = zeros(d)
    y0 = zeros(n)
    z = zero(x0)  # TODO: Why not just zeros(d)?
    grad = zero(x0)
    x_tilde = zero(x0)
    x = deepcopy(x0)
    y = y0
    idx_seq = 1:length(blocks)
    len_b = length(blocks)

    # Log initial measure
    starttime = time()
    results = Results()
    init_norm_const = norm((x0' * A_T)' - b)
    logresult!(results, 1, 0.0, init_norm_const)

    k = 1
    exitflag = false

    while !exitflag
        x[:] = x[:] - τ * (grad[:] + c[:])
        x = prox(x, τ)
        j = rand(idx_seq)

        Delta_y = -y[blocks[j]]
        y[blocks[j]] = y[blocks[j]] + σ * ((x' * A_T[:, blocks[j]])' - b[blocks[j]])
        Delta_y += y[blocks[j]]
        tmp = A_T[:, blocks[j]] * Delta_y
        z[:] = z[:] + tmp[:]
        grad[:] = z[:] + len_b * tmp[:]
        x_tilde[:] = x_tilde[:] + x[:]

        # Recording progress
        if k % exitcriterion.loggingfreg == 0
            norm_const = norm(((x_tilde/k)' * A_T)' - b)

            elapsedtime = time() - starttime
            @info "k: $(k), SPDHG constraint norm: $norm_const, elapsedtime: $elapsedtime"
            logresult!(results, k, elapsedtime, norm_const)

            exitflag = checkexitcondition(exitcriterion, k, elapsedtime, norm_const)
        end

        k += 1
    end
end
