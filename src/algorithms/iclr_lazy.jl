# the problem to optimize
#  \min_{\vx\in\gX\subset\sR^d} \max_{\vy\in\sR^n} \Big\{ L(\vx,\vy) = \vc^T\vx + r(\vx)   + \vy^T\mA\vx   - \vy^T\vb\Big\}.
# op_X_r: (I + tau_n ∂G)^{-1}
# γ: strong convexity
# σ, τ, θ: sequence for update
# L: Lipschitz constant
# to be consistent with the sparse column format, we consider
# A_T: R^{d×n}
# TODO: Update the above comments


function iclr_lazy(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    γ=1.0, σ=0.0, R=10, blocksize=10)  # TODO: What is γ and σ?

    # Algorithm 2 from the paper

    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox

    d, n = size(A_T)

    _time1 = time()
    # Precomputing blocks, nzrows, sliced_A_T
    blocks, C = compute_nnrows_for_blocks(A_T, blocksize)
    sliced_A_Ts = Array{SparseMatrixCSC{Float64, Int}}([])
    for j in 1:length(C)
        push!(sliced_A_Ts, A_T[C[j], blocks[j]])
    end
    _time2 = time()
    @info ("Initialization time = ", _time2 - _time1)

    ##### Start of iclr_lazy #####

    # Init of ICLR_Lazy
    K = exitcriterion.maxiter  #
    m = length(blocks)
    a = 1/(R * m)
    A = zeros(K+2)
    A[2] = a
    pre_a = 0.0
    idx_seq = 1:m

    x0 = zeros(d)
    y0 = zeros(n)
    q = - γ * deepcopy(x0)
    x = deepcopy(x0)
    y = deepcopy(y0)
    x_tilde = zero(x0)
    z = deepcopy(c)
    θ = ones(Int64, length(x0))

    # Log initial measure
    starttime = time()
    results = Results()
    init_norm_const = norm((x0' * A_T)' - b)
    logresult!(results, 1, 0.0, init_norm_const)

    k = 2
    exitflag = false

    # Start iterations
    while !exitflag
        # Line 4
        j = rand(idx_seq)

        # Slice of variables based on nzrowsC[j]
        z_sliced = z[C[j]]
        q_sliced = q[C[j]]
        Adelta_sliced = A[k] .- A[θ[C[j]]]

        # Line 5
        q_hat = q_sliced + Adelta_sliced .* z_sliced

        # Line 6
        x_hat = prox(- 1/γ * q_hat, 1/γ * A[k])

        # Line 7
        sliced_A_T = sliced_A_Ts[j]
        Delta_y = γ * m * a * ((x_hat' * sliced_A_T)' - b[blocks[j]])
        y[blocks[j]] = y[blocks[j]] + Delta_y

        # Line 8
        pre_a = a
        a = sqrt(1 + σ * A[k] / γ)/(R * m)
        A[k+1] = A[k] + a

        # Line 10
        Delta_Delta_y = sliced_A_T * Delta_y
        q[C[j]] = q_sliced + Adelta_sliced .* z_sliced + (a + m * pre_a) * Delta_Delta_y

        # Line 11
        x_tilde[C[j]] = x_tilde[C[j]] + (A[k] .- A[θ[C[j]]]) .* x[C[j]]

        # Line 9  TODO: Check with CB about the ordering
        z[C[j]] = z[C[j]] + Delta_Delta_y

        # Line 13
        x[C[j]] =  prox(x0[C[j]] - 1/γ * q[C[j]], 1/γ * A[k])

        # Line 14
        θ[C[j]] .= k

        # Recording progress
        if k % exitcriterion.loggingfreg == 0
            x_tilde_tmp = x_tilde[:] + (A[k] .- A[θ[:]]) .* x[:]
            norm_const = norm(((x_tilde_tmp/A[k])' * A_T)' - b)

            elapsedtime = time() - starttime
            @info "k: $(k), ICLR constraint norm: $norm_const, elapsedtime: $elapsedtime"
            logresult!(results, k, elapsedtime, norm_const)

            exitflag = checkexitcondition(exitcriterion, k, elapsedtime, norm_const)
        end

        k += 1
    end

    return results, profilings
end
