
function iclr_lazy_restart_x_y(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    γ=1.0, σ=0.0, R=10, blocksize=10)

    # Algorithm 2 from the paper

    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox

    d, n = size(A_T)
    x0 = zeros(d)
    y0 = zeros(n)

    _time1 = time()
    # Precomputing blocks, nzrows, sliced_A_T
    blocks, C = compute_nzrows_for_blocks(A_T, blocksize)
    sliced_A_Ts = Array{SparseMatrixCSC{Float64, Int}}([])
    for j in 1:length(C)
        push!(sliced_A_Ts, A_T[C[j], blocks[j]])
    end
    _time2 = time()
    @info ("Initialization time = ", _time2 - _time1)

    ##### Start of iclr_lazy_restart_x_y

    K = exitcriterion.maxiter  #
    m = length(blocks)

    # Log initial measure
    starttime = time()
    results = Results()
    init_norm_const = norm((x0' * A_T)' - b)
    logresult!(results, 1, 0.0, init_norm_const)

    outer_k = 2
    exitflag = false

    while !exitflag
        # Init of ICLR_Lazy
        a = 1 / (R * m)
        A = zeros(K + 2)  # TODO: Can be more efficient not creating A every restart
        A[2] = a
        pre_a = 0.0
        idx_seq = 1:m

        q = zero(x0)
        x = deepcopy(x0)
        y = deepcopy(y0)
        x_tilde = zero(x0)
        y_tilde = zero(y0)

        z = A_T * y + c
        θ_x = ones(Int, length(x0))
        θ_y = ones(Int, length(y0))


        k = 2
        restartflag = false
        while !exitflag && !restartflag
            # Line 4
            j = rand(idx_seq)

            # Slice of variables based on nzrowsC[j]
            z_sliced = z[C[j]]
            q_sliced = q[C[j]]
            Adelta_sliced = A[k] .- A[θ_x[C[j]]]

            # Line 5
            q_hat = q_sliced + Adelta_sliced .* z_sliced

            # Line 6
            x_hat = prox(x0[C[j]] - 1/γ * q_hat, 1/γ * A[k])

            # Line 7 & 12
            sliced_A_T = sliced_A_Ts[j]
            Delta_y = γ * m * a * ((x_hat' * sliced_A_T)' - b[blocks[j]])
            y_tilde[blocks[j]] = y_tilde[blocks[j]] + (A[k] .- A[θ_y[blocks[j]]]) .* y[blocks[j]]
            y_tilde[blocks[j]] = y_tilde[blocks[j]] + (m-1) * a * Delta_y[:]
            y[blocks[j]] = y[blocks[j]] + Delta_y

            # y_tilde[:] = y_tilde[:] + a * y[:]
            # y_tilde[blocks[j]] = y_tilde[blocks[j]] + (m-1) * a * Delta_y[:]
            
            # Line 8
            pre_a = a
            a = sqrt(1 + σ * A[k] / γ)/(R * m)
            A[k+1] = A[k] + a

            # Line 10
            Delta_Delta_y = sliced_A_T * Delta_y
            q[C[j]] = q_sliced + Adelta_sliced .* z_sliced + (a + m * pre_a) * Delta_Delta_y

            # Line 11
            x_tilde[C[j]] = x_tilde[C[j]] + (A[k] .- A[θ_x[C[j]]]) .* x[C[j]]

            # Line 9  TODO: Check with CB about the ordering
            z[C[j]] = z[C[j]] + Delta_Delta_y

            # Line 13
            x[C[j]] =  prox(x0[C[j]] - 1/γ * q[C[j]], 1/γ * A[k])

            # Line 14
            θ_x[C[j]] .= k
            θ_y[blocks[j]] .= k

            # Logging and checking exit condition
            if outer_k % exitcriterion.loggingfreg == 0
                x_tilde_tmp = x_tilde[:] + (A[k] .- A[θ_x[:]]) .* x[:]
                norm_const = norm(((x_tilde_tmp/A[k])' * A_T)' - b)
                elapsedtime = time() - starttime
                # @info "outer_k: $(outer_k), ICLR constraint norm: $norm_const, elapsedtime: $elapsedtime"

                logresult!(results, outer_k, elapsedtime, norm_const)
    
                exitflag = checkexitcondition(exitcriterion, outer_k, elapsedtime, norm_const)
                if exitflag
                    break
                end
            end

            # set restartflag when reached some measure
            if k % (2 * m) == 0
                x_tilde_tmp = x_tilde[:] + (A[k] .- A[θ_x[:]]) .* x[:]
                y_tilde_tmp = y_tilde[:] + (A[k] .- A[θ_y[:]]) .* y[:]
                x_out = x_tilde_tmp / A[k]
                y_out = y_tilde_tmp / A[k]
                # y_out = y_tilde / A[k]
                norm_const = norm((x_out' * A_T)' - b)
                if norm_const < 0.5 * init_norm_const
                    @info "restarting"
                    @info "k ÷ m: $(k ÷ m)"
                    @info "norm_const: " norm_const
                    
                    x0, y0 = deepcopy(x_out), deepcopy(y_out)
                    init_norm_const = norm_const
                    restartflag = true
                    break
                end
            end

            k += 1
            outer_k += 1
        end
    end

    return results
end
