
function iclr_lazy_restart_x_y(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    γ=1.0, σ=0.0, R=10, blocksize=10)

    @info("Running iclr_lazy_restart_x_y...")

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

    m = length(blocks)

    # Log initial measure
    starttime = time()
    results = Results()
    init_norm_const = norm((x0' * A_T)' - b)
    init_fvalue = c' * x0
    logresult!(results, 1, 0.0, init_fvalue, init_norm_const)

    outer_k = 2
    exitflag = false

    while !exitflag
        # Init of ICLR_Lazy
        a = 1 / (R * m)
        pre_a = a
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
            Adelta_sliced = a * (k .- θ_x[C[j]])

            # Line 5
            q_hat = q_sliced + Adelta_sliced .* z_sliced

            # Line 6
            x_hat = prox(x0[C[j]] - 1/γ * q_hat, 1/γ * a * k)

            # Line 7 & 12
            sliced_A_T = sliced_A_Ts[j]
            Delta_y = γ * m * a * ((x_hat' * sliced_A_T)' - b[blocks[j]])
            y_tilde[blocks[j]] = y_tilde[blocks[j]] + a * (k .- θ_y[blocks[j]]) .* y[blocks[j]]
            y_tilde[blocks[j]] = y_tilde[blocks[j]] + (m-1) * a * Delta_y[:]
            y[blocks[j]] = y[blocks[j]] + Delta_y
            
            # Line 8
            pre_a = a

            # Line 10
            Delta_Delta_y = sliced_A_T * Delta_y
            q[C[j]] = q_sliced + Adelta_sliced .* z_sliced + (a + m * pre_a) * Delta_Delta_y

            # Line 11
            x_tilde[C[j]] = x_tilde[C[j]] + a * (k .- θ_x[C[j]]) .* x[C[j]]

            # Line 9
            z[C[j]] = z[C[j]] + Delta_Delta_y

            # Line 13
            x[C[j]] =  prox(x0[C[j]] - 1/γ * q[C[j]], 1/γ * k * a)

            # Line 14
            θ_x[C[j]] .= k
            θ_y[blocks[j]] .= k

            # Logging and checking exit condition
            # set restartflag when reached some measure
            if k % (exitcriterion.loggingfreq * m) == 0
                x_tilde_tmp = x_tilde[:] + a * (k .- θ_x[:]) .* x[:]
                y_tilde_tmp = y_tilde[:] + a * (k .- θ_y[:]) .* y[:]
                x_out = x_tilde_tmp / (a * k)
                y_out = y_tilde_tmp / (a * k)
                norm_const = norm((x_out' * A_T)' - b)
                func_value = c' * x_out

                elapsedtime = time() - starttime
                @info "elapsedtime: $elapsedtime"
                @info "outer_k: $(outer_k), constraint norm: $norm_const, func value: $func_value"
                logresult!(results, outer_k, elapsedtime, func_value, norm_const)

                exitflag = checkexitcondition(exitcriterion, outer_k, elapsedtime, norm_const)
                if exitflag
                    break
                end

                if norm_const < 0.5 * init_norm_const
                    @info "<===== RESTARTING"
                    @info "k ÷ m: $(k ÷ m)"
                    
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
