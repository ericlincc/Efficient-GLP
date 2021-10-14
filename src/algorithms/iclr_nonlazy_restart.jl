
function iclr_nonlazy_restart_x_y(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    γ=1.0, σ=0.0, R=10, blocksize=10)

    @info("Running iclr_nonlazy_restart_x_y...")

    # Algorithm 1 from the paper

    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox

    d, n = size(A_T)
    x0 = zeros(d)
    y0 = zeros(n)

    _time1 = time()
    # Precomputing blocks, nzrows, sliced_A_T
    blocks, _ = compute_nzrows_for_blocks(A_T, blocksize)
    _time2 = time()
    @info ("Initialization time = ", _time2 - _time1)

    ##### Start of iclr_nonlazy_restart_x_y

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
        # Init of ICLR_Nonlazy
        a = 1 / (R * m)
        A = a
        pre_a, pre_A = 0.0, 0.0
        idx_seq = 1:m

        x = deepcopy(x0)
        y = deepcopy(y0)
        x_tilde = zero(x0)
        y_tilde = zero(y0)
        z = A_T * y0
        q = a * (z + c)

        k = 2
        restartflag = false

        while !exitflag && !restartflag

            # Line 4
            x[:] = x0[:] - 1.0/γ * q[:]
            x = prox(x, 1.0/γ * A)

            # Line 5
            j = rand(idx_seq)

            # Line 6
            Delta_y = γ * m * a * ((x' * A_T[:, blocks[j]])' - b[blocks[j]])
            y[blocks[j]] = y[blocks[j]] + Delta_y

            # Line 7
            pre_a, pre_A = a, A
            a = sqrt(1 + σ * A / γ)/(R * m)
            A = A + a

            # Line 8 & 9
            Delta_Delta_y = A_T[:, blocks[j]] * Delta_y
            z[:] = z[:] + Delta_Delta_y
            q[:] = q[:] + a * (z + c) + m * pre_a * Delta_Delta_y

            x_tilde += pre_a * x
            y_tilde += pre_a * y
            y_tilde[blocks[j]] += pre_a * (m - 1) * Delta_y 

            # Logging and checking exit condition
            # set restartflag when reached some measure
            if k % (exitcriterion.loggingfreq * m) == 0

                x_out = x_tilde / A
                y_out = y_tilde / A
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
                    @info "elapsedtime: $elapsedtime"
                    @info "outer_k: $(outer_k), constraint norm: $norm_const, func value: $func_value"

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
