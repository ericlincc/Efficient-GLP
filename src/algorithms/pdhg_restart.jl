function pdhg_restart_x_y(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    L=10)

    @info("Running pdhg_restart_x_y...")

    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox

    d, n = size(A_T)
    x0 = zeros(d)
    y0 = zeros(n)

    ##### Start of pdhg_restart

    # Log initial measure
    starttime = time()
    results = Results()
    init_norm_const = norm((x0' * A_T)' - b)
    init_fvalue = c' * x0
    logresult!(results, 1, 0.0, init_fvalue, init_norm_const)

    outer_k = 2
    exitflag = false

    while !exitflag
        # Init of PDHG
        τ, σ =  1.0 / L , 1.0 / L
        x_bar = deepcopy(x0)
        x = deepcopy(x0)
        y = deepcopy(y0)
        x_tilde = zero(x0)
        y_tilde = zero(y0)

        k = 2
        restartflag = false
        while !exitflag && !restartflag

            y[:] = y[:] + σ * ((x_bar' * A_T)' - b)
            x_pre = deepcopy(x)
            x[:] = x[:]  - τ * (A_T * y[:] + c[:])
            x = prox(x, τ)
            # x_bar[:] = 2*x[:] -  x_pre[:]  # Correct term from original paper
            x_bar[:] = x_bar[:] + (x[:] -  x_pre[:])  # Wrong term from CB
            x_tilde[:] = x_tilde[:] + x[:]
            y_tilde[:] = y_tilde[:] + y[:]

            # Logging and checking exit condition
            # set restartflag when reached some measure
            if outer_k % exitcriterion.loggingfreq == 0
                x_out = x_tilde / (k - 1)
                y_out = y_tilde / (k - 1)
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
                    @info "k: $(k)"
                    
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
