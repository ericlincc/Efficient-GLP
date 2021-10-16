
function pdhg_restart_x_y(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    γ=1.0, L=10, restartfreq=Inf)

    @info("Running pdhg_restart_x_y...")
    @info("γ = $(γ)")
    @info("L = $(L)")
    @info("restartfreq = $(restartfreq)")

    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox

    d, n = size(A_T)
    x0 = zeros(d)
    y0 = zeros(n)
    m = 1

    ##### Start of pdhg_restart

    # Log initial measure
    starttime = time()
    results = Results()
    init_fvaluegap, init_metricLP = compute_fvaluegap_metricLP(x0, y0, problem)
    logresult!(results, 1, 0.0, init_fvaluegap, init_metricLP)

    outer_k = 1
    exitflag = false

    while !exitflag
        # Init of PDHG
        τ, σ =  1.0 / (γ * L), 1.0 * γ / L
        x_bar = deepcopy(x0)
        x = deepcopy(x0)
        y = deepcopy(y0)
        x_tilde = zero(x0)
        y_tilde = zero(y0)

        k = 1
        restartflag = false
        while !exitflag && !restartflag

            y[:] = y[:] + σ * ((x_bar' * A_T)' - b)
            x_pre = deepcopy(x)
            x[:] = x[:]  - τ * (A_T * y[:] + c[:])
            x = prox(x, τ)
            x_bar[:] = 2 * x[:] -  x_pre[:]  # Correct term from original paper
            # x_bar[:] = x_bar[:] + (x[:] -  x_pre[:])  # Wrong term from CB
            x_tilde[:] = x_tilde[:] + x[:]
            y_tilde[:] = y_tilde[:] + y[:]

            # Logging and checking exit condition
            # set restartflag when reached some measure
            if outer_k % exitcriterion.loggingfreq == 0
                x_out = x_tilde / (k - 1)
                y_out = y_tilde / (k - 1)
                
                # Progress measures
                fvaluegap, metricLP = compute_fvaluegap_metricLP(x_out, y_out, problem)

                elapsedtime = time() - starttime
                # @info "elapsedtime: $elapsedtime"
                # @info "outer_k: $(outer_k), fvaluegap: $(fvaluegap), metricLP: $(metricLP)"
                elapsedtime = time() - starttime
                logresult!(results, outer_k, elapsedtime, fvaluegap, metricLP)

                exitflag = checkexitcondition(exitcriterion, outer_k, elapsedtime, metricLP)
                if exitflag
                    break
                end

                if k >= restartfreq * m || (restartfreq == Inf && metricLP <= 0.5 * init_metricLP)
                    @info "<===== RESTARTING"
                    @info "k ÷ m: $(k ÷ m)"
                    @info "elapsedtime: $elapsedtime"
                    @info "outer_k: $(outer_k), fvaluegap: $(fvaluegap), metricLP: $(metricLP)"

                    x0, y0 = deepcopy(x_out), deepcopy(y_out)
                    init_fvaluegap = fvaluegap
                    init_metricLP = metricLP
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
