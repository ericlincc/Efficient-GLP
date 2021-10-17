
function spdhg_restart_x_y(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    γ=1.0, R=10, blocksize=10, restartfreq=Inf,
    io=nothing)

    @info("Running spdhg_restart_x_y...")
    @info("blocksize = $(blocksize)")
    @info("γ = $(γ)")
    @info("R = $(R)")
    @info("restartfreq = $(restartfreq)")
    if !isnothing(io)
        flush(io)
    end

    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox

    d, n = size(A_T)
    x0 = zeros(d)
    y0 = zeros(n)
    grad = zero(x0)

    _time1 = time()
    # Precomputing blocks, nzrows, sliced_A_T
    blocks, C = compute_nzrows_for_blocks(A_T, blocksize)
    sliced_A_Ts = Array{SparseMatrixCSC{Float64, Int}}([])
    for j in 1:length(C)
        push!(sliced_A_Ts, A_T[C[j], blocks[j]])
    end
    _time2 = time()
    @info ("Initialization time = ", _time2 - _time1)

    ##### Start of spdhg_restart_x_y

    m = length(blocks)

    # Log initial measure
    starttime = time()
    results = Results()
    init_fvaluegap, init_metricLP = compute_fvaluegap_metricLP(x0, y0, problem)
    logresult!(results, 1, 0.0, init_fvaluegap, init_metricLP)

    outer_k = 1
    exitflag = false

    while !exitflag
        # Init of SPDHG
        τ, σ =  1.0 / (γ * m * R) , 1.0 * γ / R
        idx_seq = 1:m

        x = deepcopy(x0)
        y = deepcopy(y0)
        x_tilde = zero(x0)
        y_tilde = zero(y0)

        z = A_T * y
        θ_y = ones(Int, length(y0))

        k = 1
        restartflag = false
        while !exitflag && !restartflag

            x[:] = x[:] - τ * (grad[:] + c[:])
            x = prox(x, τ)
            j = rand(idx_seq)
            sliced_A_T = sliced_A_Ts[j]

            Delta_y = σ * ((x[C[j]]' * sliced_A_T)' - b[blocks[j]])

            y_tilde[blocks[j]] = y_tilde[blocks[j]] + τ * (k .- θ_y[blocks[j]]) .* y[blocks[j]]
            y_tilde[blocks[j]] = y_tilde[blocks[j]] + (m-1) * τ * Delta_y[:]

            y[blocks[j]] = y[blocks[j]] + Delta_y[:]

            Delta_Delta_y = A_T[:, blocks[j]] * Delta_y
            z[:] = z[:] + Delta_Delta_y[:]
            grad[:] = z[:] + m * Delta_Delta_y[:]

            x_tilde[:] = x_tilde[:] + x[:]
            θ_y[blocks[j]] .= k

            # Logging and checking exit condition
            # set restartflag when reached some measure
            if outer_k % (exitcriterion.loggingfreq * m) == 0
                y_tilde_tmp = y_tilde[:] + τ * ((k+1) .- θ_y[:]) .* y[:]
                x_out = x_tilde / k
                y_out = y_tilde_tmp / (τ * k)

                # Progress measures
                fvaluegap, metricLP = compute_fvaluegap_metricLP(x_out, y_out, problem)

                elapsedtime = time() - starttime
                @info "elapsedtime: $elapsedtime"
                @info "outer_k: $(outer_k), fvaluegap: $(fvaluegap), metricLP: $(metricLP)"
                elapsedtime = time() - starttime
                logresult!(results, outer_k, elapsedtime, fvaluegap, metricLP)
                if !isnothing(io)
                    flush(io)
                end

                exitflag = checkexitcondition(exitcriterion, outer_k, elapsedtime, metricLP)
                if exitflag
                    break
                end
                
                if k >= restartfreq * m || (restartfreq == Inf && metricLP <= 0.5 * init_metricLP)
                    @info "<===== RESTARTING"
                    @info "k ÷ m: $(k ÷ m)"
                    @info "elapsedtime: $elapsedtime"
                    @info "outer_k: $(outer_k), fvaluegap: $(fvaluegap), metricLP: $(metricLP)"
                    if !isnothing(io)
                        flush(io)
                    end

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
