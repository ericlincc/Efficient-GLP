
function purecd_restart_x_y(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    γ=1.0, σ=0.0, R=10, blocksize=10, restartfreq=Inf,
    io=nothing)

    @info("Running pure_cd_restart_x_y...")
    @info("blocksize = $(blocksize)")
    @info("γ = $(γ)")
    @info("σ = $(σ)")
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

    _time1 = time()
    # Precomputing blocks, nzrows, sliced_A_T
    blocks, C = compute_nzrows_for_blocks(A_T, blocksize)
    num_nnz = nnz_per_row(A_T)
    sliced_A_Ts = Array{SparseMatrixCSC{Float64, Int}}([])
    for j in 1:length(C)
        push!(sliced_A_Ts, A_T[C[j], blocks[j]])
    end
    _time2 = time()
    @info ("Initialization time = ", _time2 - _time1)

    ##### Start of PURE_CD

    m = length(blocks)

    # Log initial measure
    starttime = time()
    results = Results()
    init_fvaluegap, init_metricLP = compute_fvaluegap_metricLP(x0, y0, problem)
    @info "init_metricLP: $(init_metricLP)" 

    logresult!(results, 1, 0.0, init_fvaluegap, init_metricLP)

    outer_k = 1
    exitflag = false

    while !exitflag
        # Init of PURE_CD
        τ, σ =  0.99 /(γ *  R) , 0.99 * γ/ R
        idx_seq = 1:m

        x = deepcopy(x0)
        y = deepcopy(y0)
        x_bar = deepcopy(x0)
        x_tilde = zero(x0)
        y_tilde = zero(y0)

        z = A_T * y
        θ_x = ones(Int, length(x0))
        θ_y = ones(Int, length(y0))

        k = 1
        restartflag = false
        while !exitflag && !restartflag
            j = rand(idx_seq)

            x_tilde[C[j]] = x_tilde[C[j]] + (k .- θ_x[C[j]]) .* x[C[j]]
            y_tilde[blocks[j]] = y_tilde[blocks[j]] +  (k.- θ_y[blocks[j]]) .* y[blocks[j]]


            x_bar[C[j]] = x[C[j]] - (τ ./ num_nnz[C[j]]) .* (z[C[j]] + c[C[j]])
            x_bar[C[j]] = prox(x_bar[C[j]], τ)

            sliced_A_T = sliced_A_Ts[j]
            Delta_y = σ * ((x_bar[C[j]]' * sliced_A_T)' - b[blocks[j]])
            y[blocks[j]] = y[blocks[j]] + Delta_y[:]
            tmp = sliced_A_T * Delta_y
            z[C[j]] = z[C[j]] + tmp[:]
            x[C[j]] = x_bar[C[j]] -  τ .* tmp[:]

            θ_x[C[j]] .= k
            θ_y[blocks[j]] .= k

            # Logging and checking exit condition
            # set restartflag when reached some measure
            if outer_k % (exitcriterion.loggingfreq * m) == 0
                x_tilde_tmp = x_tilde[:] +  ((k+1) .- θ_x[:]) .* x_bar[:]
                y_tilde_tmp = y_tilde[:] +  ((k+1) .- θ_y[:]) .* y[:]
                x_out = x_tilde_tmp / k
                y_out = y_tilde_tmp / k

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

                if !isnothing(io)
                    flush(io)
                end
            end

            k += 1
            outer_k += 1
        end
    end

    return results
end
