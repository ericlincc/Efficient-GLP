
function pure_cd_restart_x_y(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    γ=1.0, σ=0.0, R=10, blocksize=10)

    @info("Running PURE_CD...")

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

    ##### Start of PURE_CD

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
        # Init of PURE_CD
        τ, σ =  1.0 / (R * m), 1.0 / R
        idx_seq = 1:m

        x = deepcopy(x0)
        y = deepcopy(y0)
        x_tilde = zero(x0)
        y_tilde = zero(y0)

        z = A_T * y
        θ_x = ones(Int, length(x0))
        θ_y = ones(Int, length(y0))

        k = 2
        restartflag = false
        while !exitflag && !restartflag
            j = rand(idx_seq)

            x_bar = x[C[j]] - τ * (z[C[j]] + c[C[j]])
            x_bar = prox(x_bar, τ)

            sliced_A_T = sliced_A_Ts[j]
            Delta_y = -y[blocks[j]]
            y_tilde[blocks[j]] = y_tilde[blocks[j]] + τ * (k .- θ_y[blocks[j]]) .* y[blocks[j]]
            y[blocks[j]] = y[blocks[j]] + σ * ((x_bar' * sliced_A_T)' - b[blocks[j]])
            Delta_y += y[blocks[j]]
            y_tilde[blocks[j]] = y_tilde[blocks[j]] + (m-1) * τ * Delta_y[:]
            tmp = sliced_A_T * Delta_y
            z[C[j]] = z[C[j]] + tmp
            x_tilde[C[j]] = x_tilde[C[j]] + τ * (k .- θ_x[C[j]]) .* x[C[j]]
            x[C[j]] = x_bar - τ * m * tmp

            θ_x[C[j]] .= k
            θ_y[blocks[j]] .= k

            # Logging and checking exit condition
            # set restartflag when reached some measure
            if k % (exitcriterion.loggingfreq * m) == 0
                x_tilde_tmp = x_tilde[:] + τ * (k .- θ_x[:]) .* x[:]
                y_tilde_tmp = y_tilde[:] + τ * (k .- θ_y[:]) .* y[:]
                x_out = x_tilde_tmp / (τ * k)
                y_out = y_tilde_tmp / (τ * k)
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
