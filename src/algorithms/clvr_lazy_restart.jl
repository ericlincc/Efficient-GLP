# Copyright 2021 The CLVR Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This is the restart version of Coordinate Linear Variance Reduction (Lazy Update Version).
"""
function clvr_lazy_restart_x_y(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    γ=1.0, σ=0.0, R=10, blocksize=10, restartfreq=Inf,
    io=nothing)

    @info("Running clvr_lazy_restart_x_y with")
    @info("blocksize = $(blocksize)")
    @info("γ = $(γ)")
    @info("σ = $(σ)")
    @info("R = $(R)")
    @info("restartfreq = $(restartfreq)")
    if !isnothing(io)
        flush(io)
    end

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

    ##### Start of clvr_lazy_restart_x_y

    m = length(blocks)

    # Log initial measure
    starttime = time()
    results = Results()
    init_fvaluegap, init_metricLP = compute_fvaluegap_metricLP(x0, y0, problem)
    logresult!(results, 1, 0.0, init_fvaluegap, init_metricLP)

    outer_k = 1
    exitflag = false

    while !exitflag
        # Init of CLVR Lazy
        a = 1 / (R * m)
        pre_a = a
        idx_seq = 1:m

        x = deepcopy(x0)
        y = deepcopy(y0)
        x_tilde = zero(x0)
        y_tilde = zero(y0)
        z = A_T * y + c
        q = a * deepcopy(z)
        θ_x = ones(Int, length(x0))
        θ_y = ones(Int, length(y0))

        k = 1
        restartflag = false
        while !exitflag && !restartflag
            # Line 4
            j = rand(idx_seq)

            # Slice of variables based on nzrowsC[j]
            z_sliced = z[C[j]]
            q_sliced = q[C[j]]
            Adelta_sliced = a * ((k-1) .- θ_x[C[j]])

            # Line 5
            q_hat = q_sliced + Adelta_sliced .* z_sliced

            # Line 6
            x_hat = prox(x0[C[j]] - 1/γ * q_hat, 1/γ * a * k)

            # Line 7 & 12
            sliced_A_T = sliced_A_Ts[j]
            Delta_y = γ * m * a * ((x_hat' * sliced_A_T)' - b[blocks[j]])

            y_tilde[blocks[j]] = y_tilde[blocks[j]] + a * (k .- θ_y[blocks[j]]) .* y[blocks[j]]+ (m-1) * a * Delta_y[:]

            y[blocks[j]] = y[blocks[j]] + Delta_y

            # Line 10
            Delta_Delta_y = sliced_A_T * Delta_y
            q[C[j]] = q_sliced + a * ((k+1) .- θ_x[C[j]]) .* z_sliced + (m+1) * a * Delta_Delta_y

            # Line 11
            x_tilde[C[j]] = x_tilde[C[j]] + a * (k .- θ_x[C[j]]) .* x[C[j]]

            # Line 9
            z[C[j]] = z[C[j]] + Delta_Delta_y

            # Line 13
            x[C[j]] =  prox(x0[C[j]] - 1/γ * q[C[j]], 1/γ * (k+1) * a)

            # Line 14
            θ_x[C[j]] .= k
            θ_y[blocks[j]] .= k

            # Logging and checking exit condition
            # set restartflag when reached some measure
            if outer_k % (exitcriterion.loggingfreq * m) == 0
                x_tilde_tmp = x_tilde[:] + a * ((k+1) .- θ_x[:]) .* x[:]
                y_tilde_tmp = y_tilde[:] + a * ((k+1) .- θ_y[:]) .* y[:]
                x_out = x_tilde_tmp / (a * k)
                y_out = y_tilde_tmp / (a * k)

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
