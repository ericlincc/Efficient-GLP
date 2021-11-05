
# TODO: THIS IS UNTESTED
function spdhg(
    problem::StandardLinearProgram,
    exitcriterion::ExitCriterion;
    R=1.0, blocksize=1)

    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox

    d, n = size(A_T)
    τ, σ =  1.0 / (R * (n ÷ blocksize)), 1.0 / R

    _time1 = time()
    # Precomputing blocks
    blocks, _ = compute_nzrows_for_blocks(A_T, blocksize)
    _time2 = time()
    @info ("Initialization time = ", _time2 - _time1)   

    x0 = zeros(d)
    y0 = zeros(n)
    z = zero(x0)
    grad = zero(x0)
    x_tilde = zero(x0)
    x = deepcopy(x0)
    y = y0
    idx_seq = 1:length(blocks)
    len_b = length(blocks)

    # Log initial measure
    starttime = time()
    results = Results()
    init_norm_const = norm((x0' * A_T)' - b)
    logresult!(results, 1, 0.0, init_norm_const)

    k = 1
    exitflag = false

    while !exitflag
        x[:] = x[:] - τ * (grad[:] + c[:])
        x = prox(x, τ)
        j = rand(idx_seq)

        Delta_y = -y[blocks[j]]
        y[blocks[j]] = y[blocks[j]] + σ * ((x' * A_T[:, blocks[j]])' - b[blocks[j]])
        Delta_y += y[blocks[j]]
        tmp = A_T[:, blocks[j]] * Delta_y
        z[:] = z[:] + tmp[:]
        grad[:] = z[:] + len_b * tmp[:]
        x_tilde[:] = x_tilde[:] + x[:]

        # Recording progress
        if k % exitcriterion.loggingfreg == 0
            norm_const = norm(((x_tilde/k)' * A_T)' - b)

            elapsedtime = time() - starttime
            @info "k: $(k), SPDHG constraint norm: $norm_const, elapsedtime: $elapsedtime"
            logresult!(results, k, elapsedtime, norm_const)

            exitflag = checkexitcondition(exitcriterion, k, elapsedtime, norm_const)
        end

        k += 1
    end
end
