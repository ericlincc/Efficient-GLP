# using LinearAlgebra
# the problem to optimize
#  \min_{\vx\in\gX\subset\sR^d} \max_{\vy\in\sR^n} \Big\{ L(\vx,\vy) = \vc^T\vx + r(\vx)   + \vy^T\mA\vx   - \vy^T\vb\Big\}.
# op_X_r: (I + tau_n ∂G)^{-1}
# γ: strong convexity
# σ, τ, θ: sequence for update
# L: Lipschitz constant

# to be consistent with the sparse column format, we consider
# D: R^{d×n}
# TODO: Update the above comments


# TODO: CB SAID THIS VERSION IS WRONG
function pdhg(problem::StandardLinearProgram, exitcriterion::ExitCriterion; γ=0.0, L=100.0)
    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox
    d, n = size(A_T)

    τ, σ = 1.0 / L, 1.0 / L
    x0 = zeros(d)
    y0 = zeros(n)

    x_bar = x0
    y = deepcopy(y0)
    x = deepcopy(x0)
    x_tilde = zero(x0)

    # Log initial measure
    starttime = time()
    results = Results()
    init_norm_const = norm((x0' * A_T)' - b)
    logresult!(results, 1, 0.0, init_norm_const)

    k = 2
    exitflag = false

    while !exitflag
        y[:] = y[:] + σ * ((x_bar' * A_T)' - b)

        x_pre = deepcopy(x)
        x[:] = x[:]  - τ * (A_T * y[:] + c[:])
        x = prox(x, τ)
        θ = 1 / sqrt(1 + 2 * γ * τ)
        τ = θ * τ
        σ = σ / θ

        x_bar[:] = x_bar[:] + θ * (x[:] -  x_pre[:])
        x_tilde[:] = x_tilde[:] + x[:]

        if k % exitcriterion.loggingfreg == 0
            norm_const = norm(((x_tilde / k)' * A_T)' - b)

            elapsedtime = time() - starttime
            @info "k: $(k), PDHG constraint norm: $norm_const, elapsedtime: $elapsedtime"

            logresult!(results, k, elapsedtime, norm_const)

            exitflag = checkexitcondition(exitcriterion, k, elapsedtime, norm_const)
        end

        k += 1

    end

    return results, profilings
end
