# the problem to optimize
#  \min_{\vx\in\gX\subset\sR^d} \max_{\vy\in\sR^n} \Big\{ L(\vx,\vy) = \vc^T\vx + r(\vx)   + \vy^T\mA\vx   - \vy^T\vb\Big\}.
# op_X_r: (I + tau_n ∂G)^{-1}
# γ: strong convexity
# σ, τ, θ: sequence for update
# L: Lipschitz constant
# to be consistent with the sparse column format, we consider
# A_T: R^{d×n}


# function test_iclr_lazy()

#     n, d = 1000, 500
#     A_T = (randn(d, n) +  ones(d, n)) /√n
#     A_T = sparse(A_T)
#     # b = randn(n)
#     x_star = ones(d)
#     b = (x_star' * A_T)'
#     c = ones(d)
#     R = 2
#     γ = 1
#     σ = 0.0
#     K = 100000
#     x0 = zeros(d)
#     y0 = zeros(n)
#     blocks = []
#     C = []
#     bs = 10  # block size
#     len_b = n ÷ bs

#     for i = 1:len_b
#         push!(blocks, (1 + (i-1) * bs): (i * bs))
#         row_set = []
#         for j = 1:length(blocks[i])
#             row_set = union(row_set, rowvals(A_T[:, blocks[i][j]]))
#         end
#         push!(C, row_set)
#         # @info row_set
#     end
#     op_X_r(x, τ) = max.(x, 0.0)

#     # for i = 1:100
#     #     x0, y0 = iclr_lazy_restart_x_y(A_T, b, c, op_X_r, x0, y0, n, d, γ, σ, R, blocks, C, K * len_b)
#     # end

#     iclr_lazy(A_T, b, c, op_X_r, x0, y0, n, d, γ, σ, R, blocks, C, K * len_b)
# end

# S: block separation
# C:
# theta:
# implement all these things


function efficientsparsearrayslicing(A::SparseMatrixCSC, slicerow::Vector{Int}, slicecol)
    row_indices = Array{Int}([])
    col_indices = Array{Int}([])
    vals = Array{Float64}([])
    for (i, col) in enumerate(slicecol)
        loc = A.colptr[col]:(A.colptr[col+1]-1)
        
        slicerow2index_dict = Dict{Int, Int}()
        for (i, x) in enumerate(slicerow)
            slicerow2index_dict[x] = i
        end
        
        if length(loc) == 0
            continue
        end

        for (r, v) in zip(A.rowval[loc], A.nzval[loc])
            if haskey(slicerow2index_dict, r)
                push!(row_indices, slicerow2index_dict[r])
                push!(col_indices, i)
                push!(vals, v)
            end
        end
    end
    sparse(row_indices, col_indices, vals, length(slicerow), length(slicecol))
end
                


# TODO: Put blocks and C generation in utils

function iclr_lazy(problem::StandardLinearProgram, exitcriterion::ExitCriterion; γ=1.0, σ=0.0, R=10, blocksize=10)  # TODO: What is γ and σ?

    A_T, b, c = problem.A_T, problem.b, problem.c
    prox = problem.prox

    d, n = size(A_T)

    x0 = zeros(d)
    y0 = zeros(n)

    blocks = Array{UnitRange{Int}}([])
    C = Array{Vector{Int}}([])
    len_b = n ÷ blocksize

    for i = 1:len_b
        push!(blocks, (1 + (i-1) * blocksize): (i * blocksize))  # TODO: Ask CB if the data beyond the last blocksize is discarded
        row_set = Array{Vector{Int}}([])
        # row_set = Set{Int}()
        # row_set = collect(1:d)
        for j = 1:length(blocks[i])
            row_set = union(row_set, rowvals(A_T[:, blocks[i][j]]))
            # union!(row_set, rowvals(A_T[:, blocks[i][j]]))
        end
        push!(C, row_set)
    end
    

    # TODO: Report initialization time
    _time1 = time()
    sliced_A_Ts = Array{SparseMatrixCSC{Float64, Int}}([])
    for j in 1:len_b
        push!(sliced_A_Ts, A_T[C[j], blocks[j]])
    end
    _time2 = time()
    println("pre time = ", _time2 - _time1)


    ##### Start of iclr_lazy

    K = exitcriterion.maxiter  #

    m = length(blocks)
    a = 1/(R * m)
    A = zeros(K+2)
    A[2] = a
    pre_a, pre_A = 0.0, 0.0
    # q = zero(x0)
    q = - γ * deepcopy(x0)
    idx_seq = 1:m
    x = deepcopy(x0)
    y = deepcopy(y0)
    x_tilde = zero(x0)
    # z = zero(x0)
    z = deepcopy(c)
    θ = ones(Int64, length(x0))

    # Log initial measure
    starttime = time()
    results = Results()
    init_norm_const = norm((x0' * A_T)' - b)
    logresult!(results, 1, 0.0, init_norm_const)

    k = 2
    exitflag = false

    # profilings = zeros(8)  #

    while !exitflag
        j = rand(idx_seq)

        # time_0 = time()  #
        z_sliced = z[C[j]]
        q_sliced = q[C[j]]
        Adelta_sliced = A[k] .- A[θ[C[j]]]

        q_hat = q_sliced + Adelta_sliced .* z_sliced ##
        # q_hat = q[C[j]] + (A[k] .- A[θ[C[j]]]) .* (z[C[j]] + c[C[j]]) ##
        # q_hat = q[C[j]] + a * (k .- θ[C[j]]) .* (z[C[j]] + c[C[j]])


        # time_1 = time()  #

        
        # x_hat = prox(x0[C[j]] - 1/γ * q_hat, 1/γ * A[k])
        x_hat = prox(- 1/γ * q_hat, 1/γ * A[k])
        

        # time_2 = time()  #


        sliced_A_T = sliced_A_Ts[j]  # A_T[C[j], blocks[j]]
        # sliced_A_T = efficientsparsearrayslicing(A_T, C[j], blocks[j])

        # time_3 = time()  #

        Delta_y = γ * m * a * ((x_hat' * sliced_A_T)' - b[blocks[j]])


        # time_4 = time()  #


        y[blocks[j]] = y[blocks[j]] + Delta_y
        pre_a = a
        a = sqrt(1 + σ * A[k] / γ)/(R * m)


        A[k+1] = A[k] + a
        Delta_Delta_y = sliced_A_T * Delta_y   

        # time_5 = time()  #


        # q[C[j]] = q[C[j]] + (A[k+1] .- A[θ[C[j]] .+ 1]) .* (z[C[j]] + c[C[j]]) + (a + m * pre_a) * Delta_Delta_y ##
        q[C[j]] = q_sliced + Adelta_sliced .* z_sliced + (a + m * pre_a) * Delta_Delta_y ##


        # time_6 = time()  #


        z[C[j]] = z[C[j]] + Delta_Delta_y
        # z[C[j]] = z[C[j]] + Delta_Delta_y + c[C[j]]


        # time_7 = time()  #


        # x[C[j]] =  prox(x0[C[j]] - 1/γ * q[C[j]], 1/γ * A[k])
        x[C[j]] =  prox(x0[C[j]] - 1/γ * q[C[j]], 1/γ * A[k])
        θ[C[j]] .= k

        # time_8 = time()  #

        x_tilde += pre_a * x

        # time_deltas = [
        #     time_1 - time_0;
        #     time_2 - time_1;
        #     time_3 - time_2;
        #     time_4 - time_3;
        #     time_5 - time_4;
        #     time_6 - time_5;
        #     time_7 - time_6;
        #     time_8 - time_7
        # ]       #
        # profilings += time_deltas

        # Recording progress
        if k % exitcriterion.loggingfreg == 0

            norm_const = norm(((x_tilde/A[k])' * A_T)' - b)

            elapsedtime = time() - starttime
            @info "k: $(k), ICLR constraint norm: $norm_const, elapsedtime: $elapsedtime"

            logresult!(results, k, elapsedtime, norm_const)    # TODO: number of data pass i/o number of iterations

            exitflag = checkexitcondition(exitcriterion, k, elapsedtime, norm_const)
        end

        k += 1
    end

    return results # , profilings
end


# function iclr_lazy(A_T, b, c, op_X_r, x0, y0, n, d, γ, σ, R, S, C, K)

#     m = length(S)
#     a = 1/(R * m)
#     A = zeros(K+2)
#     A[2] = a
#     pre_a, pre_A = 0.0, 0.0
#     q = zero(x0)
#     idx_seq = 1:m
#     x = deepcopy(x0)
#     y = deepcopy(y0)
#     x_tilde = zero(x0)
#     z = zero(x0)
#     θ = ones(Int64, length(x0))

    
#     for k = 2:K

#         j = rand(idx_seq)

#         q_hat = q[C[j]] + (A[k] .- A[θ[C[j]]]) .* (z[C[j]] + c[C[j]])

#         x_hat = op_X_r(x0[C[j]] - 1/γ * q_hat, 1/γ * A[k])  # TODO: Check why not used
#         Delta_y = γ * m * a * ((x[C[j]]' * A_T[C[j], S[j]])' - b[S[j]])
#         y[S[j]] = y[S[j]] + Delta_y
#         pre_a = a
#         a = sqrt(1 + σ * A[k] / γ)/(R * m)

#         A[k+1] = A[k] + a
#         Delta_Delta_y = A_T[C[j], S[j]] * Delta_y

#         q[C[j]] = q[C[j]] + (A[k+1] .- A[θ[C[j]] .+ 1]) .* (z[C[j]] + c[C[j]]) + (a + m * pre_a) * Delta_Delta_y

#         z[C[j]] = z[C[j]] + Delta_Delta_y

#         x[C[j]] =  op_X_r(x0[C[j]] - 1/γ * q[C[j]], 1/γ * A[k])
#         θ[C[j]] .= k

#         x_tilde += pre_a * x

#         # Record progress here: Constraint norm, time, #iteration.
        
#         if k % modulor == 0
#             # x_tilde[:] = x_tilde[:] + (A[k] .- A[θ[:]]) .* x[:]
#             norm_const = norm(((x_tilde/A[k])' * A_T)' - b)
#             @info "k:$(k), ICLR constraint norm: $norm_const"
#             modulor *= 10
#         end
#     end
#     # x_tilde[:] = x_tilde[:] + (A[K+1] .- A[θ[:]]) .* x[:]
#     return 0
# end
