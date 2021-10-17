# File containing helper functions for algorithms.

function compute_nzrows_for_blocks(A_T::SparseMatrixCSC, blocksize::Int)
    # Compute the nonzero rows in A_T for each block.

    d, n = size(A_T)

    blocks = Array{UnitRange{Int}}([])
    C = Array{Vector{Int}}([])
    len_b = n ÷ blocksize
    for i = 1:len_b
        if i == len_b
            push!(blocks, (1 + (i-1) * blocksize): n)
        else
            push!(blocks, (1 + (i-1) * blocksize): (i * blocksize))
        end

        row_set = Set{Int}()
        for j in blocks[i]
            loc = A_T.colptr[j]:(A_T.colptr[j+1]-1)
            union!(row_set, A_T.rowval[loc])
        end
        row_vec = collect(row_set)
        push!(C, row_vec)
    end

    blocks, C
end


function exportresultstoCSV(results::Results, outputfile::String)
    # Export results into CSV formatted file.

    CSV.write(
        outputfile, (
            iterations = results.iterations,
            times = results.times,
            fvaluegaps = results.fvaluegaps,
            metricLPs = results.metricLPs
        )
    )
end


function compute_fvaluegap_metricLP(x_out::Vector, y_out::Vector, problem::StandardLinearProgram)
    # Computing a common metric for LP. See Eqn (20) in Applegate et al 2020.

    A_T, b, c = problem.A_T, problem.b, problem.c

    norm1 = norm(max.(-x_out, 0))
    norm2 = norm(max.((x_out' * A_T)' - b, 0))
    norm3 = norm(max.(-(x_out' * A_T)' + b, 0))  # TODO: Can combine with norm2
    norm4 = norm(max.(-A_T * y_out - c, 0))
    norm5 = norm(c' * x_out + b' * y_out)
    
    norm5, sqrt(norm1^2 + norm2^2 + norm3^2 + norm4^2 + norm5^2)
end


function nnz_per_row(A_T::SparseMatrixCSC)
    ret = zeros(Int, size(A_T)[1])
    rows, columns, values = findnz(A_T)
    for (r, c, v) in zip(rows, columns, values)
        if !isapprox(v, 0.)
            ret[r] += 1
        end
    end
    return ret
end
