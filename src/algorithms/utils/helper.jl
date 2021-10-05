# File containing helper functions for algorithms.

function compute_nzrows_for_blocks(A_T::SparseMatrixCSC, blocksize::Int)
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
