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


# File containing helper functions for algorithms.

""" Compute the nonzero rows in A_T for each block."""
function compute_nzrows_for_blocks(A_T::SparseMatrixCSC, blocksize::Int)

    _, n = size(A_T)

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


"""Export results into CSV formatted file."""
function exportresultstoCSV(results::Results, outputfile::String)

    CSV.write(
        outputfile, (
            iterations = results.iterations,
            times = results.times,
            fvaluegaps = results.fvaluegaps,
            metricLPs = results.metricLPs
        )
    )
end


"""Computing a common metric for LP. See Eqn (20) in Applegate et al 2020."""
function compute_fvaluegap_metricLP(x_out::Vector, y_out::Vector, problem::StandardLinearProgram)

    A_T, b, c = problem.A_T, problem.b, problem.c

    norm1 = norm(max.(-x_out, 0))
    norm2 = norm(max.((x_out' * A_T)' - b, 0))
    norm3 = norm(max.(-(x_out' * A_T)' + b, 0))  # TODO: Can combine with norm2
    norm4 = norm(max.(-A_T * y_out - c, 0))
    norm5 = norm(c' * x_out + b' * y_out)
    
    norm5, sqrt(norm1^2 + norm2^2 + norm3^2 + norm4^2 + norm5^2)
end


"""Compute number of nonzero elements of each row in a sparse column matrix."""
function nnz_per_row(A_T::SparseMatrixCSC)
    
    ret = zeros(Int, size(A_T)[1])
    rows, _, values = findnz(A_T)
    for (r, v) in zip(rows, values)
        if !isapprox(v, 0.)
            ret[r] += 1
        end
    end
    return ret
end
