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


"""Read a libsvm binary dataset into a sparse data matrix transposed (i.e. y *. X^T)."""
function read_libsvm_into_yXT_sparse(filepath::String, dim_dataset::Int, num_dataset::Int)
    
    train_indices = Array{Int}([])
    feature_indices = Array{Int}([])
    values = Array{Float64}([])

    open(filepath) do f

        line = 1
        while !eof(f) 
            s = readline(f)
            split_line = split(s, " ", keepempty=false)

            label = nothing
            for v in split_line
                if isnothing(label)
                    label = parse(Int, v)
                    continue
                end

                _index, _value = split(v, ":")
                index = parse(Int, _index)
                value = label * parse(Float64, _value)  # value = b_i * x_{i j}
                
                push!(train_indices, line)
                push!(feature_indices, index)
                push!(values, value)
            end        
            line += 1
        end
    end

    sparse(feature_indices, train_indices, values, dim_dataset, num_dataset)
end
