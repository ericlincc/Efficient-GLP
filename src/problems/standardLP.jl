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


"""Data structure for formalizing a standard-form linear program."""
struct StandardLinearProgram

    A_T::SparseMatrixCSC{Float64, Int64}
    b::Vector{Float64}
    c::Vector{Float64}
    prox

    function StandardLinearProgram(A_T, b, c)
        prox(x, τ) = max.(x, 0.0)
        new(sparse(A_T), Vector{Float64}(b), Vector{Float64}(c), prox)
    end
end
