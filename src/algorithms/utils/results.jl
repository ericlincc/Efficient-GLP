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


# This file contains the Results struct's definition and its dispatches.


"""
Defines the progress of execution at each logging step.
* iterations: Number of iterations elapsed.
* times: Elapsed times since start of execution (in seconds).
* fvaluegaps: Primal and dual objective value gaps.
* metricLPs: The computed values of the LP metric.
"""
struct Results
    iterations::Vector{Float64}
    times::Vector{Float64}
    fvaluegaps::Vector{Float64}
    metricLPs::Vector{Float64}
    
    function Results()
        new(Array{Int64}([]), Array{Float64}([]), Array{Float64}([]), Array{Float64}([]))
    end
end


"""Append execution measures to Results."""
function logresult!(r::Results, currentiter, elapsedtime, fvaluegap, metricLP)

    push!(r.iterations, currentiter)
    push!(r.times, elapsedtime)
    push!(r.fvaluegaps, fvaluegap)
    push!(r.metricLPs, metricLP)
    return
end
