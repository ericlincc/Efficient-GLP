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
