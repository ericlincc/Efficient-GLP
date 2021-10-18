struct Results
    # A data structure for storing execution results.

    iterations::Vector{Float64}
    times::Vector{Float64}
    fvaluegaps::Vector{Float64}
    metricLPs::Vector{Float64}
    
    function Results()
        new(Array{Int64}([]), Array{Float64}([]), Array{Float64}([]), Array{Float64}([]))
    end
end

function logresult!(r::Results, currentiter, elapsedtime, fvaluegap, metricLP)
    # Append execution measures to Results.

    push!(r.iterations, currentiter)
    push!(r.times, elapsedtime)
    push!(r.fvaluegaps, fvaluegap)
    push!(r.metricLPs, metricLP)
    return
end
