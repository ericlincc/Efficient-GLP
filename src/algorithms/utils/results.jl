struct Results
    iterations::Vector
    times::Vector
    measures::Vector
    
    function Results()
        new(Array{Int}([]), Array{Float64}([]), Array{Float64}([]))
    end
end

function logresult!(r::Results, currentiter, elapsedtime, measure)
    push!(r.iterations, currentiter)
    push!(r.times, elapsedtime)
    push!(r.measures, measure)
    return
end
