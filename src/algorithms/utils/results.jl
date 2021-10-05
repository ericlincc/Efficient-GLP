struct Results
    iterations::Vector
    times::Vector
    fvalues::Vector
    constraintnorms::Vector
    
    function Results()
        new(Array{Int}([]), Array{Float64}([]), Array{Float64}([]), Array{Float64}([]))
    end
end

function logresult!(r::Results, currentiter, elapsedtime, fvalue, constraintnorm)
    push!(r.iterations, currentiter)
    push!(r.times, elapsedtime)
    push!(r.fvalues, fvalue)
    push!(r.constraintnorms, constraintnorm)
    return
end
