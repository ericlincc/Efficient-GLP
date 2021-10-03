struct ExitCriterion
    maxiter::Int              # Max #iterations allowed
    maxtime::Float64          # Max execution time allowed
    targetaccuracy::Float64   # Target accuracy to halt algorithm
    loggingfreg::Int          # #iterations interval between logging
end


# A function determine if it's time to halt execution
function checkexitcondition(
    exitcriterion::ExitCriterion,
    currentiter::Integer,
    elapsedtime,
    measure,
)::Bool

    if currentiter >= exitcriterion.maxiter
        return true
    elseif elapsedtime >= exitcriterion.maxtime
        return true
    elseif measure <= exitcriterion.targetaccuracy
        return true
    end

    false
end
