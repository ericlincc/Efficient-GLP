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
