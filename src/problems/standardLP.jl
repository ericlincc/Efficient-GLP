struct StandardLinearProgram
    A_T::SparseMatrixCSC
    b::Vector
    c::Vector
    prox

    function StandardLinearProgram(A_T::SparseMatrixCSC, b::Vector, c::Vector)
        prox(x, τ) = max.(x, 0.0)
        new(A_T, b, c, prox)
    end
end
