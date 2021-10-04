struct StandardLinearProgram
    A_T::SparseMatrixCSC
    b::Vector
    c::Vector
    prox

    function StandardLinearProgram(A_T, b, c)
        prox(x, τ) = max.(x, 0.0)
        new(sparse(A_T), Vector(b), Vector(c), prox)
    end
end
