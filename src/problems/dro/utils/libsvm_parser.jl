

function read_libsvm_into_yXT_sparse(filepath::String, dim_dataset::Int, num_dataset::Int)
    
    train_indices = Array{Int}([])
    feature_indices = Array{Int}([])
    values = Array{Float64}([])

    open(filepath) do f

        line = 1
        while !eof(f) 
            s = readline(f)
            split_line = split(s, " ", keepempty=false)

            label = nothing
            for v in split_line
                if isnothing(label)
                    label = parse(Int, v)
                    continue
                end

                _index, _value = split(v, ":")
                index = parse(Int, _index)
                value = label * parse(Float64, _value)  # value = b_i * x_{i j}
                
                push!(train_indices, line)
                push!(feature_indices, index)
                push!(values, value)
            end        
            line += 1
        end
    end

    yX_T = sparse(feature_indices, train_indices, values, dim_dataset, num_dataset)
end
