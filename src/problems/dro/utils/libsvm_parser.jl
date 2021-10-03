

function read_libsvm_into_yXT_sparse(filepath::String, dim_dataset::Int, num_dataset::Int)
    
    train_index = Array{Int}([])
    feature_index = Array{Int}([])
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
                
                push!(train_index, line)
                push!(feature_index, index)
                push!(values, value)
            end        
            line += 1
        end
    end

    yX_T = sparse(feature_index, train_index, values, dim_dataset, num_dataset)
end
