# Standard form without prox operator

function droreformuation_wmetric_hinge_standardformnormalized(
    yX_T::SparseMatrixCSC{Float64, Int},
    κ::Float64,
    ρ::Float64
)
    dim_dataset, num_dataset = size(yX_T)

    v1_n_T = transpose(sparse(ones(num_dataset)))
    v1_d_T = transpose(sparse(ones(dim_dataset)))
    v0_n_T = transpose(spzeros(num_dataset))

    I_nn = sparse(I, num_dataset, num_dataset)
    I_dd = sparse(I, dim_dataset, dim_dataset)
    O_nn = spzeros(num_dataset, num_dataset)
    O_dn = spzeros(dim_dataset, num_dataset)
    O_nd = spzeros(num_dataset, dim_dataset)
    O_dd = spzeros(dim_dataset, dim_dataset)

    A_T = [
        -I_nn       I_nn      I_nn      O_nd      O_nd  ;
        I_nn       O_nn      I_nn      O_nd      O_nd  ;
        O_nn      -I_nn     -I_nn      O_nd      O_nd  ;
        O_nn       O_nn     -I_nn      O_nd      O_nd  ;
        O_dn       yX_T      O_dn      I_dd      I_dd  ;
        O_dn      -yX_T      O_dn     -I_dd     -I_dd  ;
        O_dn       O_dn      O_dn     -I_dd      O_dd  ;
        O_dn       O_dn      O_dn      O_dd      I_dd  ; 
        -2*κ*v1_n_T v0_n_T    v0_n_T    v1_d_T   -v1_d_T;
        2*κ*v1_n_T v0_n_T    v0_n_T   -v1_d_T    v1_d_T;
    ]
    b = sparsevec([
        spzeros(num_dataset);
        sparse(ones(num_dataset));
        2 * sparse(ones(num_dataset));
        spzeros(dim_dataset);
        spzeros(dim_dataset);
    ])

    # Normalize all row norms of A to 1
    divrownorm_A = map(x -> 1 / norm(x), eachcol(A_T))
    range_A = collect(1:A_T.n)
    diag_divrownorm_A = sparse(range_A, range_A, divrownorm_A)
    A_T *= diag_divrownorm_A
    b = diag_divrownorm_A * b

    # for i in 1:size(A_T)[2]
    #     A_T[:, i] = A_T[:, i] / rownorm_A[i]
    #     b[i] = b[i] / rownorm_A[i]
    # end

    c = [
        1 / num_dataset * ones(num_dataset);
        zeros(num_dataset);
        zeros(num_dataset);
        zeros(num_dataset);
        zeros(dim_dataset);
        zeros(dim_dataset);
        zeros(dim_dataset);
        zeros(dim_dataset);
        ρ;
        -ρ;
    ]
    A_T, b, c
end
