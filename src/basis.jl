struct Basis{N,T<:AbstractMatrix}
    dims::NTuple{N}
    B::T
end

function construct_basis(bf, xs::Tuple, params)
    dims = length.(xs)
    nr = prod(dims)
    nc = length(params)
    B = zeros(nr,nc)
    for (i,p) in enumerate(params)
        B[:,i] .= vec(bf(xs,p))
    end
    return Basis(dims,B)
end

function construct_basis(bf, xs, params)
    construct_basis(bf,(xs,),params)
end

Base.:(*)(B::Basis,A::AbstractMatrix) = B.B*A
Base.:(*)(A::AbstractMatrix,B::Basis) = A*B.B

function evaluate(B::Basis, coeffs)
    if length(coeffs) != size(B.B,2)
        @error "Wrong Number of Coefficients for basis"
    end
    reshape(max.(B.B*coeffs,0.0),B.dims)
end
