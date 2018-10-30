# Orbit Space Covariance

function epr_cov(energy, pitch, r, orbit_type, p; independent=false, norm = ones(3))
    norb = length(energy)
    np = length(p)

    GO = Set((:trapped,:potato,:stagnation,:co_passing))
    Sigma_F = zeros(norb,norb)
    for i=1:norb
        x = [energy[i], pitch[i], r[i]]
        ox = orbit_type[i]
        for j=i:norb
            y = [energy[j], pitch[j], r[j]]
            oy = orbit_type[j]
            if independent
                same_ot = ox == oy
            else
                same_ot = ((ox in GO) && (oy in GO)) || (ox == oy)
            end
            mu = (x .- y)./norm
            if np == 2
                k = same_ot*(p[1]^2)*exp(-0.5*dot(mu,mu)/(p[2]^2)) + (i==j)*1e-6
            elseif np == 3
                k = same_ot*(p[1]^2)*exp(-0.5*dot(mu,mu)/(p[2]^2)) + (i==j)*p[3]^2
            elseif np == 4
                Σ_rho_inv = Diagonal(p[2:4].^(-2))
                k = same_ot*(p[1]^2)*exp(-0.5*dot(mu,Σ_rho_inv*mu)) + (i==j)*1e-6
            elseif np == 5
                Σ_rho_inv = Diagonal(p[2:4].^(-2))
                k = same_ot*(p[1]^2)*exp(-0.5*dot(mu,Σ_rho_inv*mu)) + (i==j)*p[5]^2
            else
                error("Unsupported number of parameters")
            end

	    Sigma_F[i,j] = k
            Sigma_F[j,i] = k
        end
    end

    return Sigma_F
end

# EPRZ Covariance
struct RepeatedBlockDiagonal{T,N,A<:AbstractArray} <: AbstractArray{T,N}
    data::A
    n::Int
end

Base.size(A::RepeatedBlockDiagonal) = size(A.data).*A.n

function Base.getindex(A::RepeatedBlockDiagonal,I::Int...)
    i, j = I
    ni, nj = size(A.data)
    ii = mod(i-1, ni) + 1
    jj = mod(j-1, nj) + 1

    return div(i-1,ni) == div(j-1,nj) ? A.data[ii,jj] : zero(eltype(A.data))
end

Base.getindex(A::RepeatedBlockDiagonal,I::Int) = A[CartesianIndices(A)[I]]

function RepeatedBlockDiagonal(A::Matrix,n::Int)
    RepeatedBlockDiagonal{eltype(A),2,typeof(A)}(A,n)
end

Base.:(*)(A::RepeatedBlockDiagonal,a::T) where {T<:Number} = RepeatedBlockDiagonal(a*A.data,A.n)
Base.:(*)(a::T, A::RepeatedBlockDiagonal) where {T<:Number} = RepeatedBlockDiagonal(a*A.data,A.n)
Base.:(/)(A::RepeatedBlockDiagonal,a::T) where {T<:Number} = RepeatedBlockDiagonal(A.data/a,A.n)

function Base.:(*)(A::RepeatedBlockDiagonal , a::Union{SparseVector{S,T},SparseMatrixCSC{S,T}}) where {S,T}
    iden = spzeros(S,A.n,A.n)
    iden[1,1] = one(S)
    B = kron(iden,A.data)*a
    iden[1,1] = zero(S)
    dropzeros!(iden)
    for i=2:A.n
        iden[i,i] = one(S)
        B = B .+ kron(iden,A.data)*a
        iden[i,i] = zero(S)
        dropzeros!(iden)
    end
    return B
end

function Base.:(*)(a::Union{SparseVector{S,T},SparseMatrixCSC{S,T}}, A::RepeatedBlockDiagonal) where {S,T}
    iden = spzeros(S,A.n,A.n)
    iden[1,1] = one(S)
    B = a*kron(iden,A.data)
    iden[1,1] = zero(S)
    dropzeros!(iden)
    for i=2:A.n
        iden[i,i] = one(S)
        B = B .+ a*kron(iden,A.data)
        iden[i,i] = zero(S)
        dropzeros!(iden)
    end
    return B
end

function ep_cov(energy, pitch, sigE, sigp)

    nenergy = length(energy)
    npitch = length(pitch)
    nep = nenergy*npitch

    Σ_ep = zeros(nep,nep)
    x = zeros(2)
    y = zeros(2)
    subs = CartesianIndices((nenergy,npitch))
    for i=1:nep
        ie,ip = Tuple(subs[i])
        x .= [energy[ie],pitch[ip]]
        for j=i:nep
            je,jp = Tuple(subs[j])
            y .= [energy[je],pitch[jp]]

            l = (x .- y)'*inv(Diagonal([sigE,sigp].^2))*(x .- y)
            Σ_ep[i,j] = exp(-0.5*l)
            Σ_ep[j,i] = Σ_ep[i,j]
        end
    end

    return Σ_ep
end

function ep_cov(energy, pitch, p::Vector)
    if p == 2
        return ep_covariance(energy, pitch, p[1], p[2])
    elseif p == 3
        return (p[1]^2)*ep_covariance(energy, pitch, p[2], p[3])
    else
        error("Unsupported number of parameters")
    end
end

function eprz_cov(energy, pitch, r, z, p::Vector)
    nr = length(r)
    nz = length(z)
    if p == 2
        Σ_ep = ep_covariance(energy, pitch, p[1], p[2])
    elseif p == 3
        Σ_ep = (p[1]^2)*ep_covariance(energy, pitch, p[2], p[3])
    else
        error("Unsupported number of parameters")
    end
    return RepeatedBlockDiagonal(Σ_ep,nr*nz)
end
