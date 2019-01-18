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
    nr,nc = size(A.data)
    colptr = vcat(1:nr:(nr*nc),fill(nr*nc+1,nr*A.n + 1 - nc))
    colptr2 = Array{Int}(undef,length(colptr))
    rowval = repeat(1:nr,nc)
    rowval2 = Array{Int}(undef,length(rowval))
    K = SparseMatrixCSC(size(A)..., colptr, rowval, vec(A.data))

    p = Progress(A.n)
    channel = RemoteChannel(()->Channel{Bool}(A.n), 1)
    B = fetch(@sync begin
        @async while take!(channel)
            next!(p)
        end
        @async begin
            B = @distributed (+) for i=1:A.n
                circshift!(colptr2,colptr,(i-1)*nc)
                rowval2 .= rowval .+ (i-1)*nr
                K = SparseMatrixCSC(K.m,K.n,colptr2,rowval2, K.nzval)
                Ka = K*a
                put!(channel,true)
                Ka
            end
            put!(channel, false)
            B
        end
    end)
    return B
end

function Base.:(*)(a::Union{SparseVector{S,T},SparseMatrixCSC{S,T}}, A::RepeatedBlockDiagonal) where {S,T}
    nr,nc = size(A.data)
    colptr = vcat(1:nr:(nr*nc),fill(nr*nc+1,nr*A.n + 1 - nc))
    colptr2 = Array{Int}(undef,length(colptr))
    rowval = repeat(1:nr,nc)
    rowval2 = Array{Int}(undef,length(rowval))
    K = SparseMatrixCSC(size(A)..., colptr, rowval, vec(A.data))

    p = Progress(A.n)
    channel = RemoteChannel(()->Channel{Bool}(A.n), 1)
    B = fetch(@sync begin
        @async while take!(channel)
            next!(p)
        end
        @async begin
            B = @distributed (+) for i=1:A.n
                circshift!(colptr2,colptr,(i-1)*nc)
                rowval2 .= rowval .+ (i-1)*nr
                K = SparseMatrixCSC(K.m,K.n,colptr2,rowval2, K.nzval)
                aK = a*K
                put!(channel,true)
                aK
            end
            put!(channel, false)
            B
        end
    end)
    return B
end

function transform_eprz_cov(C,R)
    nr,nc = size(C.data)
    colptr = vcat(1:nr:(nr*nc),fill(nr*nc+1,nr*C.n + 1 - nc))
    colptr2 = Array{Int}(undef,length(colptr))
    rowval = repeat(1:nr,nc)
    rowval2 = Array{Int}(undef,length(rowval))
    K = SparseMatrixCSC(size(C)..., colptr, rowval, vec(C.data))
    S = @distributed (+) for i=1:C.n
        circshift!(colptr2,colptr,(i-1)*nc)
        rowval2 .= rowval .+ (i-1)*nr
        K = SparseMatrixCSC(K.m,K.n,colptr2,rowval2, K.nzval)
        Array((R*(R*K)'))
    end

    return S
end

function transform_eprz_cov2(C,R)
    nr,nc = size(C.data)
    Rnr = size(R)[1]
    Cs = sparse(C.data)

    S = @distributed (+) for i=1:nr:(C.n*nr)
        Ri = R[:,i:(i+nr-1)]
        if nnz(Ri) != 0
            Si = Ri*(Ri*Cs)'
        else
            Si = spzeros(Rnr,Rnr)
        end
        Si
    end

    return Array(S)
end

function transform_eprz_cov3(C,R)
    nr,nc = size(C.data)
    Rnr = size(R)[1]
    Cs = sparse(C.data)

    Ris = SparseMatrixCSC{Float64,Int64}[]
    for i=1:nr:(C.n*nr)
        Ri = R[:,i:(i+nr-1)]
        nnz(Ri) == 0 && continue
        push!(Ris,Ri)
    end

    Si = [zeros(Rnr,Rnr) for i=1:Threads.nthreads()]
    Threads.@threads for Ri in Ris
        id = Threads.threadid()
        Si[id] .= Si[id] .+ Ri*(Ri*Cs)'
    end

    return sum(Si)
end

function LinearAlgebra.inv(A::RepeatedBlockDiagonal)
    RepeatedBlockDiagonal(inv(A.data),A.n)
end

function ep_cov(energy, pitch, sigE, sigp)

    nenergy = length(energy)
    npitch = length(pitch)
    nep = nenergy*npitch

    Σ_ep = zeros(typeof(sigE),nep,nep)
    Σ_p_inv = SMatrix{2,2}(inv(Diagonal([sigE,sigp].^2)))
    subs = CartesianIndices((nenergy,npitch))
    @inbounds for i=1:nep
        ie,ip = Tuple(subs[i])
        x = SVector{2}(energy[ie],pitch[ip])
        for j=i:nep
            je,jp = Tuple(subs[j])
            y = SVector{2}(energy[je],pitch[jp])
            d = x .- y
            l = d'*Σ_p_inv*d
            Σ_ep[i,j] = exp(-0.5*l)
            Σ_ep[j,i] = Σ_ep[i,j]
        end
    end

    return Σ_ep
end

function ep_cov(energy, pitch, p::Vector)
    if length(p) == 2
        return ep_cov(energy, pitch, p[1], p[2])
    elseif length(p) == 3
        return (p[1]^2)*ep_cov(energy, pitch, p[2], p[3])
    else
        error("Unsupported number of parameters")
    end
end

function eprz_cov(energy, pitch, r, z, p::Vector)
    nr = length(r)
    nz = length(z)
    if length(p) == 2
        Σ_ep = ep_cov(energy, pitch, p[1], p[2])
    elseif length(p) == 3
        Σ_ep = (p[1]^2)*ep_cov(energy, pitch, p[2], p[3])
    else
        error("Unsupported number of parameters")
    end
    return RepeatedBlockDiagonal(Σ_ep,nr*nz)
end

function get_covariance(M::AxisymmetricEquilibrium, o1::Orbit, J1::Vector, o2::Orbit, J2::Vector, sigma::Vector, dt::Int)
    n1 = length(o1.path)
    n2 = length(o2.path)
    if n1 == 0 || n2 == 0
        return 0.0
    end
    n1r = length(1:dt:n1)
    n2r = length(1:dt:n2)

    # change storage to be more memory cache friendly
    eprz1 = vcat(o1.path.energy',o1.path.pitch',o1.path.r',o1.path.z')
    eprz2 = vcat(o2.path.energy',o2.path.pitch',o2.path.r',o2.path.z')

    Σ_p_inv = SMatrix{4,4}(inv(Diagonal(sigma.^2)))
    K = 0.0
    @inbounds for i=1:dt:n1
        J1i = J1[i]
        x = SVector{4}(eprz1[:,i])
        Kj = 0.0
        for j=1:dt:n2
            J2j = J2[j]
            y = SVector{4}(eprz2[:,j])
            d = x .- y
            l = d'*Σ_p_inv*d
            Kj += J2j*exp(-0.5*l)
        end
        K += J1i*Kj/n2r
    end
    K = K/n1r

    return K
end

function get_covariance(M::AxisymmetricEquilibrium, o::Orbit, sigma;
                        J::Vector{Float64} = Float64[], dt::Int = 1, kwargs...)
    if isempty(J)
        J1 = get_jacobian(M, o; kwargs...)
    else
        J1 = J
    end
    return get_covariance(M, o, J1, o, J1, sigma, dt)
end

function get_covariance(M::AxisymmetricEquilibrium, o1::Orbit, o2::Orbit, sigma;
                        J1::Vector{Float64} = Float64[],
                        J2::Vector{Float64} = Float64[], dt::Int = 1, kwargs...)
    if isempty(J1)
        J11 = get_jacobian(M, o1; kwargs...)
    else
        J11 = J1
    end
    if isempty(J2)
        J22 = get_jacobian(M, o2; kwargs...)
    else
        J22 = J2
    end
    return get_covariance(M, o1, J11, o2, J22, sigma, dt)
end

function get_covariance(M::AxisymmetricEquilibrium, c1::T, c2::T, sigma;
                        dt::Int = 1, kwargs...) where T
    o1 = get_orbit(M, c1; kwargs...)
    o2 = get_orbit(M, c2; kwargs...)
    return get_covariance(M, o1, o2, sigma; dt=dt, kwargs...)
end

function get_correlation(M::AxisymmetricEquilibrium, o1::T, o2::T, sigma;
                         k1::Float64 = 0.0, k2::Float64 = 0.0,
                         J1::Vector{Float64} = Float64[],
                         J2::Vector{Float64} = Float64[],
                         dt::Int = 1, kwargs...) where T

    if isempty(J1)
        J11 = get_jacobian(M, o1; kwargs...)
    else
        J11 = J1
    end

    if isempty(J2)
        J22 = get_jacobian(M, o2; kwargs...)
    else
        J22 = J2
    end

    if k1 == 0.0
        k11 = get_covariance(M, o1, J11, o1, J11, sigma, dt)
    else
        k11 = k1
    end

    if k2 == 0.0
        k22 = get_covariance(M, o2, J22, o2, J22, sigma, dt)
    else
        k22 = k2
    end

    if k11 == 0.0 || k22 == 0.0
        return k11*k22
    end

    k12 = get_covariance(M, o1, J11, o2, J22, sigma, dt)

    return k12/sqrt(k11*k22)
end

function get_correlation_matrix(M::AxisymmetricEquilibrium, orbits::Vector, sigma::Vector;
                                ks::Vector{Float64} = Float64[],
                                Js::Vector{Vector{Float64}} = Vector{Float64}[],
                                dt::Int = 1, kwargs...)
    n = length(orbits)
    if isempty(Js)
        J = Array{Vector{Float64}}(undef, n)
        @inbounds Threads.@threads for i=1:n
            oi = orbits[i]
            Ji = get_jacobian(M, oi; kwargs...)
            J[i] = Ji
        end
    else
        J = Js
    end

    if isempty(ks)
        K = zeros(n)
        @inbounds Threads.@threads for i=1:n
            oi = orbits[i]
            Ji = J[i]
            ki = get_covariance(M, oi, sigma; J = Ji, dt = dt)
            K[i] = ki
        end
    else
        K = ks
    end

    Σ = zeros(n,n)
    @time begin
    @inbounds Threads.@threads for i=1:n
        oi = orbits[i]
        Σ[i,i] = 1.0
        Ji = J[i]
        ki = K[i]
        for j=(i+1):n
            oj = orbits[j]
            Jj = J[j]
            kj = K[j]
            Σ[i,j] = get_correlation(M, oi, oj, sigma, J1=Ji, J2=Jj, k1=ki, k2=kj, dt = dt)
            Σ[j,i] = Σ[i,j]
        end
    end
    end
    return Σ
end

function get_covariance_matrix(M::AxisymmetricEquilibrium, orbits::Vector, sigma::Vector;
                               dt::Int = 1, Js::Vector{Vector{Float64}} = Vector{Float64}[], kwargs...)
    n = length(orbits)
    Σ = zeros(n,n)

    if isempty(Js)
        J = Array{Vector{Float64}}(undef, n)
        @inbounds Threads.@threads for i=1:n
            oi = orbits[i]
            J[i] = get_jacobian(M, oi; kwargs...)
        end
    else
        J = Js
    end

    @inbounds Threads.@threads for i=1:n
        oi = orbits[i]
        Ji = J[i]
        Σ[i,i] = get_covariance(M, oi, sigma; J = Ji, dt = dt)
        for j=(i+1):n
            oj = orbits[j]
            Jj = J[j]
            Σ[i,j] = get_covariance(M, oi, Ji, oj, Jj, sigma, dt)
            Σ[j,i] = Σ[i,j]
        end
    end
    return Σ
end
