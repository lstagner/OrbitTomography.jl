# Orbit Space Covariance

function get_global_covariance(c1::EPRCoordinate, ot1::Symbol,
                               c2::EPRCoordinate, ot2::Symbol,
                               p::AbstractVector, norms::AbstractVector)

    x = S3(c1.energy, c1.pitch, c1.r)
    y = S3(c2.energy, c2.pitch, c2.r)
    mu = (x .- y)./norms

    GO = Set((:trapped,:potato,:stagnation,:co_passing))
    same_ot = ((ot1 in GO) && (ot2 in GO)) || (ot1 == ot2)

    same_orb = c1 == c2

    np = length(p)
    if np == 2
        k = same_ot*(p[1]^2)*exp(-0.5*dot(mu,mu)/(p[2]^2)) + same_orb*1e-6
    elseif np == 3
        k = same_ot*(p[1]^2)*exp(-0.5*dot(mu,mu)/(p[2]^2)) + same_orb*p[3]^2
    elseif np == 4
        Σ_rho_inv = Diagonal(p[2:4].^(-2))
        k = same_ot*(p[1]^2)*exp(-0.5*dot(mu,Σ_rho_inv*mu)) + same_orb*1e-6
    elseif np == 5
        Σ_rho_inv = Diagonal(p[2:4].^(-2))
        k = same_ot*(p[1]^2)*exp(-0.5*dot(mu,Σ_rho_inv*mu)) + same_orb*p[5]^2
    else
        error("Unsupported number of parameters")
    end

    return k
end

function get_global_covariance(o1::Orbit, o2::Orbit, p::Vector; norms=S3(1.0,1.0,1.0))
    c1 = o1.coordinate
    c2 = o2.coordinate
    ot1 = o1.class
    ot2 = o2.class
    return get_global_covariance(c1,ot1,c2,ot2,p,norms)
end

function get_global_covariance_matrix(orbs, p::Vector; norms=S3(1.0,1.0,1.0))
    n = length(orbs)
    Σ = zeros(n,n)
    for i=1:n
        oi = orbs[i]
        for j = i:n
            oj = orbs[j]
            k = get_global_covariance(oi.coordinate, oi.class,
                                      oj.coordinate, oj.class,
                                      p, norms)
            Σ[i,j] = k
            Σ[j,i] = k
        end
    end
    return Σ
end

function get_global_covariance_matrix(orbs1::Vector, orbs2::Vector, p::Vector; norms=S3(1.0,1.0,1.0))
    n1 = length(orbs1)
    n2 = length(orbs2)
    Σ = zeros(n1,n2)
    for i=1:n1
        oi = orbs1[i]
        for j = 1:n2
            oj = orbs2[j]
            k = get_global_covariance(oi.coordinate, oi.class,
                                      oj.coordinate, oj.class,
                                      p, norms)
            Σ[i,j] = k
        end
    end
    return Σ
end

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

function eprz_kernel(x,y,Σ_inv)
    d = x .- y
    l = d'*Σ_inv*d
    return exp(-0.5*l)
end

function make_jacobian_spline(J::Vector{T}, t) where T<:Number
    Js = scale(interpolate(J, BSpline(Cubic(Periodic(OnGrid())))), t)
    return Js
end

make_jacobian_spline(J::T, t) where T<:AbstractInterpolation = J

function get_covariance(o1::OrbitSpline, J1::T, o2::OrbitSpline, J2::T, Σ_inv::S44, atol) where {T<:AbstractInterpolation}
    (length(o1) == 0 || length(o2) == 0) && return 0.0
    K, kerr = hcubature(x -> J1(x[1])*J2(x[2])*eprz_kernel(o1(x[1]), o2(x[2]), Σ_inv),
                        (0.0,0.0), (1.0,1.0), atol=atol)
    return abs(K)
end

function get_covariance(o1::OrbitSpline, J1::T, o2::OrbitSpline, J2::T, sigma::Vector, atol) where {T<:AbstractInterpolation}
    Σ_inv = S44(inv(Diagonal(sigma.^2)))
    return get_covariance(o1, J1, o2, J2, Σ_inv, atol)
end

function get_covariance(M::AxisymmetricEquilibrium, o, sigma;
                        J = Float64[], atol = 1e-3, kwargs...)

    n = length(o)
    t = range(0.0, 1.0, length = n)
    oi = OrbitSpline(o, t)
    if isempty(J)
        J1i = make_jacobian_spline(get_jacobian(M, o; kwargs...),t)
    else
        J1i = make_jacobian_spline(J, t)
    end

    return get_covariance(oi, J1i, oi, J1i, sigma, atol)
end

function get_covariance(M::AxisymmetricEquilibrium, o1, o2, sigma;
                        J1 = Float64[], J2 = Float64[], atol=1e-3, kwargs...)

    n1 = length(o1)
    n2 = length(o2)
    if n1 == 0 || n2 == 0
        return 0.0
    end

    t1 = range(0.0, 1.0, length=n1)
    o1i = OrbitSpline(o1, t1)

    t2 = range(0.0, 1.0, length=n2)
    o2i = OrbitSpline(o2, t2)

    if isempty(J1)
        J1i = make_jacobian_spline(get_jacobian(M, o1; kwargs...), t1)
    else
        J1i = make_jacobian_spline(J1, t1)
    end

    if isempty(J2)
        J2i = make_jacobian_spline(get_jacobian(M, o2; kwargs...), t2)
    else
        J2i = make_jacobian_spline(J2, t2)
    end

    return get_covariance(o1i, J1i, o2i, J2i, sigma, atol)
end

function get_covariance(M::AxisymmetricEquilibrium, c1::T, c2::T, sigma;
                         k1::Float64 = 0.0, k2::Float64 = 0.0,
                         J1 = Float64[], J2 = Float64[],
                        atol = 1e-3, kwargs...) where T <: Union{AbstractOrbitCoordinate,AbstractParticle}
    o1 = get_orbit(M, c1; kwargs...)
    o2 = get_orbit(M, c2; kwargs...)
    return get_covariance(M, o1, o2, sigma; J1 = J1, k1 = k1, J2 = J2, k2 = k2, atol = atol, kwargs...)
end

function get_correlation(M::AxisymmetricEquilibrium, o1, o2, sigma;
                         k1::Float64 = 0.0, k2::Float64 = 0.0,
                         J1 = Float64[], J2 = Float64[],
                         atol = 1e-3, kwargs...)

    n1 = length(o1)
    n2 = length(o2)
    if n1 == 0 || n2 == 0
        return 0.0
    end

    t1 = range(0.0, 1.0, length=n1)
    o1i = OrbitSpline(o1, t1)

    t2 = range(0.0, 1.0, length=n2)
    o2i = OrbitSpline(o2, t2)

    if isempty(J1)
        J1i = make_jacobian_spline(get_jacobian(M, o1; kwargs...), t1)
    else
        J1i = make_jacobian_spline(J1, t1)
    end

    if isempty(J2)
        J2i = make_jacobian_spline(get_jacobian(M, o2; kwargs...), t2)
    else
        J2i = make_jacobian_spline(J2, t2)
    end

    if k1 == 0.0
        k11 = get_covariance(o1i, J1i, o1i, J1i, sigma, atol)
    else
        k11 = k1
    end

    if k2 == 0.0
        k22 = get_covariance(o2i, J2i, o2i, J2i, sigma, atol)
    else
        k22 = k2
    end

    if k11 == 0.0 || k22 == 0.0
        return k11*k22
    end

    k12 = get_covariance(o1i, J1i, o2i, J2i, sigma, atol)

    return k12/sqrt(k11*k22)
end

#1-arg Threaded Covariance
function compute_covariance_matrix(orbs, Js, Σ_inv, atol)
    n = length(orbs)
    N = Threads.nthreads()
    Σ = zeros(n,n)
    Threads.@threads for id=1:N
        @inbounds for i = id:N:n
            oi = orbs[i]
            Ji = Js[i]
            for j=i:n
                oj = orbs[j]
                Jj = Js[j]
                s = get_covariance(oi, Ji, oj, Jj, Σ_inv, atol)
                Σ[i,j] = s
                Σ[j,i] = s
            end
        end
    end

    return Σ
end

#1-arg Threaded Sparse Covariance
function compute_covariance_matrix(orbs, Js, Σ_inv, atol, minval)
    n = length(orbs)
    N = Threads.nthreads()
    Σs = [spzeros(n,n) for i=1:N]
    Threads.@threads for id=1:N
        @inbounds for i=id:N:n
            oi = orbs[i]
            Ji = Js[i]
            Σs[id][i,i] = 1.0
            for j=(i+1):n
                oj = orbs[j]
                Jj = Js[j]
                s = get_covariance(oi, Ji, oj, Jj, Σ_inv, atol)
                if s > minval
                    Σs[id][i,j] = s
                    Σs[id][j,i] = s
                end
            end
        end
    end
    Σ = sum(Σs)

    return Σ
end

#1-arg Distributed Covariance
function compute_covariance_matrix(orbs, Js, Σ_inv, atol, pool::AbstractWorkerPool; batch_size = 0)
    n = length(orbs)
    indices = [(i,j) for i=1:n, j=1:n if i >= j]

    if batch_size == 0
        batch_size = round(Int, length(indices)/(5*nprocs()))
    end

    c = pmap(x -> get_covariance(orbs[x[1]],Js[x[1]],orbs[x[2]],Js[x[2]], Σ_inv, atol),
             pool, indices, on_error=x->0.0, batch_size=batch_size)

    Σ = zeros(n,n)
    for (ii,I) in enumerate(indices)
        i,j = I
        Σ[i,j] = c[ii]
        Σ[j,i] = c[ii]
    end

    return Σ
end

#1-arg Distributed Sparse Covariance
function compute_covariance_matrix(orbs, Js, Σ_inv, atol, minval, pool::AbstractWorkerPool; batch_size=0)
    Σ = compute_covariance_matrix(orbs, Js, Σ_inv, atol, pool, batch_size=batch_size)
    Σ[Σ .< minval] .= zero(eltype(Σ))

    return sparse(Σ)
end

#1-arg Covariance
function get_covariance_matrix(M::AxisymmetricEquilibrium, orbits::Vector, sigma::Vector;
                               Js::Vector{Vector{Float64}} = Vector{Float64}[],
                               sparse::Bool = false, atol::Float64 = 1e-3,
                               minval::Float64 = atol, distributed = false,
                               batch_size = 0, kwargs...)
    n = length(orbits)
    ns = length.(orbits)
    ts = [range(0.0, 1.0, length=nn) for nn in ns]
    orbs = [OrbitSpline(o) for o in orbits]

    if isempty(Js)
        J = Array{Vector{Float64}}(undef, n)
        @inbounds Threads.@threads for i=1:n
            oi = orbits[i]::Orbit
            Ji = get_jacobian(M, oi; kwargs...)
            J[i] = Ji
        end
    else
        J = Js
    end
    Jis = [make_jacobian_spline(jj,tt) for (jj, tt) in zip(J,ts)]

    Σ_inv = S44(inv(Diagonal(sigma.^2)))

    if distributed
        pool = CachingPool(workers())
        if sparse
            Σ = compute_covariance_matrix(orbs, Jis, Σ_inv, atol, minval, pool; batch_size=batch_size)
        else
            Σ = compute_covariance_matrix(orbs, Jis, Σ_inv, atol, pool; batch_size=batch_size)
        end
        clear!(pool)
        return Σ
    else
        if sparse
            return compute_covariance_matrix(orbs, Jis, Σ_inv, atol, minval)
        end
        return compute_covariance_matrix(orbs, Jis, Σ_inv, atol)
    end
end

#1-arg Correlation
function get_correlation_matrix(cov)
    nr,nc = size(cov)
    nr != nc && error("Covariance matrix must be square")
    K = sqrt.(diag(cov))
    return cov./(K*K')
end

function get_correlation_matrix(M::AxisymmetricEquilibrium, orbits::Vector, sigma::Vector;
                                Js::Vector{Vector{Float64}} = Vector{Float64}[],
                                sparse::Bool = false, atol::Float64 = 1e-3,
                                minval::Float64 = atol, distributed = false,
                                batch_size = 0, kwargs...)

    Σ = get_covariance_matrix(M, orbits, sigma; Js=Js, sparse=sparse,
                              atol=atol, minval=minval, distributed=distributed,
                              batch_size=batch_size, kwargs...)

    return get_correlation_matrix(Σ)
end

#2-arg Threaded Covariance
function compute_covariance_matrix(orbs1, Js1, orbs2, Js2, Σ_inv, atol)
    n1 = length(orbs1)
    n2 = length(orbs2)
    Σ = zeros(n1,n2)
    N = Threads.nthreads()
    Threads.@threads for id=1:N
        @inbounds for j=id:N:n2
            oj = orbs2[j]
            Jj = Js2[j]
            for i=1:n1
                oi = orbs1[i]
                Ji = Js1[i]
                Σ[i,j] = get_covariance(oi, Ji, oj, Jj, Σ_inv, atol)
            end
        end
    end
    return Σ
end

#2-arg Threaded Sparse Covariance
function compute_covariance_matrix(orbs1, Js1, orbs2, Js2, Σ_inv, atol, minval)
    n1 = length(orbs1)
    n2 = length(orbs2)
    N = Threads.nthreads()
    Σs = [spzeros(n1,n2) for i=1:N]
    Threads.@threads for id=1:N
        @inbounds for j=id:N:n2
            oj = orbs2[j]
            Jj = Js2[j]
            for i=1:n1
                oi = orbs1[i]
                Ji = Js1[i]
                s = get_covariance(oi, Ji, oj, Jj, Σ_inv, atol)
                if s > minval
                    Σs[id][i,j] = s
                end
            end
        end
    end
    Σ = sum(Σs)
    return Σ
end

#2-arg Distributed Covariance
function compute_covariance_matrix(orbs1, Js1, orbs2, Js2, Σ_inv, atol, pool::AbstractWorkerPool; batch_size=0)
    n1 = length(orbs1)
    n2 = length(orbs2)
    indices = vec([(i,j) for i=1:n1, j=1:n2])

    if batch_size == 0
        batch_size = round(Int, length(indices)/(5*nprocs()))
    end

    Σ = pmap(x -> get_covariance(orbs1[x[1]],Js1[x[1]],orbs2[x[2]],Js2[x[2]], Σ_inv, atol),
             pool, indices, on_error=x->0.0, batch_size=batch_size)

    return reshape(Σ,(n1,n2))
end

#2-arg Distributed Sparse Covariance
function compute_covariance_matrix(orbs1, Js1, orbs2, Js2, Σ_inv, atol, minval, pool::AbstractWorkerPool; batch_size=0)
    Σ = compute_covariance_matrix(orbs1, Js1, orbs2, Js2, Σ_inv, atol, pool; batch_size=batch_size)
    Σ[Σ .< minval] .= zero(eltype(Σ))
    return sparse(Σ)
end

#2-arg Covariance
function get_covariance_matrix(M::AxisymmetricEquilibrium, orbits_1::Vector, orbits_2::Vector, sigma::Vector;
                                Js_1::Vector{Vector{Float64}} = Vector{Float64}[],
                                Js_2::Vector{Vector{Float64}} = Vector{Float64}[],
                                sparse::Bool = false, atol::Float64 = 1e-3,
                                minval::Float64 = atol, distributed=false,
                                batch_size=0, kwargs...)

    Σ_inv = S44(inv(Diagonal(sigma.^2)))

    n1 = length(orbits_1)
    ns1 = length.(orbits_1)
    ts1 = [range(0.0, 1.0, length=nn) for nn in ns1]
    orbs1 = [OrbitSpline(o) for o in orbits_1]

    if isempty(Js_1)
        J1 = Array{Vector{Float64}}(undef, n1)
        @inbounds Threads.@threads for i=1:n1
            oi = orbits_1[i]::Orbit
            Ji = get_jacobian(M, oi; kwargs...)
            J1[i] = Ji
        end
    else
        J1 = Js_1
    end
    Jis1 = [make_jacobian_spline(jj,tt) for (jj, tt) in zip(J1,ts1)]

    n2 = length(orbits_2)
    ns2 = length.(orbits_2)
    ts2 = [range(0.0, 1.0, length=nn) for nn in ns2]
    orbs2 = [OrbitSpline(o) for o in orbits_2]

    if isempty(Js_2)
        J2 = Array{Vector{Float64}}(undef, n2)
        @inbounds Threads.@threads for i=1:n2
            oi = orbits_2[i]::Orbit
            Ji = get_jacobian(M, oi; kwargs...)
            J2[i] = Ji
        end
    else
        J2 = Js_2
    end
    Jis2 = [make_jacobian_spline(jj,tt) for (jj, tt) in zip(J2,ts2)]

    if distributed
        pool = CachingPool(workers())
        if sparse
            Σ = compute_covariance_matrix(orbs1, Jis1, orbs2, Jis2, Σ_inv, atol, minval, pool; batch_size=batch_size)
        else
            Σ = compute_covariance_matrix(orbs1, Jis1, orbs2, Jis2, Σ_inv, atol, pool; batch_size=batch_size)
        end
        clear!(pool)
        return Σ
    else
        if sparse
            return compute_covariance_matrix(orbs1, Jis1, orbs2, Jis2, Σ_inv, atol, minval)
        end
        return compute_covariance_matrix(orbs1, Jis1, orbs2, Jis2, Σ_inv, atol)
    end
end

#2-arg Threaded Correlation
function compute_correlation_matrix(orbs1, Js1, Ks1, orbs2, Js2, Ks2, Σ_inv, atol)
    n1 = length(orbs1)
    n2 = length(orbs2)
    Σ = zeros(n1,n2)
    N = Threads.nthreads()
    Threads.@threads for id=1:N
        @inbounds for j=id:N:n2
            oj = orbs2[j]
            Jj = Js2[j]
            kj = Ks2[j]
            for i=1:n1
                oi = orbs1[i]
                Ji = Js1[i]
                ki = Ks1[i]
                Σ[i,j] = get_covariance(oi, Ji, oj, Jj, Σ_inv, atol)/sqrt(ki*kj)
            end
        end
    end
    return Σ
end

#2-arg Threaded Sparse Correlation
function compute_correlation_matrix(orbs1, Js1, Ks1, orbs2, Js2, Ks2, Σ_inv, atol, minval)
    n1 = length(orbs1)
    n2 = length(orbs2)
    N = Threads.nthreads()
    Σs = [spzeros(n1,n2) for i=1:N]
    Threads.@threads for id=1:N
        @inbounds for j=id:N:n2
            oj = orbs2[j]
            Jj = Js2[j]
            kj = Ks2[j]
            for i=1:n1
                oi = orbs1[i]
                Ji = Js1[i]
                ki = Ks1[i]
                s = get_covariance(oi, Ji, oj, Jj, Σ_inv, atol)/sqrt(ki*kj)
                if s > minval
                    Σs[id][i,j] = s
                end
            end
        end
    end
    Σ = sum(Σs)
    return Σ
end

#2-arg Distributed Correlation
function compute_correlation_matrix(orbs1, Js1, Ks1, orbs2, Js2, Ks2, Σ_inv, atol, pool::AbstractWorkerPool; batch_size=0)
    n1 = length(orbs1)
    n2 = length(orbs2)
    indices = vec([(i,j) for i=1:n1, j=1:n2])

    if batch_size == 0
        batch_size = round(Int, length(indices)/(5*nprocs()))
    end

    Σ = pmap(x -> get_covariance(orbs1[x[1]],Js1[x[1]],orbs2[x[2]],Js2[x[2]], Σ_inv, atol)/sqrt(Ks1[x[1]]*Ks2[x[2]]),
             pool, indices, on_error=x->0.0, batch_size=batch_size)

    return reshape(Σ,(n1,n2))
end

#2-arg Distributed Sparse Correlation
function compute_correlation_matrix(orbs1, Js1, Ks1, orbs2, Js2, Ks2, Σ_inv, atol, minval, pool::AbstractWorkerPool; batch_size=0)
    Σ = compute_correlation_matrix(orbs1, Js1, Ks1, orbs2, Js2, Ks2, Σ_inv, atol, pool; batch_size=batch_size)
    Σ[Σ .< minval] .= zero(eltype(Σ))
    return sparse(Σ)
end

function get_correlation_matrix(M::AxisymmetricEquilibrium, orbits_1::Vector, orbits_2::Vector, sigma::Vector;
                                ks_1::Vector{Float64} = Float64[],
                                Js_1::Vector{Vector{Float64}} = Vector{Float64}[],
                                ks_2::Vector{Float64} = Float64[],
                                Js_2::Vector{Vector{Float64}} = Vector{Float64}[],
                                sparse::Bool = false, atol::Float64 = 1e-3,
                                minval::Float64 = atol, distributed=false,
                                batch_size = 0, kwargs...)

    Σ_inv = S44(inv(Diagonal(sigma.^2)))

    n1 = length(orbits_1)
    ns1 = length.(orbits_1)
    ts1 = [range(0.0, 1.0, length=nn) for nn in ns1]
    orbs1 = [OrbitSpline(o) for o in orbits_1]

    if isempty(Js_1)
        J1 = Array{Vector{Float64}}(undef, n1)
        @inbounds Threads.@threads for i=1:n1
            oi = orbits_1[i]::Orbit
            Ji = get_jacobian(M, oi; kwargs...)
            J1[i] = Ji
        end
    else
        J1 = Js_1
    end
    Jis1 = [make_jacobian_spline(jj,tt) for (jj, tt) in zip(J1,ts1)]

    if isempty(ks_1)
        K1 = zeros(n1)
        @inbounds Threads.@threads for i=1:n1
            oi = orbs1[i]
            Ji = Jis1[i]
            K1[i] = get_covariance(oi, Ji, oi, Ji, Σ_inv, atol)
        end
    else
        K1 = ks_1
    end

    n2 = length(orbits_2)
    ns2 = length.(orbits_2)
    ts2 = [range(0.0, 1.0, length=nn) for nn in ns2]
    orbs2 = [OrbitSpline(o) for o in orbits_2]

    if isempty(Js_2)
        J2 = Array{Vector{Float64}}(undef, n2)
        @inbounds Threads.@threads for i=1:n2
            oi = orbits_2[i]::Orbit
            Ji = get_jacobian(M, oi; kwargs...)
            J2[i] = Ji
        end
    else
        J2 = Js_2
    end
    Jis2 = [make_jacobian_spline(jj,tt) for (jj, tt) in zip(J2,ts2)]

    if isempty(ks_2)
        K2 = zeros(n2)
        @inbounds Threads.@threads for i=1:n2
            oi = orbs2[i]
            Ji = Jis2[i]
            K2[i] = get_covariance(oi, Ji, oi, Ji, Σ_inv, atol)
        end
    else
        K2 = ks_2
    end

    if distributed
        pool = CachingPool(workers())
        if sparse
            Σ = compute_correlation_matrix(orbs1, Jis1, K1, orbs2, Jis2, K2, Σ_inv, atol, minval, pool, batch_size=batch_size)
        else
            Σ = compute_correlation_matrix(orbs1, Jis1, K1, orbs2, Jis2, K2, Σ_inv, atol, pool, batch_size=batch_size)
        end
        clear!(pool)
        return Σ
    else
        if sparse
            return compute_correlation_matrix(orbs1, Jis1, K1, orbs2, Jis2, K2, Σ_inv, atol, minval)
        end
        return compute_correlation_matrix(orbs1, Jis1, K1, orbs2, Jis2, K2, Σ_inv, atol)
    end

end
