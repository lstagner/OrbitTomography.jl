# --- FIDASIMSpectra Helper Functions ---
"""
    split_spectra(myFIDASIMSpectra)

Split FIDASIM spectra into many spectra, according to orbit class. Orbit class is
merely a unique orbit (unique EPR coordinate). The returned array should allow the user to 
manipulate each array element (spectra) with apply_instrumental! etc.
"""
function split_spectra(s::FIDASIMSpectra)
    (length(size(s.fida)) == 2) && (length(size(s.pfida)) == 2) && return s

    nlambda, nchan, nclass = size(s.fida)

    ss = Array{typeof(s)}(undef,nclass)
    for i=1:nclass
        if typeof(s.fida) <: Zeros
            dims = size(s.fida)
            fida = Zeros(dims[1],dims[2])
        else
            fida = s.fida[:,:,i]
        end
        if typeof(s.pfida) <: Zeros
            dims = size(s.pfida)
            pfida = Zeros(dims[1],dims[2])
        else
            pfida = s.pfida[:,:,i]
        end
        sss = FIDASIMSpectra(nchan, nlambda, s.lambda, s.brems, s.full, s.half, s.third,
                             s.dcx, s.halo, s.cold, fida, pfida)
        ss[i] = sss
    end

    return ss
end

function Base.hcat(x::Zeros{T,2}...) where T <: Real
    all(xx -> size(xx)[1] == size(x[1])[1], x) || throw(ArgumentError("mismatch in dimension 1"))
    n = sum(size(xx)[2] for xx in x)

    return Zeros(size(x[1])[1],n)
end

function Base.hcat(x::Zeros{T,3}...) where T <: Real
    all(xx -> size(xx)[1] == size(x[1])[1], x) || throw(ArgumentError("mismatch in dimension 1"))
    n = sum(size(xx)[2] for xx in x)

    return Zeros(size(x[1])[1],n,size(x[1])[3])
end

"""
    merge_spectra(FIDASIMSpectra1,FIDASIMSpectra2,...)

Merge many FIDASIM spectra into one.
"""
function merge_spectra(s::FIDASIMSpectra...)
    all(x -> length(size(x.fida)) == length(size(s[1].fida)), s) || error("Incompatible fida sizes")
    all(x -> length(x.lambda) == length(s[1].lambda), s) || error("Incompatible lambda sizes")
    all(x -> x.lambda[1] == s[1].lambda[1], s) || error("Incompatible lambda min")
    all(x -> x.lambda[end] == s[1].lambda[end], s) || error("Incompatible lambda max")

    nchan = sum(ss.nchan for ss in s)
    lambda = s[1].lambda
    nlambda = s[1].nlambda

    brems = hcat((ss.brems for ss in s)...)
    full = hcat((ss.full for ss in s)...)
    half = hcat((ss.half for ss in s)...)
    third = hcat((ss.third for ss in s)...)
    dcx = hcat((ss.dcx for ss in s)...)
    halo = hcat((ss.halo for ss in s)...)
    cold = hcat((ss.cold for ss in s)...)
    fida = hcat((ss.fida for ss in s)...)
    pfida = hcat((ss.pfida for ss in s)...)

    return FIDASIMSpectra(nchan, nlambda, lambda, brems, full, half, third, dcx, halo, cold, fida, pfida)
end

function TheoreticalSpectra(S::FIDASIMSpectra, stype::Symbol, ichan::Int, iclass::Int=1)
    spec = getfield(S,stype)
    TheoreticalSpectra(S.lambda,spec[:,ichan,iclass])
end

function apply_instrumental!(s::FIDASIMSpectra,instr::Vector,dL::Vector)
    ni = length(instr)
    if ni != s.nchan
        error("Instrumental has wrong shape")
    end

    for i=1:s.nchan
        k = kernel(s.lambda,instr[i],dL[i])
        length(eachindex(k)) == 0 && continue
        !(typeof(s.brems) <: Zeros)  && (s.brems[:,i]  .= imfilter(s.brems[:,i], k))
        !(typeof(s.full) <: Zeros)  && (s.full[:,i]  .= imfilter(s.full[:,i], k))
        !(typeof(s.half) <: Zeros)  && (s.half[:,i]  .= imfilter(s.half[:,i], k))
        !(typeof(s.third) <: Zeros) && (s.third[:,i] .= imfilter(s.third[:,i], k))
        !(typeof(s.dcx) <: Zeros)   && (s.dcx[:,i]   .= imfilter(s.dcx[:,i], k))
        !(typeof(s.halo) <: Zeros)  && (s.halo[:,i]  .= imfilter(s.halo[:,i], k))
        !(typeof(s.cold) <: Zeros)  && (s.cold[:,i]  .= imfilter(s.cold[:,i], k))
        !(typeof(s.fida) <: Zeros)  && (s.fida[:,i]  .= imfilter(s.fida[:,i], k))
        !(typeof(s.pfida) <: Zeros) && (s.pfida[:,i] .= imfilter(s.pfida[:,i], k))
    end
end

function apply_instrumental!(s::FIDASIMSpectra, instr::AbstractVector)
    dL = fill(abs(s.lambda[2]-s.lambda[1]),s.nchan)
    apply_instrumental!(s, instr, dL)
end

function apply_instrumental!(s::FIDASIMSpectra, i)
    instr = fill(i, s.nchan)
    dL = fill(abs(s.lambda[2]-s.lambda[1]),s.nchan)
    apply_instrumental!(s, instr, dL)
end

# --- FIDASIM Distributions Helper Functions ---
function split_particles(d::FIDASIMGuidingCenterParticles)

    d.nclass <= 1 && return d

    darr = Array{typeof(d)}(undef,d.nclass)
    for c = 1:d.nclass
        w = d.class .== c
        npart = sum(w)
        dd = FIDASIMGuidingCenterParticles(npart,1,fill(c,npart),
                                           d.weight[w], d.r[w], d.z[w],
                                           d.energy[w],d.pitch[w],sum(d.weight[w]))
        darr[c] = dd
    end

    return darr
end

function split_particles(d::FIDASIMFullOrbitParticles)

    d.nclass <= 1 && return d

    darr = Array{typeof(d)}(undef,d.nclass)
    for c = 1:d.nclass
        w = d.class .== c
        npart = sum(w)
        dd = FIDASIMFullOrbitParticles(npart,1,fill(1,npart),
                                       d.weight[w], d.r[w], d.z[w],
                                       d.vr[w], d.vt[w], d.vz[w],sum(d.weight[w]))
        darr[c] = dd
    end

    return darr
end

function sample_array(x::Array{T,N},ns) where {T,N}
    w_tot = sum(x)
    l = length(x)
    dims = size(x)
    x_cum = cumsum(vec(x))
    inds = zeros(Int,N,ns)
    subs = CartesianIndices(dims)
    @inbounds for i=1:ns
        p = rand()*w_tot
        j = searchsortedfirst(x_cum,p,Base.Order.Forward)
        inds[:,i] = collect(Tuple(subs[j]))
    end
    return inds
end

function sample_f(f::Array{T,N}, w, x, y, z; n=100) where {T,N}

    inds = sample_array(f,n)
    dw = abs(w[2]-w[1])
    dx = abs(x[2]-x[1])
    dy = abs(y[2]-y[1])
    dz = abs(z[2]-z[1])

    r = rand(N,n) .- 0.5
    xx = zeros(n)
    yy = zeros(n)
    zz = zeros(n)
    ww = zeros(n)
    o = Array{NTuple{4,Float64}}(undef,n)
    @inbounds for i=1:n
        ww[i] = max(w[inds[1,i]] + r[1,i]*dw, 0.0)
        xx[i] = x[inds[2,i]] + r[2,i]*dx
        yy[i] = y[inds[3,i]] + r[3,i]*dy
        zz[i] = z[inds[4,i]] + r[4,i]*dz
    end

    return ww, xx, yy, zz
end

"""
    sample_f(fr,dvols,energy,pitch,R,Z)
    sample_f(-||-,n=100_000)

Unlike regular sample_f, take non-equidistant 4D grid-points into consideration.
"""
function sample_f(fr::Array{T,N}, dvols::AbstractArray, w, x, y, z; n=100_000) where {T,N}

    inds = sample_array(fr.*dvols,n)

    dw = vcat(abs.(diff(w)),abs(w[end]-w[end-1]))
    dx = vcat(abs.(diff(x)),abs(x[end]-x[end-1]))
    dy = vcat(abs.(diff(y)),abs(y[end]-y[end-1]))
    dz = vcat(abs.(diff(z)),abs(z[end]-z[end-1]))

    r = rand(N,n) .- 0.5
    xx = zeros(n)
    yy = zeros(n)
    zz = zeros(n)
    ww = zeros(n)

    @inbounds for i=1:n
        ww[i] = max(w[inds[1,i]] + r[1,i]*dw[inds[1,i]], 0.0)
        xx[i] = x[inds[2,i]] + r[2,i]*dx[inds[2,i]]
        yy[i] = y[inds[3,i]] + r[3,i]*dy[inds[3,i]]
        zz[i] = z[inds[4,i]] + r[4,i]*dz[inds[4,i]]
    end

    return ww, xx, yy, zz
end

function fbm2mc(d::FIDASIMGuidingCenterFunction; n=1_000_000)
    fr = d.f .* reshape(d.r,(1,1,length(d.r),1))

    energy, pitch, r, z = sample_f(fr,d.energy,d.pitch,d.r,d.z,n=n)
    weight = fill(d.nfast/n,n)
    nclass = 1
    class = fill(1,n)

    return FIDASIMGuidingCenterParticles(n,nclass,class,weight,r,z,energy,pitch,d.nfast)
end

"""
    fbm2mc(F_ps, equidistant)
    fbm2mc(-||-, n=100_000)

Unlike regular fbm2mc, take non-equidistant 4D grid-points into consideration if equidistant=false. 
Assume 4D grid to be rectangular.
"""
function fbm2mc(d::FIDASIMGuidingCenterFunction, equidistant::Bool; n=1_000_000)

    fr = d.f .* reshape(d.r,(1,1,length(d.r),1))
    if equidistant
        energy, pitch, r, z = sample_f(fr,d.energy,d.pitch,d.r,d.z,n=n)
    else
        dvols = get4DVols(d.energy,d.pitch,d.r,d.z)
        energy, pitch, r, z = sample_f(fr,dvols,d.energy,d.pitch,d.r,d.z,n=n)
    end

    weight = fill(d.nfast/n,n)
    nclass = 1
    class = fill(1,n)

    return FIDASIMGuidingCenterParticles(n,nclass,class,weight,r,z,energy,pitch,d.nfast)
end

# --- FIDASIM Spectra Geometry Helper Functions ---
function merge_spectra_geometry(s::FIDASIMSpectraGeometry...)
    length(s) == 1 && return s

    nchan = sum(ss.nchan for ss in s)
    system = join((ss.system for ss in s),"; ")
    id = vcat((ss.id for ss in s)...)
    radius = vcat((ss.radius for ss in s)...)
    lens = hcat((ss.lens for ss in s)...)
    axis = hcat((ss.axis for ss in s)...)
    spot_size = vcat((ss.spot_size for ss in s)...)
    sigma_pi = vcat((ss.sigma_pi for ss in s)...)

    return FIDASIMSpectraGeometry(nchan,system,id,radius,lens,axis,spot_size,sigma_pi)
end

# --- FIDASIM Plasma Profiles Helper Functions ---
function impurity_density(s::FIDASIMPlasmaParameters, imp)
    zeff = clamp.(s.zeff, 1, imp)
    return ((zeff .- 1)./(imp*(imp-1))) .* s.dene
end

function ion_density(s::FIDASIMPlasmaParameters, imp)
    denimp = impurity_density(s, imp)
    return s.dene .- imp*denimp
end


# --- FIDASIM Velocity-space Weight Functions Helper Functions ---
function apply_instrumental!(s::FIDAWeightFunction, instr, dL)
    for ie=1:length(s.energy), ip=1:length(s.pitch)
        k = kernel(s.lambda,instr,dL)
        length(eachindex(k)) == 0 && continue
        s.W[:,ie,ip] .= imfilter(s.W[:,ie,ip], k)
    end
end

function apply_instrumental!(s::FIDAWeightFunction, instr)
    dL = abs(s.lambda[2]-s.lambda[1])
    apply_instrumental!(s, instr, dL)
end

function weight_matrix(s::FIDAWeightFunction, lambda)
    nenergy = length(s.energy)
    npitch = length(s.pitch)
    nL = length(lambda)
    WW = zeros(nL,nenergy,npitch)
    for ie=1:nenergy, ip=1:npitch
        itp = LinearInterpolation(s.lambda,s.W[:,ie,ip])
        WW[:,ie,ip] .= itp.(lambda)
    end

    return reshape(WW,(nL,nenergy*npitch))
end

function make_synthetic_weight_matrix(w::FIDAWeightFunction, s::FIDASIMSpectra, ic; fida_fraction=0.95,dL=0.1)
    bes = s.full[:,ic] .+ s.half[:,ic] .+ s.third[:,ic] .+ s.dcx[:,ic] .+ s.halo[:,ic]
    fida = s.fida[:,ic]
    bes_itp = LinearInterpolation(s.lambda,bes)
    fida_itp = LinearInterpolation(s.lambda,fida)

    lambda = range(extrema(w.lambda)...,step=dL)
    f = fida_itp.(lambda)
    b = bes_itp.(lambda)
    ww = (f./(f .+ b)) .> fida_fraction
    return weight_matrix(w,lambda[ww])
end
