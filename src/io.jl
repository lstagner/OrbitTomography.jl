struct FIDASIMSpectra{T<:Real}
    nchan::Int
    nlambda::Int
    lambda::Vector{T}
    full::Matrix{T}
    half::Matrix{T}
    third::Matrix{T}
    dcx::Matrix{T}
    halo::Matrix{T}
    cold::Matrix{T}
    fida::Matrix{T}
    pfida::Matrix{T}
end

function FIDASIMSpectra(fname::String)
    isfile(fname) || error("File does not exist")

    f = h5open(fname)
    nchan = read(f["nchan"])
    lambda = read(f["lambda"])
    nlam = length(lambda)

    full = "full" in names(f) ? read(f["full"]) : zeros(nlam,nchan)
    half = "half" in names(f) ? read(f["half"]) : zeros(nlam,nchan)
    third = "third" in names(f) ? read(f["third"]) : zeros(nlam,nchan)
    dcx = "dcx" in names(f) ? read(f["dcx"]) : zeros(nlam,nchan)
    halo = "halo" in names(f) ? read(f["halo"]) : zeros(nlam,nchan)
    cold = "cold" in names(f) ? read(f["cold"]) : zeros(nlam,nchan)
    fida = "fida" in names(f) ? read(f["fida"]) : zeros(nlam,nchan)
    pfida = "pfida" in names(f) ? read(f["pfida"]) : zeros(nlam,nchan)

    return FIDASIMSpectra(Int(nchan),Int(nlam),lambda,full,half,third,dcx,halo,cold,fida,pfida)
end

function Base.show(io::IO, s::FIDASIMSpectra)
    println(io, "FIDASIMSpectra")
    println(io, "  nchan   = $(s.nchan)")
    println(io, "  nlambda = $(s.nlambda)")
end

struct InstrumentalResponse
    amp::Vector{Float64}
    width::Vector{Float64}    #in pixels
    location::Vector{Float64} #in pixels
end

function (IR::InstrumentalResponse)(lambda,d=1.0)
    #d converts from pixels to wavelength
    sum([a*exp((-((d*l) - lambda)^2)/(d*w)^2) for (a,l,w) in zip(IR.amp,IR.location,IR.width)])
end

function kernel(lambda,IR::InstrumentalResponse,d=1.0)
    dL = lambda[2] - lambda[1]
    lmax = maximum(abs.(d*IR.location)) + 4*maximum(sqrt.(((d*IR.width).^2)/2))
    Lp = 0.0:dL:lmax
    L= vcat(-reverse(Lp)[1:end-1],Lp)
    k = IR.(L)
    return centered(k./sum(k))
end

function kernel(lambda,IR::T,d=1.0) where T<:Real
    return Images.Kernel.gaussian((IR/d,), (21,))
end

function apply_instrumental!(s::FIDASIMSpectra,instr::Vector,dL=0.0)
    ni = length(instr)
    if ni != s.nchan
        error("Instrumental has wrong shape")
    end

    if dL == 0.0
        dL = abs(s.lambda[2] - s.lambda[1])
    end
    for i=1:s.nchan
        k = kernel(s.lambda,instr[i],dL)
        s.full[:,i]  .= imfilter(s.full[:,i], k)
        s.half[:,i]  .= imfilter(s.half[:,i], k)
        s.third[:,i] .= imfilter(s.third[:,i], k)
        s.dcx[:,i]   .= imfilter(s.dcx[:,i], k)
        s.halo[:,i]  .= imfilter(s.halo[:,i], k)
        s.cold[:,i]  .= imfilter(s.cold[:,i], k)
        s.fida[:,i]  .= imfilter(s.fida[:,i], k)
        s.pfida[:,i] .= imfilter(s.pfida[:,i], k)
    end
end

function apply_instrumental!(s::FIDASIMSpectra, i)
    instr = fill(i, s.nchan)
    apply_instrumental!(s, instr)
end

abstract type AbstractDistribution end

struct FIDASIMGuidingCenterFunction <: AbstractDistribution
    nr::Int
    nz::Int
    nenergy::Int
    npitch::Int
    r::Vector{Float64}
    z::Vector{Float64}
    energy::Vector{Float64}
    pitch::Vector{Float64}
    denf::Matrix{Float64}
    f::Array{Float64,4}
end

function FIDASIMGuidingCenterFunction(f::HDF5.HDF5File)

    nr = read(f["nr"])
    nz = read(f["nz"])
    nenergy = read(f["nenergy"])
    npitch = read(f["npitch"])

    energy = read(f["energy"])
    pitch = read(f["pitch"])
    r = read(f["r"])
    z = read(f["z"])
    denf = read(f["denf"])
    dist = read(f["f"])

    return FIDASIMGuidingCenterFunction(nr,nz,nenergy,npitch,r,z,energy,pitch,denf,dist)
end

struct FIDASIMGuidingCenterParticles <: AbstractDistribution
    npart::Int
    nclass::Int
    class::Vector{Int}
    weight::Vector{Float64}
    r::Vector{Float64}
    z::Vector{Float64}
    energy::Vector{Float64}
    pitch::Vector{Float64}
end

function FIDASIMGuidingCenterParticles(f::HDF5.HDF5File)

    npart = read(f["nparticle"])
    nclass = read(f["nclass"])
    class = read(f["class"])
    weight = read(f["weight"])

    energy = read(f["energy"])
    pitch = read(f["pitch"])
    r = read(f["r"])
    z = read(f["z"])

    return FIDASIMGuidingCenterParticles(npart,nclass,class,weight,r,z,energy,pitch)
end

struct FIDASIMFullOrbitParticles <: AbstractDistribution
    npart::Int
    nclass::Int
    class::Vector{Int}
    weight::Vector{Float64}
    r::Vector{Float64}
    z::Vector{Float64}
    vr::Vector{Float64}
    vt::Vector{Float64}
    vz::Vector{Float64}
end

function FIDASIMFullOrbitParticles(f::HDF5.HDF5File)

    npart = read(f["nparticle"])
    nclass = read(f["nclass"])
    class = read(f["class"])
    weight = read(f["weight"])

    r = read(f["r"])
    z = read(f["z"])
    vr = read(f["vr"])
    vt = read(f["vt"])
    vz = read(f["vz"])

    return FIDASIMFullOrbitParticles(npart,nclass,class,weight,r,z,vr,vt,vz)
end

function read_fidasim_distribution(f::String)
    isfile(filename) || error("File does not exist")

    f = h5open(filename)
    dtype = read(f["type"])

    if dtype == 1
        d = FIDASIMGuidingCenterFunction(f)
    elseif dtype == 2
        d = FIDASIMGuidingCenterParticles(f)
    elseif dtype == 3
        d = FIDASIMFullOrbitParticles(f)
    else
        error("Unknown distribution type")
    end
    close(f)

    return d
end

function write_fidasim_distribution(M::AxisymmetricEquilibrium, orbits; filename="orbits.h5",time=0.0)

    norbs = length(orbits)
    energy = Float64[]
    pitch = Float64[]
    r = Float64[]
    z = Float64[]
    weight = Float64[]
    class = Int16[]
    tau_t = Float64[]
    tau_p = Float64[]
    energy_c = Float64[]
    pitch_c = Float64[]
    r_c = Float64[]
    z_c = Float64[]

    for (i,o) in enumerate(orbits)
        append!(energy,o.path.energy)
        append!(pitch, M.sigma*o.path.pitch)
        append!(r, o.path.r*100)
        append!(z, o.path.z*100)
        append!(class, fill(Int16(i),length(o.path.r)))
        append!(weight, o.path.dt.*(1e19/sum(o.path.dt)))
        push!(tau_t, o.tau_t)
        push!(tau_p, o.tau_p)
        if isa(o.coordinate,EPRCoordinate)
            push!(energy_c, o.coordinate.energy)
            push!(pitch_c, o.coordinate.pitch)
            push!(r_c, o.coordinate.r)
            push!(z_c, o.coordinate.z)
        end
    end
    npart = length(energy)

    h5open(filename,"w") do file
        if length(energy_c) == norbs
            file["energy_c","shuffle",(),"chunk",(norbs),"compress",4] = energy_c
            file["pitch_c","shuffle",(),"chunk",(norbs),"compress",4] = pitch_c
            file["r_c","shuffle",(),"chunk",(norbs),"compress",4] = r_c
            file["z_c","shuffle",(),"chunk",(norbs),"compress",4] = z_c
        end
        file["tau_p","shuffle",(),"chunk", (norbs),"compress", 4] = tau_p
        file["tau_t","shuffle",(),"chunk", (norbs),"compress", 4] = tau_t
        file["data_source"] = "Generated from GEQDSK"
        file["type"] = 2
        file["nclass"] = norbs
        file["nparticle"] = npart
        file["time"] = time
        file["energy", "shuffle",(), "chunk", (npart),"compress", 4] = energy
        file["pitch","shuffle",(), "chunk", (npart),"compress", 4] = pitch
        file["r","shuffle",(), "chunk", (npart),"compress", 4] = r
        file["z","shuffle",(), "chunk", (npart),"compress", 4] = z
        file["class","shuffle",(), "chunk", (npart),"compress", 4] = class
        file["weight","shuffle",(), "chunk", (npart),"compress", 4] = weight
    end
    nothing
end
