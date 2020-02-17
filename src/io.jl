"""

Struct to represent .h5 spectra produced by FIDASIM.
"""
struct FIDASIMSpectra{T<:Real}
    nchan::Int
    nlambda::Int
    lambda::Vector{T}
    brems::Union{Zeros{T,2},Matrix{T}}
    full::Union{Zeros{T,2},Matrix{T}}
    half::Union{Zeros{T,2},Matrix{T}}
    third::Union{Zeros{T,2},Matrix{T}}
    dcx::Union{Zeros{T,2},Matrix{T}}
    halo::Union{Zeros{T,2},Matrix{T}}
    cold::Union{Zeros{T,2},Matrix{T}}
    fida::Union{Zeros{T,2},Zeros{T,3},Matrix{T},Array{T,3}}
    pfida::Union{Zeros{T,2},Zeros{T,3},Matrix{T},Array{T,3}}
end

"""
    FIDASIMSpectra("path/to/FIDASIM/spectra/file.h5")

Create instance of FIDASIMSpectra struct from .h5 file. The data can then be
accessed via Julia struct syntax.

# Example
```julia-repl
julia> myFIDASIMSpectra = FIDASIMSpectra("path/to/FIDASIM/spectra/file.h5")
FIDASIMSpectra
    nchan = 81
    nlambda = 2000

julia> lambda_array = myFIDASIMSpectra.lambda
2000-element Array{Float64,1}:
647.005
647.015
647.025
647.035
⋮

julia> myfida_data = myFIDASIMSpectra.fida
2000×81 Array{Float64,2}:
0.0 0.0 ... 0.0 0.0
0.0 0.0 ... 0.0 0.0
⋮   ⋮    ⋮    ⋮   ⋮
0.0 0.0 ... 0.0 0.0
0.0 0.0 ... 0.0 0.0
```
"""
function FIDASIMSpectra(fname::String)
    isfile(fname) || error("File does not exist")

    f = h5open(fname)
    nchan = Int(read(f["nchan"]))
    lambda = read(f["lambda"])
    nlam = length(lambda)
    vars = names(f)

    brems = "brems" in vars ? read(f["brems"]) : Zeros(nlam,nchan)
    full = "full" in vars ? read(f["full"]) : Zeros(nlam,nchan)
    half = "half" in vars ? read(f["half"]) : Zeros(nlam,nchan)
    third = "third" in vars ? read(f["third"]) : Zeros(nlam,nchan)
    dcx = "dcx" in vars ? read(f["dcx"]) : Zeros(nlam,nchan)
    halo = "halo" in vars ? read(f["halo"]) : Zeros(nlam,nchan)
    cold = "cold" in vars ? read(f["cold"]) : Zeros(nlam,nchan)
    if ("pfida" in vars) && !("fida" in vars)
        pfida = read(f["pfida"])
        fida = Zeros(size(pfida))
    end
    if !("pfida" in vars) && ("fida" in vars)
        fida = read(f["fida"])
        pfida = Zeros(size(fida))
    end
    if ("pfida" in vars) && ("fida" in vars)
        fida = read(f["fida"])
        pfida = read(f["pfida"])
    end
    if !("pfida" in vars) && !("fida" in vars)
        fida = Zeros(nlam,nchan)
        pfida = Zeros(size(fida))
    end

    return FIDASIMSpectra(Int(nchan),Int(nlam),lambda,brems,full,half,third,dcx,halo,cold,fida,pfida)
end

function Base.show(io::IO, s::FIDASIMSpectra)
    println(io, "FIDASIMSpectra")
    println(io, "  nchan   = $(s.nchan)")
    println(io, "  nlambda = $(s.nlambda)")
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
    nfast::Float64
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

    fr = dist .* reshape(r,(1,1,length(r),1))
    dE = abs(energy[2] - energy[1])
    dp = abs(pitch[2] - pitch[1])
    dr = abs(r[2] - r[1])
    dz = abs(z[2] - z[1])
    nfast = sum(fr)*(2*pi*dE*dp*dr*dz)

    return FIDASIMGuidingCenterFunction(nr,nz,nenergy,npitch,r,z,energy,pitch,denf,dist,nfast)
end

function Base.show(io::IO, s::FIDASIMGuidingCenterFunction)
    println(io, "FIDASIMGuidingCenterFunction")
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
    nfast::Float64
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

    return FIDASIMGuidingCenterParticles(npart,nclass,class,weight,r,z,energy,pitch,sum(weight))
end

function Base.show(io::IO, s::FIDASIMGuidingCenterParticles)
    println(io, "FIDASIMGuidingCenterParticles")
    println(io, "  nfast = $(s.nfast)")
    println(io, "  nparticles = $(s.npart)")
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
    nfast::Float64
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

    return FIDASIMFullOrbitParticles(npart,nclass,class,weight,r,z,vr,vt,vz,sum(weight))
end

function Base.show(io::IO, s::FIDASIMFullOrbitParticles)
    println(io, "FIDASIMFullOrbitParticles")
    println(io, "  nfast = $(s.nfast)")
    println(io, "  nparticles = $(s.npart)")
end

"""
    read_fidasim_distribution("path/to/FIDASIM/distribution/.h5/file")

Read FIDASIM distribution .h5 file. Return FIDASIMGuidingCenterFunction, FIDASIMGuidingCenterParticles
or FIDASIMFullOrbitParticles depending on dtype variable. The data can then be accessed via Julia struct
syntax.

# Example
```julia-repl
julia> myFIDASIM_distribution = read_fidasim_distribution("path/to/FIDASIM/distribution/.h5/file")
FIDASIMGuidingCenterFunction

julia> r_array = myFIDASIM_distribution.r
70-element Array{Float64,1}:
 100.0
 102.0
 104.0
 106.0
 108.0
 110.0
 112.0
 114.0
 ⋮

julia> 
```
"""
function read_fidasim_distribution(filename::String)
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

"""
    write_fidasim_distribution(M,orbit_array)
    write_fidasim_distribution(M,orbit_array,filename="myOrbit_distribution.h35",chunksize=1000)

Write array of orbits (as found in Julia package GuidingCenterOrbits.jl/src/orbit.jl/Orbit) calculated in Julia to an 
.h5 distribution file which can be used as input to FIDASIM. If the array of orbits is longer than 1000, it is highly 
recommended to set the chunksize to approx 1000. This will then produce several .h5 distribution files. This is because 
each distribution file should have a reasonable size to be read by FIDASIM. So if write_fidasim_distribution() creates 
15 .h5 files for example, you will need to do 15 FIDASIM runs and then combine the 15 output spectras into one. 
This can be done in numerous ways, taking care due to the likely large file amount of memory required.
"""
function write_fidasim_distribution(M::AxisymmetricEquilibrium, orbits::Array; filename="orbits.h5",time=0.0,ntot=1e19,chunksize=0)

    if chunksize == 0
        orbs = (orbits,)
        nchunks = 1
    else
        orbs = partition(orbits,chunksize)
        nchunks = length(orbs)
    end

    #for io = 1:length(orbs)
    @sync @distributed for io = 1:length(orbs)
        oo = nth(orbs,io)
        norbs = length(oo)
        npart = sum(length(o.path.r) for o in oo)
        if chunksize != 0
            fname= splitext(filename)[1]*@sprintf("_%04d_%04d.h5",io,nchunks)
        else
            fname = filename
        end
        h5open(fname,"w") do file
            file["data_source"] = "Generated from GEQDSK"
            file["nclass"] = norbs
            file["nparticle"] = npart
            file["time"] = time
            file["type"] = 2
            file["tau_p","shuffle",(),"chunk", (norbs),"compress", 4] = [o.tau_p for o in oo]
            file["tau_t","shuffle",(),"chunk", (norbs),"compress", 4] = [o.tau_t for o in oo]

            if isa(oo[1].coordinate, EPRCoordinate)
                file["energy_c","shuffle",(),"chunk",(norbs),"compress",4] = [o.coordinate.energy for o in oo]
                file["pitch_c","shuffle",(),"chunk",(norbs),"compress",4] = [o.coordinate.pitch for o in oo]
                file["r_c","shuffle",(),"chunk",(norbs),"compress",4] = [o.coordinate.r for o in oo]
                file["z_c","shuffle",(),"chunk",(norbs),"compress",4] = [o.coordinate.z for o in oo]
            end

            file["energy","shuffle",(),"chunk",(npart),"compress",4] = vcat((o.path.energy for o in oo)...)
            file["pitch","shuffle",(),"chunk",(npart),"compress",4] = vcat((M.sigma*o.path.pitch for o in oo)...)
            file["r","shuffle",(),"chunk",(npart),"compress",4] = vcat((100*o.path.r for o in oo)...)
            file["z","shuffle",(),"chunk",(npart),"compress",4] = vcat((100*o.path.z for o in oo)...)
            file["class","shuffle",(),"chunk",(npart),"compress",4] = vcat((fill(i,length(o.path)) for (i,o) in enumerate(oo))...)
            file["weight","shuffle",(),"chunk",(npart),"compress",4] = vcat((o.path.dt.*(ntot/sum(o.path.dt)) for o in oo)...)

            ## create datasets
            #energy = d_create(file, "energy", datatype(Float64), (npart,), "shuffle", (), "chunk", (npart,), "compress", 4)
            #pitch = d_create(file, "pitch", datatype(Float64), (npart,), "shuffle", (), "chunk", (npart,), "compress", 4)
            #r = d_create(file, "r", datatype(Float64), (npart,), "shuffle", (), "chunk", (npart,), "compress", 4)
            #z = d_create(file, "z", datatype(Float64), (npart,), "shuffle", (), "chunk", (npart,), "compress", 4)
            #class = d_create(file, "class", datatype(Int32), (npart,), "shuffle", (), "chunk", (npart,), "compress", 4)
            #weight = d_create(file, "weight", datatype(Float64), (npart,), "shuffle", (), "chunk", (npart,), "compress", 4)

            ## incrementally write to dataset
            #istart = 1
            #for (i,o) in enumerate(oo)
            #    npath = length(o.path.energy)
            #    iend = istart + npath - 1
            #    energy[istart:iend] = o.path.energy
            #    pitch[istart:iend] = M.sigma*o.path.pitch
            #    r[istart:iend] = o.path.r*100
            #    z[istart:iend] = o.path.z*100
            #    class[istart:iend] = fill(i,npath)
            #    weight[istart:iend] = o.path.dt.*(ntot/sum(o.path.dt))
            #    istart = istart + npath
            #end
        end
    end
    nothing
end

struct FIDASIMBeamGeometry
    name::String
    src::Vector{Float64}
    axis::Vector{Float64}
    shape::Int
    widy::Float64
    widz::Float64
    divy::Vector{Float64}
    divz::Vector{Float64}
    focy::Float64
    focz::Float64
    naperture::Int
    ashape::Vector{Int}
    awidy::Vector{Float64}
    awidz::Vector{Float64}
    aoffy::Vector{Float64}
    aoffz::Vector{Float64}
    adist::Vector{Float64}
end

function FIDASIMBeamGeometry(filename::String)
    isfile(filename) || error("File does not exist")

    f = h5open(filename)
    if !("nbi" in names(f))
        close(f)
        error("Beam geometry not in file")
    end

    name = read(f["nbi"]["name"])
    src = read(f["nbi"]["src"])
    axis = read(f["nbi"]["axis"])
    shape = Int(read(f["nbi"]["shape"]))
    widy = read(f["nbi"]["widy"])
    widz = read(f["nbi"]["widz"])
    divy = read(f["nbi"]["divy"])
    divz = read(f["nbi"]["divz"])
    focy = read(f["nbi"]["focy"])
    focz = read(f["nbi"]["focz"])
    naperture = Int(read(f["nbi"]["naperture"]))
    if naperture > 0
        ashape = Int.(read(f["nbi"]["ashape"]))
        awidy = read(f["nbi"]["awidy"])
        awidz = read(f["nbi"]["awidz"])
        aoffy = read(f["nbi"]["aoffy"])
        aoffz = read(f["nbi"]["aoffz"])
        adist = read(f["nbi"]["adist"])
    else
        ashape = Int[]
        awidy = Float64[]
        awidz = Float64[]
        aoffy = Float64[]
        aoffz = Float64[]
        adist = Float64[]
    end
    close(f)

    return FIDASIMBeamGeometry(name,src,axis,shape,widy,widz,divy,divz,focy,focz,
                               naperture, ashape, awidy, awidz, aoffy, aoffz, adist)
end

function Base.show(io::IO, s::FIDASIMBeamGeometry)
    println(io, "FIDASIMBeamGeometry")
    println(io, "  name: $(s.name)")
end

struct FIDASIMSpectraGeometry
    nchan::Int
    system::String
    id::Vector{String}
    radius::Vector{Float64}
    lens::Matrix{Float64}
    axis::Matrix{Float64}
    spot_size::Vector{Float64}
    sigma_pi::Vector{Float64}
end

function FIDASIMSpectraGeometry(filename::String)
    isfile(filename) || error("File does not exist")

    f = h5open(filename)
    if !("spec" in names(f))
        close(f)
        error("Spectra geometry not in file")
    end

    nchan = Int(read(f["spec"]["nchan"]))
    system = read(f["spec"]["system"])
    id = read(f["spec"]["id"])
    radius = read(f["spec"]["radius"])
    lens = read(f["spec"]["lens"])
    axis = read(f["spec"]["axis"])
    spot_size = read(f["spec"]["spot_size"])
    sigma_pi = read(f["spec"]["sigma_pi"])

    close(f)

    return FIDASIMSpectraGeometry(nchan,system,id,radius,lens,axis,spot_size,sigma_pi)
end

function Base.show(io::IO, s::FIDASIMSpectraGeometry)
    println(io, "FIDASIMSpectraGeometry")
    println(io, "  nchan: $(s.nchan)")
    println(io, "  system: "*s.system)
end

struct FIDASIMNPAGeometry
    nchan::Int
    system::String
    id::Vector{String}
    radius::Vector{Float64}
    a_shape::Vector{Int}
    d_shape::Vector{Int}
    a_cent::Matrix{Float64}
    a_redge::Matrix{Float64}
    a_tedge::Matrix{Float64}
    d_cent::Matrix{Float64}
    d_redge::Matrix{Float64}
    d_tedge::Matrix{Float64}
end

function FIDASIMNPAGeometry(filename::String)
    isfile(filename) || error("File does not exist")

    f = h5open(filename)
    if !("npa" in names(f))
        close(f)
        error("NPA geometry not in file")
    end

    nchan = Int(read(f["npa"]["nchan"]))
    system = read(f["npa"]["system"])
    id = read(f["npa"]["id"])
    radius = read(f["npa"]["radius"])
    a_shape = Int.(read(f["npa"]["a_shape"]))
    d_shape = Int.(read(f["npa"]["d_shape"]))
    a_cent = read(f["npa"]["a_cent"])
    a_redge = read(f["npa"]["a_redge"])
    a_tedge = read(f["npa"]["a_tedge"])
    d_cent = read(f["npa"]["d_cent"])
    d_redge = read(f["npa"]["d_redge"])
    d_tedge = read(f["npa"]["d_tedge"])

    close(f)

    return FIDASIMNPAGeometry(nchan, system, id, radius,
                              a_shape, d_shape,
                              a_cent, a_redge, a_tedge,
                              d_cent, d_redge, d_tedge)
end

function Base.show(io::IO, s::FIDASIMNPAGeometry)
    println(io, "FIDASIMNPAGeometry")
    println(io, "  nchan: $(s.nchan)")
    print(io, "  system: "*s.system)
end

function write_fidasim_geometry(nbi::FIDASIMBeamGeometry;
                                filename="geometry.h5",
                                data_source="Generated by OrbitTomography.jl",
                                spec::Union{FIDASIMSpectraGeometry,Nothing}=nothing,
                                npa::Union{FIDASIMNPAGeometry,Nothing}=nothing)

    h5open(filename,"w") do file
        file["/nbi/data_source"] = data_source
        file["/nbi/name"] = nbi.name
        file["/nbi/src"] = nbi.src
        file["/nbi/axis"] = nbi.axis
        file["/nbi/focy"] = nbi.focy
        file["/nbi/focz"] = nbi.focz
        file["/nbi/divy"] = nbi.divy
        file["/nbi/divz"] = nbi.divz
        file["/nbi/widy"] = nbi.widy
        file["/nbi/widz"] = nbi.widz
        file["/nbi/shape"] = nbi.shape
        file["/nbi/naperture"] = nbi.naperture
        if nbi.naperture > 0
            file["/nbi/ashape"] = nbi.ashape
            file["nbi/awidy"] = nbi.awidy
            file["nbi/awidz"] = nbi.awidz
            file["nbi/aoffy"] = nbi.aoffy
            file["nbi/aoffz"] = nbi.aoffz
            file["nbi/adist"] = nbi.adist
        end

        if spec != nothing
            file["spec/data_source"] = data_source
            file["spec/nchan"] = spec.nchan
            file["spec/system"] = spec.system
            file["spec/id"] = spec.id
            file["spec/radius"] = spec.radius
            file["spec/lens"] = spec.lens
            file["spec/axis"] = spec.axis
            file["spec/spot_size"] = spec.spot_size
            file["spec/sigma_pi"] = spec.sigma_pi
        end

        if npa != nothing
            file["npa/data_source"] = data_source
            file["npa/nchan"] = npa.nchan
            file["npa/system"] = npa.system
            file["npa/id"] = npa.id
            file["npa/radius"] = npa.radius
            file["npa/a_shape"] = npa.a_shape
            file["npa/d_shape"] = npa.d_shape
            file["npa/a_cent"] = npa.a_cent
            file["npa/a_redge"] = npa.a_redge
            file["npa/a_tedge"] = npa.a_tedge
            file["npa/d_cent"] = npa.d_cent
            file["npa/d_redge"] = npa.d_redge
            file["npa/d_tedge"] = npa.d_tedge
        end
    end
end

struct FIDASIMPlasmaParameters
    nr::Int
    nz::Int
    nphi::Int
    r::Vector{Float64}
    z::Vector{Float64}
    phi::Vector{Float64}
    mask::Union{Matrix{Int16},Array{Int16,3}}
    te::Union{Matrix{Float64},Array{Float64,3}}
    ti::Union{Matrix{Float64},Array{Float64,3}}
    dene::Union{Matrix{Float64},Array{Float64,3}}
    zeff::Union{Matrix{Float64},Array{Float64,3}}
    vr::Union{Matrix{Float64},Array{Float64,3}}
    vt::Union{Matrix{Float64},Array{Float64,3}}
    vz::Union{Matrix{Float64},Array{Float64,3}}
    denn::Union{Matrix{Float64},Array{Float64,3}}
end

function FIDASIMPlasmaParameters(filename::String)
    isfile(filename) || error("File does not exist")

    f = h5open(filename)

    fp = f["plasma"]
    nr = read(fp["nr"])
    nz = read(fp["nz"])
    if "nphi" in names(fp)
        nphi = read(fp["nphi"])
    else
        nphi = 1
    end
    r = read(fp["r"])
    z = read(fp["z"])
    if nphi > 1
        phi = read(fp["phi"])
    else
        phi = [0.0]
    end

    mask = read(fp["mask"])
    te = read(fp["te"])
    ti = read(fp["ti"])
    dene = read(fp["dene"])
    zeff = read(fp["zeff"])

    vr = read(fp["vr"])
    vt = read(fp["vt"])
    vz = read(fp["vz"])

    if "denn" in names(fp)
        denn = read(fp["denn"])
    else
        denn = zero(dene)
    end

    close(f)

    return FIDASIMPlasmaParameters(nr,nz,nphi,r,z,phi,mask,te,ti,dene,zeff,vr,vt,vz,denn)
end

function Base.show(io::IO, s::FIDASIMPlasmaParameters)
    print(io, "FIDASIMPlasmaParameters: $(s.nr)×$(s.nz)×$(s.nphi)")
end
