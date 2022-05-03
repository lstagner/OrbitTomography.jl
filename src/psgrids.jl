struct PSGrid{T}
    energy::AbstractVector{T}
    pitch::AbstractVector{T}
    r::AbstractVector{T}
    z::AbstractVector{T}
    counts::Vector{Int}
    point_index::Array{Int,4}
    class::Array{Symbol,4}
    tau_p::Array{T,4}
    tau_t::Array{T,4}
end

function Base.show(io::IO, psgrid::PSGrid)
    print(io, "PSGrid: $(length(psgrid.energy))×$(length(psgrid.pitch))×$(length(psgrid.r))×$(length(psgrid.z)):$(length(psgrid.counts))")
end

"""
    getGCEPRCoord(M::AbstractEquilibrium,wall::Union{Nothing,Wall},gcp::GCParticle; gcvalid_check::Bool=false, vacuum::Bool=false,drift::Bool=true, kwargs...)

This function takes a gcp particle, calculates the orbit using integrate, and the jacobian determinant of the transform using ForwardDiff, and returns a GCEPRCoordinate. Used in fill_PSGrid.
"""
function getGCEPRCoord(M::AbstractEquilibrium, wall::Union{Nothing,Wall},gcp::GCParticle; gcvalid_check::Bool=false, vacuum::Bool=false,drift::Bool=true, kwargs...)
    ed = ForwardDiff.Dual(gcp.energy,  (0.0,0.0,0.0))
    pd = ForwardDiff.Dual(gcp.pitch,  (1.0,0.0,0.0))
    rd = ForwardDiff.Dual(gcp.r,  (0.0,1.0,0.0))
    zd = ForwardDiff.Dual(gcp.z,  (0.0,0.0,1.0))

    gcp0 = GCParticle(ed,pd,rd,zd,gcp.m,gcp.q)

    if gcvalid_check 
        path, stat = integrate(M, gcp0,wall=wall,one_transit=true, r_callback=true, classify_orbit=true, store_path=true, drift=drift, vacuum=vacuum, toa=true, limit_phi=true, kwargs...)
        CleanPath = OrbitPath(vacuum,drift,ForwardDiff.value.(path.energy),ForwardDiff.value.(path.pitch),ForwardDiff.value.(path.r),ForwardDiff.value.(path.z),ForwardDiff.value.(path.phi),ForwardDiff.value.(path.dt))               
        gcvalid = gcde_check(M, gcp, CleanPath) 
    else
        path, stat = integrate(M, gcp0,wall=wall,one_transit=true, r_callback=true, classify_orbit=true, store_path=false, drift=drift, vacuum=vacuum, toa=true, limit_phi=true, kwargs...)
        gcvalid=nothing
    end

    if stat.class == :incomplete || stat.class == :lost
        return GCEPRCoordinate(gcp.energy,
                    gcp.pitch,
                    gcp.r,
                    gcp.z,
                    typeof(0.0),
                    typeof(0.0),
                    typeof(0.0),
                    typeof(0.0),
                    stat.class,
                    typeof(0.0),
                    typeof(0.0),
                    typeof(0.0),
                    gcvalid,
                    gcp.m,
                    gcp.q)
    end

    hc = HamiltonianCoordinate(M,gcp)
    KE = get_kinetic_energy(M,hc,stat.rm,stat.zm)
    tm = (stat.tau_p - stat.tm)/stat.tau_p

    #transform = x -> x 
    #jac = transform([GCstat.pm,GCstat.rm,tm]) 
    jacdet = max(abs(det(hcat((ForwardDiff.partials(x) for x in [stat.pm,stat.rm,tm])...))),0.0)

    return GCEPRCoordinate(ForwardDiff.value(KE),
                    gcp.pitch,
                    gcp.r,
                    gcp.z,
                    ForwardDiff.value(stat.pm),
                    ForwardDiff.value(stat.rm),
                    ForwardDiff.value(stat.zm),
                    ForwardDiff.value(tm),
                    stat.class,
                    ForwardDiff.value(stat.tau_p),
                    ForwardDiff.value(stat.tau_t),
                    jacdet,
                    gcvalid,
                    gcp.m,
                    gcp.q)
end

"""
    fill_PSGrid(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, energy::AbstractVector, pitch::AbstractVector, r::AbstractVector, z::AbstractVector;  q=1, amu=OrbitTomography.H2_amu, verbose = false, print_results = false, filename_prefactor = "", distributed=true, kwargs...)

This function uses a distributed for loop to apply getGCEPRCoord to every point in a 4D grid specified by the vectors energy, pitch, r and z.
"""
function fill_PSGrid(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, energy::AbstractVector, pitch::AbstractVector, r::AbstractVector, z::AbstractVector;  q=1, amu=OrbitTomography.H2_amu, verbose = false, print_results = false, filename_prefactor = "", distributed=true, kwargs...)
    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)
    subs = CartesianIndices((nenergy,npitch,nr,nz))

        class = fill(:incomplete,(nenergy,npitch,nr,nz))
        point_index = zeros(Int,nenergy,npitch,nr,nz)
        tau_t = zeros(Float64,nenergy,npitch,nr,nz)
        tau_p = zeros(Float64,nenergy,npitch,nr,nz)

    npoints = nenergy*npitch*nr*nz

    if distributed
        psorbs = @showprogress @distributed (vcat) for i=1:npoints
            ie,ip,ir,iz = Tuple(subs[i])
            c = GCParticle(energy[ie],pitch[ip],r[ir],z[iz],amu*OrbitTomography.mass_u,q)

            if !in_vessel(wall,r[ir],z[iz])
                o = GCEPRCoordinate(c,:incomplete) 
            else
                try
                    o = getGCEPRCoord(M,wall,c, kwargs...)
                catch
                    o = GCEPRCoordinate(c,:incomplete) 
                end
            end

            if o.class in (:incomplete,:invalid,:lost)
                o = GCEPRCoordinate(c,:incomplete) 
            end

            o
        end
    else
        p = Progress(npoints)

        psorbs = GCEPRCoordinate[]
        for i=1:npoints
            ie,ip,ir,iz = Tuple(subs[i])
            c = GCParticle(energy[ie],pitch[ip],r[ir],z[iz],amu*OrbitTomography.mass_u,q)

            if !in_vessel(wall,r[ir],z[iz])
                o = GCEPRCoordinate(c,:incomplete) 
            else
                try
                    o = getGCEPRCoord(M,wall,c, kwargs...)
                catch
                    o = GCEPRCoordinate(c,:incomplete) 
                end
            end

            if o.class in (:incomplete,:invalid,:lost)
                o = GCEPRCoordinate(c,:incomplete) 
            end

            ProgressMeter.next!(p)
            push!(psorbs,o)
        end
    end

    for i=1:npoints
        class[subs[i]] = psorbs[i].class
        if class[subs[i]]!= :incomplete 
            tau_p[subs[i]] = psorbs[i].tau_p
            tau_t[subs[i]] = psorbs[i].tau_t
        end 
    end

    grid_index = filter(i -> psorbs[i].class != :incomplete, 1:npoints)
    psorbs = filter(x -> x.class != :incomplete, psorbs)
    npoints = length(psorbs)
    point_index[grid_index] = 1:npoints

    psgrid = PSGrid(energy,pitch,r,z,fill(1,npoints),point_index,class,tau_p,tau_t)

    if print_results
        write_PSGrid(psgrid,filename = string(filename_prefactor, "PSGrid.h5"))
        write_GCEPRCoords(psorbs,filename = string(filename_prefactor, "GCEPRCoords.h5")) 
    end
    if verbose
        print("PSGrid, PSOrbs Calculated\n")
    end

    return psorbs, psgrid
end

function write_PSGrid(grid::PSGrid;filename="PSGrid.h5")
    h5open(filename,"w") do file
        file["energy"] = collect(grid.energy)
        file["pitch"] = collect(grid.pitch)
        file["r"] = collect(grid.r)
        file["z"] = collect(grid.z)
        file["counts"] = grid.counts
        file["point_index"] = grid.point_index
        file["class"] = String.(grid.class)
        file["tau_p"] = grid.tau_p
        file["tau_t"] = grid.tau_t
    end
    nothing
end

function read_PSGrid(filename)
    isfile(filename) || error("File does not exist")

    f = h5open(filename)
    energy = read(f["energy"])
    pitch = read(f["pitch"])
    r = read(f["r"])
    z = read(f["z"])
    counts = read(f["counts"])
    point_index = read(f["point_index"])
    class = Symbol.(read(f["class"]))
    tau_p = read(f["tau_p"])
    tau_t = read(f["tau_t"])
    close(f)

    return PSGrid(energy,pitch,r,z,counts,point_index,class,tau_p,tau_t)
end

function write_GCEPRCoords(psorbs;filename = "GCEPRCoords.h5")
    numcoords = length(psorbs)

    classes = Vector{String}()
    gcvalids = Vector{Union{Bool,Nothing}}()

    E = zeros(Float64,numcoords)
    p = zeros(Float64,numcoords)
    R = zeros(Float64,numcoords)
    Z = zeros(Float64,numcoords)
    pm = zeros(Float64,numcoords)
    Rm = zeros(Float64,numcoords)
    Zm = zeros(Float64,numcoords)
    tm = zeros(Float64,numcoords)
    jacdets = zeros(Float64,numcoords)

    t_p = zeros(Float64,numcoords)
    t_t = zeros(Float64,numcoords)

    for (io,o) in enumerate(orbs) 
        E[io] = o.energy
        p[io] = o.pitch
        R[io] = o.r
        Z[io] = o.z
        pm[io] = o.pitch_m
        Rm[io] = o.r_m
        Zm[io] = o.z_m
        tm[io] = o.t
        jacdets[io] = o.jacdet

        t_p[io] = o.tau_p
        t_t[io] = o.tau_t

        push!(classes,string(o.class))
        push!(gcvalids,o.gcvalid)
    end
    
    h5open(filename,"w") do file
        file["E"] = E
        file["p"] = p
        file["R"] = R
        file["Z"] = Z
        file["pm"] = pm
        file["Rm"] = Rm
        file["Zm"] = Zm
        file["tm_normalised"] = tm
        file["jacdets"] = jacdets

        file["tau_p"] = t_p
        file["tau_t"] = t_t

        file["class"] = classes
        file["gcvalids"] = gcvalids

        file["m"] = orbs[1].coordinate.m
        file["q"] = orbs[1].coordinate.q

        file["vacuum"] = orbs[1].path.vacuum
        file["drift"] = orbs[1].path.drift    
    end
    nothing
end

function read_GCEPRCoords(filename)
    isfile(filename) || error("File does not exist")

    f = h5open(filename)
        E = read(f["E"])
        p = read(f["p"])
        R = read(f["R"])
        Z = read(f["Z"])
        pm = read(f["pm"])
        Rm = read(f["Rm"])
        Zm = read(f["Zm"])
        tm = read(f["tm_normalised"])
        jacdets = read(f["jacdets"])

        t_p = read(f["tau_p"])
        t_t = read(f["tau_t"])

        classes = Symbol.(read(f["class"]))
        gcvalids = read(f["gcvalids"])

        m = read(f["m"])
        q = read(f["q"])

        vacuum = read(f["vacuum"])
        drift = read(f["drift"])
    close(f)

    coords = GCEPRCoordinate[]

    print("Appending Orbits\n")
    prog = Progress(length(E))

    for i=1:length(E)
        c = GCEPRCoordinate(E[i],p[i],R[i],Z[i],pm[i],Rm[i],Zm[i],tm[i],classes[i],t_p[i],t_t[i],jacdets[i],gcvalids[i],m,q)

        ProgressMeter.next!(prog)
        push!(coords,c)
    end

    return coords,vacuum,drift
end 


"""
    fill_PSGrid_batch(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, energy::AbstractVector, pitch::AbstractVector, r::AbstractVector, z::AbstractVector; batch_multipler=20, q=1, amu=OrbitTomography.H2_amu, verbose = false, filename_prefactor = "", kwargs...)

This function uses pmap to apply getGCEPRCoord in distributed batches to every point in a 4D grid specified by the vectors energy, pitch, r and z.
Each distributed batch of GCEPRCoords is printed straight to file by its remote worker. The function reconstruct_PSGrid must then be ran in a single high-RAM node to read and collate these GCEPRCoord batches and make a single PSGrid.

Just before you call this function you must input the following: 
    "path = pwd()
    filename_prefactor = <whatever you are using>
 
    for i in workers()
        remotecall_fetch(()->path, i)
        remotecall_fetch(()->filename_prefactor, i)
    end
    @everywhere cd(path)"

Then state filename_prefactor=filename_prefactor in the function input
"""
function fill_PSGrid_batch(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, energy::AbstractVector, pitch::AbstractVector, r::AbstractVector, z::AbstractVector; batch_multipler=20, q=1, amu=OrbitTomography.H2_amu, verbose = false, filename_prefactor = "", kwargs...)
    path=pwd()

    mkdir(string(filename_prefactor,"_Batches"))
    @everywhere cd(string(path,"/",filename_prefactor,"_Batches"))

    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)

    npoints = nenergy*npitch*nr*nz

    (nworkers()*batch_multipler-1) > npoints ? (loop_batches = (npoints-1)) : (loop_batches = (nworkers()*batch_multipler)-1)

    batch = ceil(Int,npoints/(loop_batches+1))
    if batch*loop_batches >= npoints  
        print("Unoptimised batch num.\n")
        batch = floor(Int,npoints/(loop_batches+1))
    end

    print("Num batches = ",(loop_batches+1),"\n")

    writebatch = x -> fillPSGrid_single_pmap(M,wall,energy,pitch,r,z,x,batch,loop_batches, q=q, amu=amu, filename_prefactor = filename_prefactor, kwargs...)

    pmap(writebatch, collect(1:(loop_batches+1)))

    #Writes a file to give info needed to reconstruct the whole PSGrid:
    #Need energy,pitch,r,z, q and amu 
    h5open(string(filename_prefactor,"PSGridInfo.h5"),"w") do file
        file["energy"] = collect(energy)
        file["pitch"] = collect(pitch)
        file["r"] = collect(r)
        file["z"] = collect(z)
        file["batch"] = batch
        file["loop_batches"] = loop_batches
    end

    cd(path)
    return Nothing()
end

"""
    fillPSGrid_single_pmap(M::AbstractEquilibrium,wall::Union{Nothing,Wall},energy::AbstractVector{T},pitch::AbstractVector{T},r::AbstractVector{T},z::AbstractVector{T},batch_num::Int,batch::Int,loop_batches::Int; q::Int=1, amu::Float64=OrbitTomography.H2_amu, filename_prefactor::String="", kwargs...) where T<:AbstractFloat

This function is used internally within fill_PSGrid_batch. It is inputted to pmap within fill_PSGrid_batch. It sets up a local for loop on each remote worker that calculates the orbits of a specific batch specified by the inputs batch_num, batch and loop_batches.
"""
function fillPSGrid_single_pmap(M::AbstractEquilibrium,wall::Union{Nothing,Wall},energy::AbstractVector{T},pitch::AbstractVector{T},r::AbstractVector{T},z::AbstractVector{T},batch_num::Int,batch::Int,loop_batches::Int; q::Int=1, amu::Float64=OrbitTomography.H2_amu, filename_prefactor::String="", kwargs...) where T<:AbstractFloat
    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)
    subs = CartesianIndices((nenergy,npitch,nr,nz))

    npoints = nenergy*npitch*nr*nz

    if batch_num <= loop_batches
        j = batch_num

        GCEPRCs = GCEPRCoordinate[]
        for i=(1+(j-1)*batch):(j*batch)
            ie,ip,ir,iz = Tuple(subs[i])
            c = GCParticle(energy[ie],pitch[ip],r[ir],z[iz],amu*OrbitTomography.mass_u,q)

            if !in_vessel(wall,r[ir],z[iz])
                o = GCEPRCoordinate(c,:incomplete) 
            else
                try
                    o = getGCEPRCoord(M,wall,c, kwargs...)
                catch
                    o = GCEPRCoordinate(c,:incomplete) 
                end
            end

            if o.class in (:incomplete,:invalid,:lost)
                o = GCEPRCoordinate(c,:incomplete) 
            end

            push!(GCEPRCs,o)
        end

        write_GCEPRCoords(GCEPRCs,filename = string(filename_prefactor,"_batch(",batch,")",j,"of",(loop_batches+1), "_GCEPRCoords.h5")) 
    elseif batch_num == (loop_batches+1)
        GCEPRCs = GCEPRCoordinate[]
        for i=(1+loop_batches*batch):npoints
            ie,ip,ir,iz = Tuple(subs[i])
            c = GCParticle(energy[ie],pitch[ip],r[ir],z[iz],amu*OrbitTomography.mass_u,q)

            if !in_vessel(wall,r[ir],z[iz])
                o = GCEPRCoordinate(c,:incomplete) 
            else
                try
                    o = getGCEPRCoord(M,wall,c, kwargs...)
                catch
                    o = GCEPRCoordinate(c,:incomplete) 
                end
            end

            if o.class in (:incomplete,:invalid,:lost)
                o = GCEPRCoordinate(c,:incomplete) 
            end

            push!(GCEPRCs,o)
        end

        write_GCEPRCoords(GCEPRCs,filename = string(filename_prefactor,"_batch(",batch,")",(loop_batches+1),"of",(loop_batches+1),"_GCEPRCoords.h5")) 
    else
        error("Inputted batch number is greater than total number of batches.\n")
    end

    return Nothing()
end

"""
    reconstruct_PSGrid(filename_prefactor; print_results = false, verbose = false, print_dir = "", init_dir = "")

This function reads and collates the direct-to-file printed batches of GCEPRCoords produced by fill_PSGrid_batch. It outputs them as a single vector, as well as calculting and outputting their corresponding PSGrid.
The function should be ran on a single high-RAM node. The input variable filename_prefactor must be the same as that used in fill_PSGrid_batch to create the initial batches.
"""
function reconstruct_PSGrid(filename_prefactor; print_results = false, verbose = false, print_dir = "", init_dir = "")
    length(init_dir)>0 && cd(init_dir)

    isfile(string(filename_prefactor,"PSGridInfo.h5")) || error("PSGridInfo does not exist")
    
    f = h5open(string(filename_prefactor,"PSGridInfo.h5"))
        energy = read(f["energy"])
        pitch = read(f["pitch"])
        r = read(f["r"])
        z = read(f["z"])
        batch = read(f["batch"])
        loop_batches = read(f["loop_batches"])
    close(f)

    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)
    subs = CartesianIndices((nenergy,npitch,nr,nz))

    class = fill(:incomplete,(nenergy,npitch,nr,nz))
    point_index = zeros(Int,nenergy,npitch,nr,nz)
    tau_t = zeros(Float64,nenergy,npitch,nr,nz)
    tau_p = zeros(Float64,nenergy,npitch,nr,nz)

    npoints = nenergy*npitch*nr*nz

    GCEPRCoords = reconstruct_GCEPRCoords(filename_prefactor, batch, loop_batches)

    for i=1:npoints
        class[subs[i]] = GCEPRCoords[i].class
        tau_p[subs[i]] = GCEPRCoords[i].tau_p
        tau_t[subs[i]] = GCEPRCoords[i].tau_t
    end

    grid_index = filter(i -> GCEPRCoords[i].class != :incomplete, 1:npoints)
    GCEPRCoords = filter(x -> x.class != :incomplete, GCEPRCoords)
    npoints = length(GCEPRCoords)
    point_index[grid_index] = 1:npoints

    psgrid = PSGrid(energy,pitch,r,z,fill(1,npoints),point_index,class,tau_p,tau_t)

    if print_results
        length(print_dir)>0 && cd(print_dir)
        write_PSGrid(psgrid,filename = string(filename_prefactor, "PSGrid.h5"))
        write_GCEPRCoords(GCEPRCoords,filename = string(filename_prefactor, "GCEPRCoords.h5")) 
    end

    verbose && print("PSGrid and GCEPRCoords Calculated\n")

    return GCEPRCoords, psgrid
end 

"""
    reconstruct_GCEPRCoords(filename_prefactor::String, batch::Int, loop_batches::Int)

This function is used internally within reconstruct_PSGrid to read batches of GCEPRCoordinates, and collate them into a single vector.
"""
function reconstruct_GCEPRCoords(filename_prefactor::String, batch::Int, loop_batches::Int)
    total_GCEPRCoords = GCEPRCoordinate[]
    batch_error = Int[]
    p = Progress(loop_batches+1)

    for j=1:(loop_batches+1)
        try
            coords = read_GCEPRCoords(string(filename_prefactor,"_batch(",batch,")",j,"of",(loop_batches+1), "_GCEPRCoords.h5")) 
            append!(total_GCEPRCoords,coords)
        catch
            push!(batch_error,j)
            ProgressMeter.next!(p)
            continue
        end
        ProgressMeter.next!(p)
    end

    if length(batch_error)>0
        display(batch_error)
        error("Batch Reading Error, failed batches seen above.\n")
    end 

    return total_GCEPRCoords
end 