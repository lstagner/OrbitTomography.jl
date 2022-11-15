struct PSGrid{T}
    energy::AbstractVector{T}
    pitch::AbstractVector{T}
    r::AbstractVector{T}
    z::AbstractVector{T}
    counts::Vector{Int}
    point_index::Array{Int,4}
    class::Array{Char,4}
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
function getGCEPRCoord(M::AbstractEquilibrium, wall::Union{Nothing,Wall},gcp::GCParticle; gcvalid_check::Bool=false, vacuum::Bool=false,drift::Bool=true, toa::Bool=true, kwargs...)
    ed = ForwardDiff.Dual(gcp.energy,  (0.0,0.0,0.0))
    pd = ForwardDiff.Dual(gcp.pitch,  (1.0,0.0,0.0))
    rd = ForwardDiff.Dual(gcp.r,  (0.0,1.0,0.0))
    zd = ForwardDiff.Dual(gcp.z,  (0.0,0.0,1.0))

    gcp0 = GCParticle(ed,pd,rd,zd,gcp.m,gcp.q)

    if gcvalid_check 
        path, stat = integrate(M, gcp0; wall=wall,one_transit=true, r_callback=true, classify_orbit=true, store_path=true, drift=drift, vacuum=vacuum, toa=toa, kwargs...)
        CleanPath = OrbitPath(vacuum,drift,ForwardDiff.value.(path.energy),ForwardDiff.value.(path.pitch),ForwardDiff.value.(path.r),ForwardDiff.value.(path.z),ForwardDiff.value.(path.phi),ForwardDiff.value.(path.dt))               
        gcvalid = gcde_check(M, gcp, CleanPath) 
    else
        path, stat = integrate(M, gcp0; wall=wall,one_transit=true, r_callback=true, classify_orbit=true, store_path=false, drift=drift, vacuum=vacuum, toa=toa, kwargs...)
        gcvalid=false
    end

    if stat.class == :incomplete || stat.class == :lost
        return GCEPRCoordinate(gcp.energy,
                    gcp.pitch,
                    gcp.r,
                    gcp.z,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    class_char(stat.class),
                    0.0,
                    0.0,
                    0.0,
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
                    class_char(stat.class),
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
function fill_PSGrid(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, energy::AbstractVector, pitch::AbstractVector, r::AbstractVector, z::AbstractVector;  q=1, amu=OrbitTomography.H2_amu, verbose = false, print_results = false, filename_prefactor = "", distributed=true,  gcvalid_check::Bool=false, toa::Bool=true, kwargs...)
    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)
    subs = CartesianIndices((nenergy,npitch,nr,nz))

    class = fill('i',(nenergy,npitch,nr,nz))
    point_index = zeros(Int,nenergy,npitch,nr,nz)
    tau_t = zeros(Float64,nenergy,npitch,nr,nz)
    tau_p = zeros(Float64,nenergy,npitch,nr,nz)

    npoints = nenergy*npitch*nr*nz

    if distributed
        psorbs = @showprogress @distributed (vcat) for i=1:npoints
            ie,ip,ir,iz = Tuple(subs[i])
            c = GCParticle(energy[ie],pitch[ip],r[ir],z[iz],amu*OrbitTomography.mass_u,q)

            if !in_vessel(wall,r[ir],z[iz])
                o = GCEPRCoordinate(c,'i') 
            else
                try
                    o = getGCEPRCoord(M,wall,c; gcvalid_check=gcvalid_check, toa=toa, kwargs...)
                catch
                    o = GCEPRCoordinate(c,'i') 
                end
            end

            if o.class in ('i','v','l')
                o = GCEPRCoordinate(c,'i') 
            end

            o
        end
    else
        psorbs = GCEPRCoordinate[]
        @showprogress for i=1:npoints
            ie,ip,ir,iz = Tuple(subs[i])
            c = GCParticle(energy[ie],pitch[ip],r[ir],z[iz],amu*OrbitTomography.mass_u,q)

            if !in_vessel(wall,r[ir],z[iz])
                o = GCEPRCoordinate(c,'i') 
            else
                try
                    o = getGCEPRCoord(M,wall,c; gcvalid_check=gcvalid_check, toa=toa, kwargs...)
                catch
                    o = GCEPRCoordinate(c,'i') 
                end
            end

            if o.class in ('i','v','l')
                o = GCEPRCoordinate(c,'i') 
            end

            push!(psorbs,o)
        end
    end

    for i=1:npoints
        class[subs[i]] = psorbs[i].class
        if class[subs[i]] != 'i' 
            tau_p[subs[i]] = psorbs[i].tau_p
            tau_t[subs[i]] = psorbs[i].tau_t
        end 
    end

    grid_index = filter(i -> psorbs[i].class != 'i', 1:npoints)
    psorbs = filter(x -> x.class != 'i', psorbs)
    npoints = length(psorbs)
    point_index[grid_index] = 1:npoints

    psgrid = PSGrid(energy,pitch,r,z,fill(1,npoints),point_index,class,tau_p,tau_t)

    if print_results
        write_PSGrid(psgrid,filename = string(filename_prefactor, "PSGrid.jld2"))
        write_GCEPRCoords(psorbs,filename = string(filename_prefactor, "GCEPRCoords.jld2"), gcvalid_check) 
    end
    if verbose
        print("PSGrid, PSOrbs Calculated\n")
    end

    return psorbs, psgrid, gcvalid_check
end

"""
    energy_slice(psgrid::PSGrid, PS_orbs::Vector{GCEPRCoordinate}, gcvalid_check::Bool, ind::Int)

Makes a smaller PSGrid with a single energy, from an existing PSGrid.
"""
function energy_slice(psgrid::PSGrid, PS_orbs::Vector{GCEPRCoordinate}, gcvalid_check::Bool, ind::Int) where {T}
    newenergy = Float64[]
    push!(newenergy,psgrid.energy[ind])

    nenergy = length(newenergy)
    npitch = length(psgrid.pitch)
    nr = length(psgrid.r)
    nz = length(psgrid.z)

    npoints = nenergy*npitch*nr*nz

    class = fill('i',(nenergy,npitch,nr,nz))
    point_index = zeros(Int,nenergy,npitch,nr,nz)
    tau_t = zeros(Float64,nenergy,npitch,nr,nz)
    tau_p = zeros(Float64,nenergy,npitch,nr,nz)

    newPS_orbs = GCEPRCoordinate[]
    eInt_point_index = psgrid.point_index[ind,:,:,:] #3D matrix

    for (io,o) in enumerate(eInt_point_index)
        if o!=0 
            push!(newPS_orbs,PS_orbs[o])
            class[io]=PS_orbs[o].class
            tau_p[io] = PS_orbs[o].tau_p
            tau_t[io] = PS_orbs[o].tau_t
        end
    end

    grid_index = filter(i -> eInt_point_index[i] != 0, 1:npoints)
    npoints = length(newPS_orbs)
    point_index[grid_index] = 1:npoints

    return newPS_orbs, PSGrid(newenergy,psgrid.pitch,psgrid.r,psgrid.z,fill(1,npoints),point_index,class,tau_p,tau_t), gcvalid_check
end

"""
    ps_VectorToMatrix(F_ps_VEC::AbstractVector{Float64},PS_Grid::PSGrid)

Converts a 1D vector where each value corresponds to a valid orbit (using PS_Grid.point_index) into a 4D matrix of values.
Using sharedArray = true uses shared arrays, which break down on some clusters: https://stackoverflow.com/questions/64802561/julia-sharedarray-with-remote-workers-becomes-a-0-element-array
"""
function ps_VectorToMatrix(F_ps_VEC::AbstractVector{Float64},PS_Grid::PSGrid; sharedArray::Bool=false)
    nenergy = length(PS_Grid.energy)
    npitch = length(PS_Grid.pitch)
    nr = length(PS_Grid.r)
    nz = length(PS_Grid.z)

    subs = CartesianIndices((nenergy,npitch,nr,nz))
    npoints = nenergy*npitch*nr*nz

    if !sharedArray
        F_ps_Matrix = zeros(Float64,nenergy,npitch,nr,nz)
        @inbounds for i = 1:npoints
            (PS_Grid.point_index[subs[i]] == 0) ? (F_ps_Matrix[subs[i]] = 0.0) : (F_ps_Matrix[subs[i]]=F_ps_VEC[PS_Grid.point_index[subs[i]]])
        end
    else
        F_ps_Matrix0 = SharedArray{Float64}(nenergy,npitch,nr,nz)
        PS_Grid_point_index = PS_Grid.point_index
        @sync @distributed for i = 1:npoints
            (PS_Grid_point_index[subs[i]] == 0) ? (F_ps_Matrix0[subs[i]] = 0.0) : (F_ps_Matrix0[subs[i]]=F_ps_VEC[PS_Grid_point_index[subs[i]]])
        end

        F_ps_Matrix = convert(Array,F_ps_Matrix0)
        @everywhere F_ps_Matrix0 = nothing
    end

    return F_ps_Matrix
end

"""
    ps_MatrixToVector(F_ps_Matrix::Array{Float64,4},PS_Grid::PSGrid)

Converts a 4D matrix of values in particle-space, and converts it into a 1D vector where each value corresponds to a valid orbit (using PS_Grid.point_index).
Using sharedArray = true uses shared arrays, which break down on some clusters: https://stackoverflow.com/questions/64802561/julia-sharedarray-with-remote-workers-becomes-a-0-element-array
"""
function ps_MatrixToVector(F_ps_Matrix::Array{Float64,4},PS_Grid::PSGrid; sharedArray::Bool=false)
    nenergy = length(PS_Grid.energy)
    npitch = length(PS_Grid.pitch)
    nr = length(PS_Grid.r)
    nz = length(PS_Grid.z)

    subs = CartesianIndices((nenergy,npitch,nr,nz))

    npoints = nenergy*npitch*nr*nz

    if !sharedArray
        F_ps_VEC = Float64[]
        @inbounds for i = 1:npoints
            (PS_Grid.point_index[subs[i]] != 0) && push!(F_ps_VEC,F_ps_Matrix[subs[i]])
        end
    else
        PS_Grid_point_index = PS_Grid.point_index
        F_ps_VEC0 = SharedVector{Float64}(length(PS_Grid.counts))
        @sync @distributed for i = 1:npoints
            (PS_Grid_point_index[subs[i]] != 0) && (F_ps_VEC0[PS_Grid_point_index[subs[i]]] = F_ps_Matrix[subs[i]])
        end

        F_ps_VEC = convert(Array,F_ps_VEC0)
        @everywhere F_ps_VEC0 = nothing
    end
    return F_ps_VEC
end

function write_PSGrid(psgrid::PSGrid;filename="PSGrid.jld2")
    @save filename psgrid
    nothing
end

function read_PSGrid(filename::String; old::Bool=false)
    (old||(last(filename,2)=="h5")) && (return read_PSGridOld(filename))

    isfile(filename) || error("File does not exist")
    @load filename psgrid
    return psgrid
end

function read_PSGridOld(filename::String) 
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

    class = class_char.(class)
    psgrid = PSGrid(energy,pitch,r,z,counts,point_index,class,tau_p,tau_t)

    filename_new = string(chop(filename,tail=2),"jld2")
    write_PSGrid(psgrid,filename=filename_new)

    return psgrid
end

"""
    write_GCEPRCoords(psorbs,...)

Prints a vector of GCEPRCoords. *MUST manually specify vacuum, drift.
"""
function write_GCEPRCoords(psorbs::Vector{GCEPRCoordinate},gcvalid_check::Bool; vacuum=false, drift=true, filename = "GCEPRCoords.jld2") 
    @save filename psorbs gcvalid_check vacuum drift
    nothing
end

"""
    write_GCEPRCoordsMatrix(psorbs,...)

Prints GCEPRCoords as matrices as outputted by matrix_GCEPRCoords. *MUST manually specify vacuum, drift.
"""
function write_GCEPRCoordsMatrix(psorbs::Vector{GCEPRCoordinate},gcvalid_check::Bool; vacuum::Bool=false, drift::Bool=true, filename = "GCEPRCoordsMatrix.jld2", distributed::Bool=false, sharedArray::Bool=false, verbose::Bool = true)
    GCEPR_vals,GCEPR_times,classes,gcvalids = matrix_GCEPRCoords(psorbs, gcvalid_check; distributed=distributed, sharedArray=sharedArray, verbose=verbose)
    m = psorbs[1].m
    q = psorbs[1].q
    @save filename GCEPR_vals GCEPR_times classes gcvalids m q vacuum drift

    nothing
end


"""
    read_GCEPRCoordsMatrix(filename;verbose=true)

Reads GCEPRCoords that were printed to file by write_GCEPRCoordsMatrix.
"""
function read_GCEPRCoordsMatrix(filename;verbose::Bool=true)
    isfile(filename) || error("File does not exist")

    @load filename GCEPR_vals GCEPR_times classes gcvalids m q vacuum drift
    gcvalid_check = !isempty(gcvalids) 

    return GCEPR_vals,GCEPR_times,classes,gcvalid_check,gcvalids,m,q,vacuum,drift
end 

"""
    read_GCEPRCoords(filename;verbose=true)

Reads GCEPRCoords that were printed to file by write_GCEPRCoords. Note the output of this function is coords,vacuum,drift.
"""
function read_GCEPRCoords(filename; old::Bool=false, distributed::Bool=false)
    (old||(last(filename,2)=="h5")) && (return read_GCEPRCoordsOld(filename,distributed=distributed))

    isfile(filename) || error("File does not exist")
    @load filename psorbs gcvalid_check vacuum drift

    return psorbs,gcvalid_check,vacuum,drift
end 

function read_GCEPRCoordsOld(filename; verbose=true, distributed::Bool=false)
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

    gcvalid_check = !isempty(gcvalids)
    classes = class_char.(classes)

    verbose && print("Appending Orbits\n")
    if !distributed
        coords = Vector{GCEPRCoordinate}(undef,length(E))
        if !gcvalid_check
            @showprogress for i=1:length(E)
                coords[i] = GCEPRCoordinate(E[i],p[i],R[i],Z[i],pm[i],Rm[i],Zm[i],tm[i],classes[i],t_p[i],t_t[i],jacdets[i],false,m,q)
            end
        else
            @showprogress for i=1:length(E)
                coords[i] = GCEPRCoordinate(E[i],p[i],R[i],Z[i],pm[i],Rm[i],Zm[i],tm[i],classes[i],t_p[i],t_t[i],jacdets[i],gcvalids[i],m,q)
            end
        end
    else
        if !gcvalid_check
            coords = @distributed (vcat) for i=1:length(E)
                GCEPRCoordinate(E[i],p[i],R[i],Z[i],pm[i],Rm[i],Zm[i],tm[i],classes[i],t_p[i],t_t[i],jacdets[i],false,m,q)
            end
        else
            coords = @distributed (vcat) for i=1:length(E)
                GCEPRCoordinate(E[i],p[i],R[i],Z[i],pm[i],Rm[i],Zm[i],tm[i],classes[i],t_p[i],t_t[i],jacdets[i],gcvalids[i],m,q)
            end
        end
    end

    filename_new = string(chop(filename,tail=2),"jld2")
    write_GCEPRCoords(coords,gcvalid_check; vacuum=vacuum, drift=drift, filename=filename_new)

    return coords,gcvalid_check,vacuum,drift
end 

"""
    matrix_GCEPRCoords(psorbs::Vector{GCEPRCoordinate}, gcvalid_check::Bool;...)

Converts a vector of GCEPRCoords to matrices of their values. distributed=true, sharedArray=false uses DistributedArrays which works on clusters. sharedArray=true uses SharedArrays which may crash on the cluster - see https://stackoverflow.com/questions/64802561/julia-sharedarray-with-remote-workers-becomes-a-0-element-array.
"""
function matrix_GCEPRCoords(psorbs::Vector{GCEPRCoordinate}, gcvalid_check::Bool; distributed::Bool=false, sharedArray::Bool=false, verbose::Bool = true)
    num_psorbs = length(psorbs)
    sharedArray && (distributed=true)

    if !distributed
        GCEPR_vals = Array{Float64}(undef,num_psorbs,8)
        GCEPR_times = Array{Float64}(undef,num_psorbs,3)
        classes = Vector{Char}(undef,num_psorbs)

        if gcvalid_check
            gcvalids = Vector{Bool}(undef,num_psorbs)
        else
            gcvalids = Vector{Bool}()
        end

        m = psorbs[1].m
        q = psorbs[1].q

        @inbounds for i in 1:num_psorbs
            GCEPR_vals[i,:] = [psorbs[i].energy,psorbs[i].pitch,psorbs[i].r,psorbs[i].z,psorbs[i].pitch_m,psorbs[i].r_m,psorbs[i].z_m,psorbs[i].jacdet]
            GCEPR_times[i,:] = [psorbs[i].t,psorbs[i].tau_p,psorbs[i].tau_t]
            classes[i] = psorbs[i].class
            gcvalid_check && (gcvalids[i]=psorbs[i].gcvalid)
        end

        return (GCEPR_vals,GCEPR_times,classes,gcvalids)
    elseif sharedArray
        GCEPR_vals = SharedArray{Float64}(num_psorbs,8)
        GCEPR_times = SharedArray{Float64}(num_psorbs,3)
        classes = SharedVector{Char}(num_psorbs)

        if gcvalid_check
            gcvalids = SharedVector{Bool}(num_psorbs)
        else
            gcvalids = Vector{Bool}()
        end

        m = psorbs[1].m
        q = psorbs[1].q

        @sync @distributed for i in 1:num_psorbs
            GCEPR_vals[i,:] = [psorbs[i].energy,psorbs[i].pitch,psorbs[i].r,psorbs[i].z,psorbs[i].pitch_m,psorbs[i].r_m,psorbs[i].z_m,psorbs[i].jacdet]
            GCEPR_times[i,:] = [psorbs[i].t,psorbs[i].tau_p,psorbs[i].tau_t]
            classes[i] = psorbs[i].class
            gcvalid_check && (gcvalids[i]=psorbs[i].gcvalid)
        end

        gcvalid_check ? (gcvalids0 = convert(Vector,gcvalids)) : gcvalids0 = Vector{Bool}()

        returns = (convert(Array,GCEPR_vals),convert(Array,GCEPR_times),convert(Vector,classes),gcvalids0)

        verbose && print("Garbage collecting shared arrays.")
        @everywhere begin
            GCEPR_vals = nothing
            GCEPR_times = nothing
            classes = nothing
            gcvalids=nothing
            GC.gc()
        end

        return returns
    else
        num_psorbs = length(psorbs)

        GCEPR_vals0 = dzeros(Float64,(num_psorbs,8),workers(),[nworkers(),1]) #SharedArray{Float64}(num_psorbs,8)
        GCEPR_times0 = dzeros(Float64,(num_psorbs,3),workers(),[nworkers(),1]) #SharedArray{Float64}(num_psorbs,3)
        classes0 = dfill('0',num_psorbs)
    
        if gcvalid_check
            gcvalids0 = dfill(false,num_psorbs)
        else
            gcvalids0 = Vector{Bool}()
        end

        @sync @distributed for j = 1:nworkers()
            for (io,i) in enumerate(DistributedArrays.localindices(GCEPR_vals0)[1])
                localpart(GCEPR_vals0)[io,:] = [psorbs[i].energy,psorbs[i].pitch,psorbs[i].r,psorbs[i].z,psorbs[i].pitch_m,psorbs[i].r_m,psorbs[i].z_m,psorbs[i].jacdet]
                localpart(GCEPR_times0)[io,:] = [psorbs[i].t,psorbs[i].tau_p,psorbs[i].tau_t]
                localpart(classes0)[io] = psorbs[i].class
                gcvalid_check && (localpart(gcvalids0)[io]=psorbs[i].gcvalid)
            end
        end

        gcvalid_check ? (gcvalids = convert(Vector,gcvalids0)) : gcvalids= Vector{Bool}()
        returns = (convert(Array,GCEPR_vals0),convert(Array,GCEPR_times0),convert(Vector,classes0),gcvalids)

        verbose && print("Closing distributed arrays.")
        close(GCEPR_vals0)
        close(GCEPR_times0)
        close(classes0)
        gcvalid_check && close(gcvalids0)

        return returns
    end
    nothing
end

"""
    fill_PSGrid_batch(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, energy::AbstractVector, pitch::AbstractVector, r::AbstractVector, z::AbstractVector; batch_multipler=20, q=1, amu=OrbitTomography.H2_amu, verbose = false, filename_prefactor = "", kwargs...)

This function uses pmap to apply getGCEPRCoord in distributed batches to every point in a 4D grid specified by the vectors energy, pitch, r and z.
Each distributed batch of GCEPRCoords is printed straight to file by its remote worker. The function reconstruct_PSGrid must then be ran in a single high-RAM node to read and collate these GCEPRCoord batches and make a single PSGrid.


*Just before you call this function you must input the following: 
    "path = pwd()
    filename_prefactor = <whatever you are using>
 
    for i in workers()
        remotecall_fetch(()->path, i)
        remotecall_fetch(()->filename_prefactor, i)
    end
    @everywhere cd(path)"
Then state filename_prefactor=filename_prefactor in the function input
"""
function fill_PSGrid_batch(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, energy::AbstractVector, pitch::AbstractVector, r::AbstractVector, z::AbstractVector; filename_prefactor = "", gcvalid_check::Bool=false, batch_multipler=20, q=1, amu=OrbitTomography.H2_amu, toa::Bool=true, verbose = false, debug=false, kwargs...)
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

    writebatch = x -> fillPSGrid_single_pmap(M,wall,energy,pitch,r,z,x,batch,loop_batches; q=q, amu=amu, filename_prefactor = filename_prefactor, gcvalid_check=gcvalid_check, toa=toa, kwargs...)

    if !debug
        pmap(writebatch, collect(1:(loop_batches+1)))
    else
        for i in 1:(loop_batches+1)
            fillPSGrid_single_pmap(M,wall,energy,pitch,r,z,i,batch,loop_batches; q=q, amu=amu, filename_prefactor = filename_prefactor, gcvalid_check=gcvalid_check, toa=toa, kwargs...)
        end
    end



    #Writes a file to give info needed to reconstruct the whole PSGrid:
    #Need energy,pitch,r,z, q and amu 
    h5open(string(filename_prefactor,"PSGridInfo.h5"),"w") do file
        file["energy"] = collect(energy)
        file["pitch"] = collect(pitch)
        file["r"] = collect(r)
        file["z"] = collect(z)
        file["batch"] = batch
        file["loop_batches"] = loop_batches
        file["gcvalid_check"] = gcvalid_check
    end

    cd(path)
    return Nothing()
end

"""
    fillPSGrid_single_pmap(M::AbstractEquilibrium,wall::Union{Nothing,Wall},energy::AbstractVector{T},pitch::AbstractVector{T},r::AbstractVector{T},z::AbstractVector{T},batch_num::Int,batch::Int,loop_batches::Int; q::Int=1, amu::Float64=OrbitTomography.H2_amu, filename_prefactor::String="", kwargs...) where T<:AbstractFloat

This function is used internally within fill_PSGrid_batch. It is inputted to pmap within fill_PSGrid_batch. It sets up a local for loop on each remote worker that calculates the orbits of a specific batch specified by the inputs batch_num, batch and loop_batches.
"""
function fillPSGrid_single_pmap(M::AbstractEquilibrium,wall::Union{Nothing,Wall},energy::AbstractVector{T},pitch::AbstractVector{T},r::AbstractVector{T},z::AbstractVector{T},batch_num::Int,batch::Int,loop_batches::Int; q::Int=1, amu::Float64=OrbitTomography.H2_amu, filename_prefactor::String="", gcvalid_check=false, toa::Bool=true, kwargs...) where T<:AbstractFloat
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
                o = GCEPRCoordinate(c,'i')  
            else
                try
                    o = getGCEPRCoord(M,wall,c; gcvalid_check=gcvalid_check, toa=toa, kwargs...)
                catch
                    o = GCEPRCoordinate(c,'i')  
                end
            end

            if o.class in ('i','v','l')
                o = GCEPRCoordinate(c,'i')  
            end

            push!(GCEPRCs,o)
        end

        write_GCEPRCoords(GCEPRCs,filename = string(filename_prefactor,"_batch(",batch,")",j,"of",(loop_batches+1), "_GCEPRCoords.jld2"),gcvalid_check) 
    elseif batch_num == (loop_batches+1)
        GCEPRCs = GCEPRCoordinate[]
        for i=(1+loop_batches*batch):npoints
            ie,ip,ir,iz = Tuple(subs[i])
            c = GCParticle(energy[ie],pitch[ip],r[ir],z[iz],amu*OrbitTomography.mass_u,q)

            if !in_vessel(wall,r[ir],z[iz])
                o = GCEPRCoordinate(c,'i')  
            else
                try
                    o = getGCEPRCoord(M,wall,c; gcvalid_check=gcvalid_check, toa=toa, kwargs...)
                catch
                    o = GCEPRCoordinate(c,'i')  
                end
            end

            if o.class in ('i','v','l')
                o = GCEPRCoordinate(c,'i')  
            end

            push!(GCEPRCs,o)
        end

        write_GCEPRCoords(GCEPRCs,filename = string(filename_prefactor,"_batch(",batch,")",(loop_batches+1),"of",(loop_batches+1),"_GCEPRCoords.jld2"),gcvalid_check) 
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
        gcvalid_check = read(f["gcvalid_check"])
    close(f)

    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)
    subs = CartesianIndices((nenergy,npitch,nr,nz))

    class = fill('i',(nenergy,npitch,nr,nz))
    point_index = zeros(Int,nenergy,npitch,nr,nz)
    tau_t = zeros(Float64,nenergy,npitch,nr,nz)
    tau_p = zeros(Float64,nenergy,npitch,nr,nz)

    npoints = nenergy*npitch*nr*nz

    GCEPRCoords,gcvalid_check2 = reconstruct_GCEPRCoords(filename_prefactor, batch, loop_batches)

    gcvalid_check2 != gcvalid_check && error("gcvalid_inconsistency")

    for i=1:npoints
        class[subs[i]] = GCEPRCoords[i].class
        tau_p[subs[i]] = GCEPRCoords[i].tau_p
        tau_t[subs[i]] = GCEPRCoords[i].tau_t
    end

    grid_index = filter(i -> GCEPRCoords[i].class != 'i', 1:npoints)
    GCEPRCoords = filter(x -> x.class != 'i', GCEPRCoords)
    npoints = length(GCEPRCoords)
    point_index[grid_index] = 1:npoints

    psgrid = PSGrid(energy,pitch,r,z,fill(1,npoints),point_index,class,tau_p,tau_t)

    if print_results
        length(print_dir)>0 && cd(print_dir)
        write_PSGrid(psgrid,filename = string(filename_prefactor, "PSGrid.jld2"))
        write_GCEPRCoords(GCEPRCoords,filename = string(filename_prefactor, "GCEPRCoords.jld2"),gcvalid_check) 
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

    gcvalid_check = false
    @showprogress for j=1:(loop_batches+1)
        try
            coords,gcvalid_check,vacuum,drift = read_GCEPRCoords(string(filename_prefactor,"_batch(",batch,")",j,"of",(loop_batches+1), "_GCEPRCoords.jld2")) 
            append!(total_GCEPRCoords,coords)
        catch
            push!(batch_error,j)
        end
    end

    if length(batch_error)>0
        display(batch_error)
        error("Batch Reading Error, failed batches seen above.\n")
    end 

    return total_GCEPRCoords, gcvalid_check
end 