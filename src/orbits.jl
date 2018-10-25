struct OrbitGrid{T}
    energy::AbstractVector{T}
    pitch::AbstractVector{T}
    r::AbstractVector{T}
    counts::Vector{Int}
    orbit_index::Array{Int,3}
    class::Array{Symbol,3}
    tau_p::Array{T,3}
    tau_t::Array{T,3}
end

function Base.show(io::IO, og::OrbitGrid)
    print(io, "OrbitGrid: $(length(og.energy))×$(length(og.pitch))×$(length(og.r)):$(length(og.counts))")
end

function orbit_grid(M::AxisymmetricEquilibrium, eo::AbstractVector, po::AbstractVector, ro::AbstractVector;
                    kwargs...)

    nenergy = length(eo)
    npitch = length(po)
    nr = length(ro)

    orbit_index = zeros(Int,nenergy,npitch,nr)
    class = fill(:incomplete,(nenergy,npitch,nr))
    tau_t = zeros(Float64,nenergy,npitch,nr)
    tau_p = zeros(Float64,nenergy,npitch,nr)

    norbs = nenergy*npitch*nr
    subs = CartesianIndices((nenergy,npitch,nr))

    p = Progress(norbs)
    channel = RemoteChannel(()->Channel{Bool}(norbs), 1)
    orbs = fetch(@sync begin
        @async while take!(channel)
            next!(p)
        end
        @async begin
            orbs = @distributed (vcat) for i=1:norbs
                ie,ip,ir = Tuple(subs[i])
                c = EPRCoordinate(M,eo[ie],po[ip],ro[ir])
                try
                    o = get_orbit(M, c; kwargs...)
                catch
                    o = Orbit(EPRCoordinate(),:incomplete)
                end

                if o.class in (:incomplete,:degenerate,:lost)
                    o = Orbit(o.coordinate,:incomplete)
                end
                put!(channel, true)
                o
            end
            put!(channel, false)
            orbs
        end
    end)

    for i=1:norbs
        class[subs[i]] = orbs[i].class
        tau_p[subs[i]] = orbs[i].tau_p
        tau_t[subs[i]] = orbs[i].tau_t
    end

    grid_index = filter(i -> orbs[i].class != :incomplete, 1:norbs)
    orbs = filter(x -> x.class != :incomplete, orbs)
    norbs = length(orbs)
    orbit_index[grid_index] = 1:norbs

    return orbs, OrbitGrid(eo,po,ro,fill(1,norbs),orbit_index,class,tau_p,tau_t)

end

function segment_orbit_grid(M::AxisymmetricEquilibrium, orbit_grid::OrbitGrid, orbits::Vector;
                        norbits=1000, combine=(length(orbits[1].path) != 0), kwargs...)

    eo = orbit_grid.energy
    po = orbit_grid.pitch
    ro = orbit_grid.r

    nenergy = length(eo)
    npitch = length(po)
    nr = length(ro)
    norbs = length(orbits)

    e_range = extrema(eo)
    p_range = extrema(po)
    r_range = extrema(ro)
    orbs_index = zeros(Int,norbs)
    for i = 1:length(orbit_grid.orbit_index)
        ii = orbit_grid.orbit_index[i]
        ii == 0 && continue
        orbs_index[ii] != 0 && continue
        orbs_index[ii] = i
    end

    orbit_index = zeros(Int,nenergy,npitch,nr)

    norm = abs.([-(e_range...), -(p_range...), -(r_range...)])
    mins = [e_range[1],p_range[1],r_range[1]]
    oclasses = [:potato, :stagnation, :trapped, :co_passing, :ctr_passing]

    orbit_counts = Dict{Symbol,Int}(o=>count(x -> x.class == o, orbits)
                                    for o in oclasses)

    nclusters = 0
    orbs = eltype(orbits)[]
    orbit_num = 0
    for oc in oclasses
        nk = max(Int(ceil(norbits*orbit_counts[oc]/norbs)),1)
        if (nclusters + nk) > norbits
            nk = norbits - nclusters
        end
        nk == 0 && continue

        inds_oc = findall([o.class == oc for o in orbits])
        coords = hcat((([o.coordinate.energy, o.coordinate.pitch, o.coordinate.r] .- mins)./norm
                       for o in orbits if o.class == oc)...)

        if nk == 1
            if !combine
                c = coords .* norm .+ mins
                cc = EPRCoordinate(M,mean(c,dims=2)...)
                try
                    o = get_orbit(M, GCParticle(cc.energy,cc.pitch,cc.r,cc.z,cc.m,cc.q); kwargs...)
                    push!(orbs,o)
                    orbit_num = orbit_num + 1
                catch
                    o = Orbit(cc)
                    push!(orbs,o)
                    orbit_num = orbit_num + 1
                end
            else
                o = combine_orbits(orbits[inds_oc])
                push!(orbs,o)
                orbit_num = orbit_num + 1
            end
            orbit_index[orbs_index[inds_oc]] .= orbit_num
            nclusters = nclusters + nk
            continue
        end

        k = kmeans(coords,nk)
        if !combine
            coords = k.centers.*norm .+ mins
            for i=1:size(coords,2)
                w = k.assignments .== i
                sum(w) == 0 && continue
                cc = EPRCoordinate(M,coords[1,i],coords[2,i],coords[3,i])
                try
                    o = get_orbit(M, GCParticle(cc.energy,cc.pitch,cc.r,cc.z,cc.m,cc.q); kwargs...)
                    push!(orbs,o)
                    orbit_num = orbit_num + 1
                catch
                    o = Orbit(cc)
                    push!(orbs,o)
                    orbit_num = orbit_num + 1
                end
                orbit_index[orbs_index[inds_oc[w]]] .= orbit_num
            end
        else
            for i=1:nk
                w = k.assignments .== i
                sum(w) == 0 && continue
                o = combine_orbits(orbits[inds_oc[w]])
                push!(orbs,o)
                orbit_num = orbit_num + 1
                orbit_index[orbs_index[inds_oc[w]]] .= orbit_num
            end
        end
        nclusters = nclusters + nk
    end

    counts = [count(x -> x == i, orbit_index) for i=1:length(orbs)]
    return orbs, OrbitGrid(eo,po,ro,counts,orbit_index,orbit_grid.class,orbit_grid.tau_p,orbit_grid.tau_t)

end

function write_orbit_grid(grid::OrbitGrid;filename="orbit_grid.h5")
    h5open(filename,"w") do file
        file["energy"] = collect(grid.energy)
        file["pitch"] = collect(grid.pitch)
        file["r"] = collect(grid.r)
        file["counts"] = grid.counts
        file["orbit_index"] = grid.orbit_index
        file["class"] = String.(grid.class)
        file["tau_p"] = grid.tau_p
        file["tau_t"] = grid.tau_p
    end
    nothing
end

function read_orbit_grid(filename)
    isfile(filename) || error("File does not exist")

    f = h5open(filename)
    energy = read(f["energy"])
    pitch = read(f["pitch"])
    r = read(f["r"])
    counts = read(f["counts"])
    orbit_index = read(f["orbit_index"])
    class = Symbol.(read(f["class"]))
    tau_p = read(f["tau_p"])
    tau_t = read(f["tau_t"])
    close(f)

    return OrbitGrid(energy,pitch,r,counts,orbit_index,class,tau_p,tau_t)
end

function Base.map(grid::OrbitGrid, f::Vector)
    if length(grid.counts) != length(f)
        throw(ArgumentError("Incompatible sizes"))
    end
    return [i == 0 ? zero(f[1]) : f[i]/grid.counts[i] for i in grid.orbit_index]
end

function Base.bin(grid::OrbitGrid, orbits; weights::Union{Nothing,Vector})

    if weights != nothing
        length(weights) == length(orbits) || error("Incompatible weight vector size")
        w = weights
    else
        w = fill(1.0,length(orbits))
    end

    f = zeros(length(grid.counts))
    for (io,o) in enumerate(orbits)
        i = argmin(abs.(o.energy .- grid.energy))
        j = argmin(abs.(o.pitch .- grid.pitch))
        k = argmin(abs.(o.r .- grid.r))
        ind = grid.orbit_index[i,j,k]
        if ind != 0
            f[ind] += w[io]
        end
    end

    return f
end

function combine_orbits(orbits)
    norbits = length(orbits)
    norbits == 1 && return orbits[1]

    o = orbits[1]
    r = o.path.r
    z = o.path.z
    phi = o.path.phi
    pitch = o.path.pitch
    energy = o.path.energy
    dt = o.path.dt
    dl = o.path.dl

    c = o.coordinate
    isa(c, EPRCoordinate) || error("Wrong orbit coordinate. Expected EPRCoordinate")
    ec = c.energy
    pc = c.pitch
    rc = c.r
    zc = c.z
    tau_p = o.tau_p
    tau_t = o.tau_t

    for i=2:norbits
        oo = orbits[i]
        ec = ec + oo.coordinate.energy
        pc = pc + oo.coordinate.pitch
        rc = rc + oo.coordinate.r
        zc = zc + oo.coordinate.z
        tau_t = tau_t + oo.tau_t
        tau_p = tau_p + oo.tau_p
        append!(r, oo.path.r)
        append!(z, oo.path.z)
        append!(phi, oo.path.phi)
        append!(pitch, oo.path.pitch)
        append!(energy, oo.path.energy)
        append!(dt, oo.path.dt)
        append!(dl, oo.path.dl)
    end
    ec = ec/norbits
    pc = pc/norbits
    rc = rc/norbits
    zc = zc/norbits
    tau_p = tau_p/norbits
    tau_t = tau_p/norbits

    cc = EPRCoordinate(ec,pc,rc,zc,c.amu,c.q)
    path = OrbitPath(r,z,phi,pitch,energy,dt,dl)

    if all(x -> x.class == orbits[1].class, orbits)
        class = orbits[1].class
    else
        class = :meta
    end

    return Orbit(cc, class, tau_p, tau_t, path)
end

function mc2orbit(M::AxisymmetricEquilibrium, d::FIDASIMGuidingCenterParticles; kwargs...)
    orbits = @distributed (vcat) for i=1:d.npart
        o = get_orbit(M,GCParticle(d.energy[i],M.sigma*d.pitch[i],d.r[i]/100,d.z[i]/100); kwargs...,store_path=false)
        o.coordinate
    end
    return orbits
end

function fbm2orbit(M::AxisymmetricEquilibrium,d::FIDASIMGuidingCenterFunction; n=1_000_000, kwargs...)
    dmc = fbm2mc(d,n=n)
    return mc2orbit(M,dmc; kwargs...)
end

struct LocalDistribution{T<:AbstractMatrix,S<:AbstractVector}
    d::T
    energy::S
    pitch::S
end

function local_distribution(M::AxisymmetricEquilibrium, grid::OrbitGrid, f::Vector, r, z;
                            energy=1:80, pitch=-1.0:0.02:1.0, kwargs...)
    f3d = map(grid,f)
    nenergy = length(energy)
    npitch = length(pitch)
    d = zeros(length(energy),length(pitch))
    @showprogress "Calculating Local Distribution..." for i=1:nenergy, j=1:npitch
        v, detJ = eprz_to_eprt(M, energy[i], pitch[j], r, z; kwargs...)
        ii = argmin(abs.(v[1] .- grid.energy))
        jj = argmin(abs.(v[2] .- grid.pitch))
        kk = argmin(abs.(v[3] .- grid.r))
        d[i,j] = detJ*f3d[ii,jj,kk]
    end
    return LocalDistribution(d,energy,pitch)
end
