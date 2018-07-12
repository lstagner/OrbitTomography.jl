function orbit_grid(M::AxisymmetricEquilibrium, wall, e_range, p_range, r_range;
                    nenergy=100, npitch=100, nr=100, norbits=1000, nstep=12000,
                    tmax=1200.0, dl=0.0, combine=false)

    eo = linspace(e_range...,nenergy)
    po = linspace(p_range...,npitch)
    ro = linspace(r_range...,nr)

    norbs = nenergy*npitch*nr
    orbs = @parallel (vcat) for i=1:norbs
        ie,ip,ir = ind2sub((nenergy,npitch,nr),i)
        c = EPRCoordinate(M,eo[ie],po[ip],ro[ir])
        try
            o = get_orbit(M, c, nstep=nstep, tmax=tmax)
        catch
            o = Orbit(EPRCoordinate(),:incomplete)
        end

        if o.class == :incomplete || o.class == :degenerate || hits_wall(o,wall)
            o = Orbit(o.coordinate,:incomplete)
        else
            if dl > 0.0
                rpath = down_sample(o.path,mean_dl=dl)
                o = Orbit(o.coordinate,o.class,o.tau_p,o.tau_t,rpath)
            end
            if !combine
                o = Orbit(o.coordinate,o.class,o.tau_p,o.tau_t,OrbitPath(typeof(o.tau_p)))
            end
        end
        o
    end
    orbs = filter(x -> x.class != :incomplete, orbs)
    norbs = length(orbs)

    norm = abs.([-(e_range...), -(p_range...), -(r_range...)])
    mins = [e_range[1],p_range[1],r_range[1]]
    oclasses = [:co_passing, :ctr_passing, :potato, :trapped, :stagnation]

    orbit_counts = Dict{Symbol,Int}(o=>count(x -> x.class == o, orbs)
                                    for o in oclasses)
    nclusters = 0
    orbit_grid = eltype(orbs)[]
    for oc in oclasses
        nk = max(Int(ceil(norbits*orbit_counts[oc]/norbs)),1)
        if (nclusters + nk) > norbits
            nk = norbits - nclusters
        end
        nk == 0 && continue
        inds_oc = find([o.class == oc for o in orbs])
        coords = hcat((([o.coordinate.energy, o.coordinate.pitch, o.coordinate.r] .- mins)./norm
                       for o in orbs if o.class == oc)...)

        if nk == 1
            if !combine
                c = coords .* norm .+ mins
                cc = EPRCoordinate(M,mean(c,2)...)
                try
                    o = get_orbit(M, cc.energy,cc.pitch,cc.r,cc.z, nstep=nstep,tmax=tmax)
                    if dl > 0.0
                        rpath = down_sample(o.path,mean_dl=dl)
                        o = Orbit(o.coordinate,o.class,o.tau_p,o.tau_t,rpath)
                    end
                    push!(orbit_grid,o)
                catch
                    o = Orbit(cc)
                    push!(orbit_grid,o)
                end
            else
                o = combine_orbits(orbs[inds_oc])
                push!(orbit_grid,o)
            end
            continue
        end

        k = kmeans(coords,nk)
        if !combine
            coords = k.centers.*norm .+ mins
            for i=1:size(coords,2)
                cc = EPRCoordinate(M,coords[1,i],coords[2,i],coords[3,i])
                try
                    o = get_orbit(M, cc.energy, cc.pitch, cc.r, cc.z, nstep=nstep,tmax=tmax)
                    if dl > 0.0
                        rpath = down_sample(o.path,mean_dl=dl)
                        o = Orbit(o.coordinate,o.class,o.tau_p,o.tau_t,rpath)
                    end
                    push!(orbit_grid,o)
                catch
                    o = Orbit(cc)
                    push!(orbit_grid,o)
                end
            end
        else
            for i=1:nk
                w = k.assignments .== i
                sum(w) == 0 && continue
                o = combine_orbits(orbs[inds_oc[w]])
                push!(orbit_grid,o)
            end
        end
        nclusters = nclusters + nk
    end

    return orbit_grid

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

function mc2orbit(M::AxisymmetricEquilibrium, d::FIDASIMGuidingCenterParticles; tmax=1200.0,nstep=12000)
    orbits = @parallel (vcat) for i=1:d.npart
        o = get_orbit(M,d.energy[i],M.sigma*d.pitch[i],d.r[i]/100,d.z[i]/100,tmax=tmax,nstep=nstep)
        o.coordinate
    end
    return orbits
end

function fbm2orbit(M::AxisymmetricEquilibrium,d::FIDASIMGuidingCenterFunction; n=1_000_000)
    dmc = fbm2mc(d,n=n)
    return mc2orbit(M,dmc)
end

