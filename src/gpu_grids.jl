#ensemblealg=EnsembleSerial()/EnsembleThreads()/
#           EnsembleDistributed()/EnsembleSplitThreads()/EnsembleGPUArray()
function fill_PSGridGPU(M::AbstractEquilibrium,wall::Wall, energy::AbstractVector, pitch::AbstractVector, r::AbstractVector, z::AbstractVector;  
                q=1, amu=OrbitTomography.H2_amu, vacuum = false, drift = true, 
                phi0=0.0,dt_multipler = 1e-1 ,tmin=0.0,tmax_multiplier = 1e6,
                integrator=Tsit5(), ensemblealg=EnsembleThreads(), maxiters=Int(1e6), interp_dt = 0.0, classify_orbits=true, gcvalid_check=false, dual=false, one_transit=true, limit_phi=true, maxphi=10*2*pi, max_length=500, r_callback=true,
                verbose = false, print_results = false, filename_prefactor = "", distributed=true,  default_output=false,  kwargs...)

    rlims, zlims = limits(M)
    if !((rlims[1] < r[1] < rlims[end]) && (rlims[1] < r[end] < rlims[end]) && (zlims[1] < z[1] < zlims[end]) && (zlims[1] < z[end] < zlims[end]))
        error("RZ grid lies outside domain of magnetic field")
    end

    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)

    npoints = nenergy*npitch*nr*nz

    #Make initial ODE. This is basically what prob_func does for each new i.
    m = amu*OrbitTomography.mass_u
    r0,tspan,gcp,dt = initial_condition(M,1,energy,pitch,r,z,m,q,phi0,dt_multipler,tmin,tmax_multiplier,dual)

    i_r, i_phi, i_z = cylindrical_cocos_indices(cocos(M))
    stat = GCStatus(typeof(gcp.r))
    stat.ri = r0
    stat.vi = gc_velocity(M, gcp, r0[i_r], r0[i_z], r0[4], r0[5],vacuum,drift)
    stat.initial_dir = abs(stat.vi[i_r]) > abs(stat.vi[i_z]) ? i_r : i_z # Is it mostly in the R- or z-direction? Element number 1 or 3?

    gc_ode = GuidingCenterOrbits.make_gc_ode(M,gcp,stat,vacuum,drift)
    ode_prob = ODEProblem(gc_ode,r0,tspan,one_transit)

    prob_func1 = make_prob_func(M, stat, energy, pitch, r, z, m, q, phi0, one_transit, dt_multipler, tmin, tmax_multiplier, vacuum, drift, dual)
    output_func1 = make_output_func(M,classify_orbits,gcvalid_check,dual,m,q,tmin,max_length)

    if default_output
        ensemble_prob = EnsembleProblem(ode_prob,prob_func=prob_func1)
    else
        ensemble_prob = EnsembleProblem(ode_prob,prob_func=prob_func1,output_func = output_func1, reduction = reduction1)
    end

    #Make callbacks to send as kwargs to solve. Please see callbacks.jl for specification.
    if wall !== nothing
        wall_cb = GuidingCenterOrbits.wall_callback(wall)
    end
    if one_transit
        cb = CallbackSet(GuidingCenterOrbits.r_cb_passive, GuidingCenterOrbits.pol_cb, GuidingCenterOrbits.oob_cb, GuidingCenterOrbits.brr_cb)
        if wall !== nothing
            cb = CallbackSet(cb.continuous_callbacks..., wall_cb, cb.discrete_callbacks...)
        end
    else
        cb = GuidingCenterOrbits.oob_cb
        if wall !== nothing
            cb = CallbackSet(wall_cb,GuidingCenterOrbits.oob_cb)
        end
    end
    if limit_phi
        phi_cb = GuidingCenterOrbits.phi_callback(maxphi)
        cb = CallbackSet(cb.continuous_callbacks..., phi_cb, cb.discrete_callbacks...)
    end
    
    DifferentialEquations.solve(ensemble_prob,integrator,ensemblealg,trajectories=npoints,reltol=1e-8, abstol=1e-12, verbose=false,callback=cb,adaptive=true,save_start=true,maxiters=maxiters)
end
#sim.u is your reduction
#sim[i] is your ith solution

#CALL WITH: m = amu*OrbitTomography.mass_u
function initial_condition(M::AbstractEquilibrium, i, energy::AbstractVector, pitch::AbstractVector, r::AbstractVector, z::AbstractVector, m::AbstractFloat, q::Int, phi0::AbstractFloat, dt_multipler::AbstractFloat, tmin::AbstractFloat, tmax_multiplier::AbstractFloat, dual::Bool)
    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)
    subs = CartesianIndices((nenergy,npitch,nr,nz))


    ie,ip,ir,iz = Tuple(subs[i]) 
    
    gcp = GCParticle(energy[ie],pitch[ip],r[ir],z[iz],m,q)
    
    if dual
        ed = ForwardDiff.Dual(gcp.energy,  (0.0,0.0,0.0))
        pd = ForwardDiff.Dual(gcp.pitch,  (1.0,0.0,0.0))
        rd = ForwardDiff.Dual(gcp.r,  (0.0,1.0,0.0))
        zd = ForwardDiff.Dual(gcp.z,  (0.0,0.0,1.0))
    
        gcp = GCParticle(ed,pd,rd,zd,gcp.m,gcp.q)
    end

    mc2 = m*c0^2
    p0 = sqrt(((1e3*e0*gcp.energy + mc2)^2 - mc2^2)/(c0*c0)) # The initial particle momentum
    # move abs(pitch) away from one to avoid issues when calculating jacobian;
    # don't try to be clever by using clamp it doesn't work with the autodiff
    if abs(gcp.pitch) == 1.0
        pitch0 = sign(gcp.pitch)*prevfloat(abs(gcp.pitch))
    else
        pitch0 = gcp.pitch
    end
    p_para0 = p0*pitch0*B0Ip_sign(M) # The initial parallel momentum
    p_perp0 = p0*sqrt(1.0 - pitch0^2) # The initial perpendicular momentum
    mu_0 = (p_perp0^2)/(2*gcp.m*norm(Bfield(M,gcp.r,gcp.z))) # The initial (and presumably constant) magnetic moment

    dt=dt_multipler*cyclotron_period(M,gcp)
    tmax=tmax_multiplier*dt
    tspan = tspan = (one(gcp.r)*tmin,one(gcp.r)*tmax)

    r0 = SVector{5}(cylindrical_cocos(cocos(M), gcp.r, one(gcp.r)*phi0, gcp.z)...,
    one(gcp.r)*p_para0, one(gcp.r)*mu_0)

    return r0,tspan,gcp,dt # The initial guiding-centre element vector
end

#if classify_orbits, each sol = [sol[ind],class,sol.retcode]
#if !classify_orbits, each sol = [sol[ind],sol.retcode]
function make_output_func(M::AbstractEquilibrium,classify_orbits::Bool,gcvalid_check::Bool,dual::Bool,m::AbstractFloat,q::Int,tmin::AbstractFloat,max_length::Int)
    if !dual
        if !gcvalid_check
            output_func = function fout1(sol::DESolution,i)
                if sol.retcode == :Pol_terminated
                    ir, iphi, iz = cylindrical_cocos_indices(cocos(M))
                    #Get tau_p, tau_t
                        tau_p = sol.t[end] # Set the current time to be the poloidal transit time
                        tau_t = 2pi*tau_p/abs(sol[iphi,end] - sol[iphi,1]) # Compute the toroidal transit time

                    #Get extremum coords, JUST WITH R-vec
                        ind = argmax(sol[ir,:])

                        rm = sol[ir,ind]
                        zm = sol[iz,ind]
                        ppara_m = sol[4,ind]
                        mu_m = sol[5,ind]
                        tm = (tau_p - sol.t[ind])/tau_p

                        pm = get_pitch(M, m, ppara_m, mu_m, rm, zm)
                        energy = get_kinetic_energy(M, m, ppara_m, mu_m, rm, zm)

                    #Get starting GCP coods
                        r0 = sol[ir,1]
                        z0 = sol[iz,1]
                        ppara0 = sol[4,1]
                        mu0 = sol[5,1]
                        p0 = get_pitch(M, m, ppara0, mu0, r0, z0)

                    #Get class 
                        #HERE I ONLY INTERPOLATE IF ITS A LONG SOLUTION, IS THIS WORTH IT?
                        (length(sol) > max_length) && (sol = sol(range(tmin,sol.t[end],length=max_length))) # Interpolate solution onto n points evently spaced in time

                        r = getindex.(sol.u,ir)
                        z = getindex.(sol.u,iz)
                        ppara = getindex.(sol.u,4)
                        mu = getindex.(sol.u,5)
                        pitch = get_pitch.(M, m, ppara, mu, r, z)

                        class = class_char(classify(r,z,pitch,magnetic_axis(M)))
                    return ((GCEPRCoordinate(energy,p0,r0,z0,pm,rm,zm,tm,class,tau_p,tau_t,0.0,false,m,q),i),false)
                else
                    return (nothing,false)
                end
            end
        else
            output_func = function fout2(sol::DESolution,i)
                if sol.retcode == :Pol_terminated
                    ir, iphi, iz = cylindrical_cocos_indices(cocos(M))
                    #Get tau_p, tau_t
                        tau_p = sol.t[end] # Set the current time to be the poloidal transit time
                        tau_t = 2pi*tau_p/abs(sol[iphi,end] - sol[iphi,1]) # Compute the toroidal transit time

                    #Get extremum coords, JUST WITH R-vec
                        #r = getindex.(sol.u,ir)
                        ind = argmax(sol[ir,:])

                        rm = sol[ir,ind]
                        zm = sol[iz,ind]
                        ppara_m = sol[4,ind]
                        mu_m = sol[5,ind]
                        tm = (tau_p - sol.t[ind])/tau_p

                        pm = get_pitch(M, m, ppara_m, mu_m, rm, zm)
                        energy = get_kinetic_energy(M, m, ppara_m, mu_m, rm, zm)

                    #Get starting GCP coods
                        r0 = sol[ir,1]
                        z0 = sol[iz,1]
                        ppara0 = sol[4,1]
                        mu0 = sol[5,1]
                        p0 = get_pitch(M, m, ppara0, mu0, r0, z0)

                        #z = getindex.(sol.u,iz)
                        #ppara = getindex.(sol.u,4)
                        #mu = getindex.(sol.u,5)
                        #pitch = get_pitch.(M, m, ppara, mu, r, z)

                    #Get class and gcvalid
                        #HERE I INTERPOLATE REGARDLESS OF SOLUTION LENGTH, BECAUSE I ASSUME GCValid BETTER WHEN EVENLY SPACED?
                        n = min(max_length, length(sol))
                        sol = sol(range(tmin,sol.t[end],length=n)) # Interpolate solution onto n points evently spaced in time

                        r = getindex.(sol.u,ir)
                        phi = getindex.(sol.u,iphi)
                        z = getindex.(sol.u,iz)
                        ppara = getindex.(sol.u,4)
                        mu = getindex.(sol.u,5)
                        pitch = get_pitch.(M, m, ppara, mu, r, z)

                        class = class_char(classify(r,z,pitch,magnetic_axis(M)))
                        gcvalid = gcde_check(M,energy,m,e0*q,r,phi,z,pitch)
                    return ((GCEPRCoordinate(energy,p0,r0,z0,pm,rm,zm,tm,class,tau_p,tau_t,0.0,gcvalid,m,q),i),false)
                else
                    return (nothing,false)
                end
            end
        end
    else 
        if !gcvalid_check
            output_func = function fout3(sol::DESolution,i)
                if sol.retcode == :Pol_terminated
                    ir, iphi, iz = cylindrical_cocos_indices(cocos(M))
                    #Get tau_p, tau_t
                        tau_p = sol.t[end] # Set the current time to be the poloidal transit time
                        tau_t = 2pi*tau_p/abs(sol[iphi,end] - sol[iphi,1]) # Compute the toroidal transit time

                    #Get extremum coords, JUST WITH R-vec
                        ind = argmax(sol[ir,:])

                        rm = sol[ir,ind]
                        zm = sol[iz,ind]
                        ppara_m = sol[4,ind]
                        mu_m = sol[5,ind]
                        tm = (tau_p - sol.t[ind])/tau_p

                        pm = get_pitch(M, m, ppara_m, mu_m, rm, zm)
                        energy = ForwardDiff.value(get_kinetic_energy(M, m, ppara_m, mu_m, rm, zm))
                        jacdet = max(abs(det(hcat((ForwardDiff.partials(x) for x in [pm,rm,tm])...))),0.0)

                    #Get starting GCP coods
                        r0 = ForwardDiff.value(sol[ir,1])
                        z0 = ForwardDiff.value(sol[iz,1])
                        ppara0 = ForwardDiff.value(sol[4,1])
                        mu0 = ForwardDiff.value(sol[5,1])
                        p0 = get_pitch(M, m, ppara0, mu0, r0, z0)

                    #Get class 
                        #HERE I ONLY INTERPOLATE IF ITS A LONG SOLUTION, IS THIS WORTH IT?
                        (length(sol) > 2*max_length) && (sol = sol(range(tmin,sol.t[end],length=max_length))) # Interpolate solution onto n points evently spaced in time

                        r = ForwardDiff.value.(getindex.(sol.u,ir))
                        z = ForwardDiff.value.(getindex.(sol.u,iz))
                        ppara = ForwardDiff.value.(getindex.(sol.u,4))
                        mu = ForwardDiff.value.(getindex.(sol.u,5))
                        pitch = get_pitch.(M, m, ppara, mu, r, z)

                        class = class_char(classify(r,z,pitch,magnetic_axis(M)))
                    return ((GCEPRCoordinate(energy,p0,r0,z0,ForwardDiff.value(pm),ForwardDiff.value(rm),ForwardDiff.value(zm),ForwardDiff.value(tm),class,ForwardDiff.value(tau_p),ForwardDiff.value(tau_t),jacdet,false,m,q),i),false)
                else
                    return (nothing,false)
                end
            end
        else
            output_func = function fout4(sol::DESolution,i)
                if sol.retcode == :Pol_terminated
                    ir, iphi, iz = cylindrical_cocos_indices(cocos(M))
                    #Get tau_p, tau_t
                        tau_p = sol.t[end] # Set the current time to be the poloidal transit time
                        tau_t = 2pi*tau_p/abs(sol[iphi,end] - sol[iphi,1]) # Compute the toroidal transit time

                    #Get extremum coords, JUST WITH R-vec
                        ind = argmax(sol[ir,:])

                        rm = sol[ir,ind]
                        zm = sol[iz,ind]
                        ppara_m = sol[4,ind]
                        mu_m = sol[5,ind]
                        tm = (tau_p - sol.t[ind])/tau_p

                        pm = get_pitch(M, m, ppara_m, mu_m, rm, zm)
                        energy = ForwardDiff.value(get_kinetic_energy(M, m, ppara_m, mu_m, rm, zm))
                        jacdet = max(abs(det(hcat((ForwardDiff.partials(x) for x in [pm,rm,tm])...))),0.0)

                    #Get starting GCP coods
                        r0 = ForwardDiff.value(sol[ir,1])
                        z0 = ForwardDiff.value(sol[iz,1])
                        ppara0 = ForwardDiff.value(sol[4,1])
                        mu0 = ForwardDiff.value(sol[5,1])
                        p0 = get_pitch(M, m, ppara0, mu0, r0, z0)

                    #Get class and gcvalid
                        #HERE I INTERPOLATE REGARDLESS OF SOLUTION LENGTH, BECAUSE I ASSUME GCValid BETTER WHEN EVENLY SPACED?
                        n = min(max_length, length(sol))
                        sol = sol(range(tmin,sol.t[end],length=n)) # Automatically interpolate solution onto n points evently spaced in time

                        r = ForwardDiff.value.(getindex.(sol.u,ir))
                        phi = ForwardDiff.value.(getindex.(sol.u,iphi))
                        z = ForwardDiff.value.(getindex.(sol.u,iz))
                        ppara = ForwardDiff.value.(getindex.(sol.u,4))
                        mu = ForwardDiff.value.(getindex.(sol.u,5))
                        pitch = get_pitch.(M, m, ppara, mu, r, z)

                        gcvalid = gcde_check(M,energy,m,e0*q,r,phi,z,pitch)
                        class = class_char(classify(r,z,pitch,magnetic_axis(M)))
                    return ((GCEPRCoordinate(energy,p0,r0,z0,ForwardDiff.value(pm),ForwardDiff.value(rm),ForwardDiff.value(zm),ForwardDiff.value(tm),class,ForwardDiff.value(tau_p),ForwardDiff.value(tau_t),jacdet,gcvalid,m,q),i),false)
                else
                    return (nothing,false)
                end
            end
        end
    end
    return output_func
end

function reduction1(u,batch,I)
    filter!(x->!isnothing(x), batch)
    append!(u,batch)
    u,false
end

#CALL WITH: m = amu*OrbitTomography.mass_u
#Make prob_func:
    #inputs: prob,i,repeat
        #prob = ODEProblem
        #i = index
        #repeat -> bool
    #outputs: 
        #new prob but with u0 changed
function make_prob_func(M::AbstractEquilibrium, stat::GCStatus, energy::AbstractVector, 
    pitch::AbstractVector, r::AbstractVector, z::AbstractVector, m::AbstractFloat, 
    q::Int, phi0::AbstractFloat, one_transit::Bool, dt_multipler::AbstractFloat, tmin::AbstractFloat, 
    tmax_multiplier::AbstractFloat, vacuum::Bool, drift::Bool,dual::Bool)

    prob_func = function fprob(prob,i,repeat)

        u0new,tspan_new,gcp,dts = initial_condition(M,i,energy,pitch,r,z,m,q,phi0,dt_multipler,tmin,tmax_multiplier,dual)

        i_r, i_phi, i_z = cylindrical_cocos_indices(cocos(M))
        stat = GCStatus(typeof(gcp.r))
        stat.ri = u0new
        stat.vi = gc_velocity(M, gcp, u0new[i_r], u0new[i_z], u0new[4], u0new[5],vacuum,drift) #THIS HERE
        stat.initial_dir = abs(stat.vi[i_r]) > abs(stat.vi[i_z]) ? i_r : i_z # Is it mostly in the R- or z-direction? Element number 1 or 3?

        gc_ode = GuidingCenterOrbits.make_gc_ode(M,gcp,stat,vacuum,drift) #I NEED TO CALL THIS TO UPDATE THE STAT IN THE GC_ODE..

        ODEProblem(gc_ode,u0new,tspan_new,one_transit, dt=dts, force_dtmin=true)
    end
    return prob_func
end