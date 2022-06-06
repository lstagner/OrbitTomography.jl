struct EPDensity{T<:AbstractMatrix,S<:AbstractVector}
    d::T
    detJ::T
    energy::S
    pitch::S
end

struct PSDist2EPR_info
    orbit_weight::Float64
    path_indices::Union{Array{Int,2},Nothing}
    path_jacobians::Union{Array{Float64,1},Nothing}
end

function local_distribution(M::AbstractEquilibrium, grid::OrbitGrid, f::Vector, r, z;
                            energy=1.0:80.0, pitch=-1.0:0.02:1.0, nearest=false,kwargs...)
    f3d = map(grid,f)
    nenergy = length(energy)
    npitch = length(pitch)
    if nearest
        inds = filter(x->grid.orbit_index[x] != 0, CartesianIndices(grid.orbit_index))
        data = hcat(([grid.energy[I[1]], grid.pitch[I[2]], grid.r[I[3]]] for I in inds)...)
        tree = KDTree(data)
    end

    d = zeros(length(energy),length(pitch))
    detJ = zeros(length(energy),length(pitch))
    for i=1:nenergy, j=1:npitch
        for rr in r, zz in z
            v, dJ = eprz_to_eprt(M, energy[i], pitch[j], rr, zz; adaptive=false, kwargs...)
            dJ == 0 && continue
            if nearest
                idxs, dists = knn(tree,v[1:3],1,false)
                ii,jj,kk = Tuple(inds[idxs[1]])
            else
                !(grid.energy[1] <= v[1] <= grid.energy[end]) && continue
                !(grid.pitch[1] <= v[2] <= grid.pitch[end]) && continue
                !(grid.r[1] <= v[3] <= grid.r[end]) && continue
                ii = argmin(abs.(v[1] .- grid.energy))
                jj = argmin(abs.(v[2] .- grid.pitch))
                kk = argmin(abs.(v[3] .- grid.r))
            end
            d[i,j] = d[i,j] + dJ*f3d[ii,jj,kk]
            detJ[i,j] = dJ
        end
    end
    d = d/(length(r)*length(z))
    return EPDensity(d,detJ,energy,pitch)
end

function local_distribution(M::AbstractEquilibrium, OS::OrbitSystem, f, orbs, sigma, r, z;
                            Js=Vector{Vector{Float64}}[],
                            energy=range(1.0,80.0,length=25),
                            pitch=range(-0.99,0.99,length=25),
                            distributed=false, atol=1e-3,
                            covariance=:local, norms=S3(1.0,1.0,1.0), kwargs...)

    nenergy = length(energy)
    npitch = length(pitch)
    m = orbs[1].coordinate.m
    q = orbs[1].coordinate.q
    #lorbs = Array{Orbit}(undef,nenergy,npitch)
    #@showprogress for i=1:nenergy
    #    for j = 1:npitch
    #        lorbs[i,j] = get_orbit(M, GCParticle(energy[i],pitch[j],r,z,m,q); kwargs...)
    #    end
    #end
    #lorbs = reshape(lorbs,nenergy*npitch)
    lorbs = reshape([get_orbit(M, GCParticle(energy[i],pitch[j],r,z,m,q); kwargs...) for i=1:nenergy,j=1:npitch],nenergy*npitch)
    if distributed
        lJs = pmap(o->get_jacobian(M,o), lorbs, on_error = ex->zeros(2))
                   #batch_size=round(Int, nenergy*npitch/(5*nprocs())))
    else
        lJs = [get_jacobian(M, o) for o in lorbs]
    end

    if covariance == :local
        Si = get_covariance_matrix(M, lorbs, orbs, sigma, Js_1=lJs, Js_2=Js,
                                   distributed=distributed, atol=atol)
    else
        Si = get_global_covariance_matrix(lorbs, orbs, sigma, norms=norms)
    end

    f_ep = reshape(max.(Si*(OS.S_inv*f),0.0),nenergy,npitch)
    detJ = reshape([length(j) != 0 ? j[1] : 0.0 for j in lJs],nenergy,npitch)
    f_ep .= f_ep./detJ
    f_ep[detJ .== 0.0] .= 0.0
    w = reshape([o.class in (:lost, :incomplete, :unknown) for o in lorbs],nenergy, npitch)
    f_ep[w] .= 0.0

    return EPDensity(f_ep,1.0./detJ,energy,pitch)
end

struct RZDensity{T<:AbstractMatrix,S<:AbstractVector}
    d::T
    r::S
    z::S
end

function rz_profile(M::AbstractEquilibrium, OS::OrbitSystem, f::Vector, orbs, sigma;
                            Js=Vector{Vector{Float64}}[],
                            energy=range(1.0,80.0,length=25),
                            pitch=range(-0.99,0.99,length=25),
                            r = range(limits(M)[1]...,length=25),
                            z = range(limits(M)[2]...,length=25),
                            distributed=false, atol=1e-3, domain_check= (xx,yy) -> true,
                            covariance=:local, norms=S3(1.0,1.0,1.0),
                            checkpoint=true, warmstart=false,file="rz_progress.jld2", kwargs...)

    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)
    inds = CartesianIndices((nr,nz))
    m = orbs[1].coordinate.m
    q = orbs[1].coordinate.q

    f_rz = zeros(nr,nz)

    if warmstart && isfile(file) && (filesize(file) != 0)
        @load file f_rz last_ind
    else
        last_ind = inds[1]
    end

    if checkpoint
        touch(file)
    end

    for I in inds
        (I != inds[1] && I < last_ind) && continue
        i = I[1]
        j = I[2]
        rr = r[i]
        zz = z[j]
        !(domain_check(rr,zz)) && continue

        lorbs = reshape([get_orbit(M, GCParticle(energy[k],pitch[l],rr,zz,m,q); kwargs...) for k=1:nenergy,l=1:npitch],nenergy*npitch)
        if distributed
            lJs = pmap(o->get_jacobian(M,o), lorbs, on_error = ex->zeros(2))
                       #batch_size=round(Int, nenergy*npitch/(5*nprocs())))
        else
            lJs = [get_jacobian(M, o) for o in lorbs]
        end

        if covariance == :local
            Si = get_covariance_matrix(M, lorbs, orbs, sigma, Js_1=lJs, Js_2=Js,
                                       distributed=distributed, atol=atol)
        else
            Si = get_global_covariance_matrix(lorbs, orbs, sigma, norms=norms)
        end

        f_ep = reshape(max.(Si*(OS.S_inv*f),0.0),nenergy,npitch)
        detJ = reshape([length(j) != 0 ? j[1] : 0.0 for j in lJs],nenergy,npitch)
        f_ep .= f_ep./detJ
        f_ep[detJ .== 0.0] .= 0.0
        w = reshape([o.class in (:lost, :incomplete, :unknown) for o in lorbs],nenergy, npitch)
        f_ep[w] .= 0.0
        f_rz[i,j] = sum(f_ep)*step(energy)*step(pitch)
        if checkpoint
            @save file f_rz last_ind
        end
        last_ind = I
    end
    return RZDensity(f_rz,r,z)
end

struct EPRZDensity{T<:AbstractArray,S<:AbstractVector}
    d::T
    energy::S
    pitch::S
    r::S
    z::S
end

function eprz_dist(M::AbstractEquilibrium, OS::OrbitSystem, f::Vector, orbs, sigma;
                            Js=Vector{Vector{Float64}}[],
                            energy=range(1.0,80.0,length=25),
                            pitch=range(-0.99,0.99,length=25),
                            r = range(limits(M)[1]...,length=25),
                            z = range(limits(M)[2]...,length=25),
                            distributed=false, atol=1e-3, domain_check= (xx,yy) -> true,
                            covariance=:local, norms=S3(1.0,1.0,1.0),
                            checkpoint=true, warmstart=false,file="eprz_progress.jld2", kwargs...)

    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)
    inds = CartesianIndices((nr,nz))
    m = orbs[1].coordinate.m
    q = orbs[1].coordinate.q

    f_eprz = zeros(nenergy,npitch,nr,nz)

    if warmstart && isfile(file) && (filesize(file) != 0)
        progress_file = jldopen(file,false,false,false,IOStream)
        f_eprz = progress_file["f_eprz"]
        last_ind = progress_file["last_ind"]
        close(progress_file)
    else
        last_ind = inds[1]
    end

    if checkpoint
        touch(file)
    end

    for I in inds
        (I != inds[1] && I < last_ind) && continue
        i = I[1]
        j = I[2]
        rr = r[i]
        zz = z[j]
        !(domain_check(rr,zz)) && continue

        lorbs = reshape([get_orbit(M, GCParticle(energy[k],pitch[l],rr,zz,m,q); kwargs...) for k=1:nenergy,l=1:npitch],nenergy*npitch)
        if distributed
            lJs = pmap(o->get_jacobian(M,o), lorbs, on_error = ex->zeros(2))
                       #batch_size=round(Int, nenergy*npitch/(5*nprocs())))
        else
            lJs = [get_jacobian(M, o) for o in lorbs]
        end

        if covariance == :local
            Si = get_covariance_matrix(M, lorbs, orbs, sigma, Js_1=lJs, Js_2=Js,
                                       distributed=distributed, atol=atol)
        else
            Si = get_global_covariance_matrix(lorbs, orbs, sigma, norms=norms)
        end

        f_ep = reshape(max.(Si*(OS.S_inv*f),0.0),nenergy,npitch)
        detJ = reshape([length(j) != 0 ? j[1] : 0.0 for j in lJs],nenergy,npitch)
        f_ep .= f_ep./detJ
        f_ep[detJ .== 0.0] .= 0.0
        w = reshape([o.class in (:lost, :incomplete, :unknown) for o in lorbs],nenergy, npitch)
        f_ep[w] .= 0.0
        f_eprz[:,:,i,j] .= f_ep
        if checkpoint
            progress_file = jldopen(file, true,true,true,IOStream)
            write(progress_file,"f_eprz",f_eprz)
            write(progress_file,"last_ind",last_ind)
            close(progress_file)
        end
        last_ind = I
    end

    return EPRZDensity(f_eprz,energy,pitch,r,z)
end

function epr2ps(F_os_VEC::AbstractArray{Float64},oenergy,opitch,or,PS_orbs::Vector{GCEPRCoordinate},og_orbit_index::Array{Int,3},og_class::Array{Symbol,3}; topological_force=true, simple_trans = true, mislabelled=false, distributed=false)
    num_orbs = length(PS_orbs)
    (mislabelled || topological_force) && (distributed=false)

    dE = oenergy[2]-oenergy[1]
    dp = opitch[2]-opitch[1]
    dr = or[2]-or[1]

    if distributed
        F_ps_VEC = @showprogress @distributed (vcat) for iter=1:num_orbs
            i = argmin(abs.(PS_orbs[iter].energy .- oenergy))
            j = argmin(abs.(PS_orbs[iter].pitch_m .- opitch))
            k = argmin(abs.(PS_orbs[iter].r_m .- or))
            ind = og_orbit_index[i,j,k]

            if ind != 0
                F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
            else 
                F_ps_i = 0.0
            end

            #Checking Energy Bounds:
            if (PS_orbs[iter].energy > (oenergy[end]+0.5*(oenergy[end]-oenergy[end-1]))) || (PS_orbs[iter].energy < (oenergy[1]-0.5*(oenergy[2]-oenergy[1]))) || (PS_orbs[iter].energy < 0.0)
                F_ps_i = 0.0
            end
            #Checking Pitch Bounds:
            if opitch[end]>opitch[1]
                if (PS_orbs[iter].pitch_m > (opitch[end]+0.5*abs(opitch[end]-opitch[end-1]))) || (PS_orbs[iter].pitch_m < (opitch[1]-0.5*abs(opitch[2]-opitch[1]))) || abs(PS_orbs[iter].pitch_m) > 1.0
                    F_ps_i = 0.0
                end
            else
                if (PS_orbs[iter].pitch_m < (opitch[end]-0.5*abs(opitch[end]-opitch[end-1]))) || (PS_orbs[iter].pitch_m > (opitch[1]+0.5*abs(opitch[2]-opitch[1]))) || abs(PS_orbs[iter].pitch_m) > 1.0
                    F_ps_i = 0.0
                end
            end
            #Rely on wall for R-bounds

            F_ps_i
        end
    else
        F_ps_VEC = Float64[] 
        p = Progress(num_orbs)

        if mislabelled 
            mislabelled_PSOrb_Ints = Int[]
            corresponding_OGOrb_locations = []
            mislabelledPS_types = Symbol[]
            mislabelledOS_types = Symbol[]
            if topological_force
                mislabelled_fixed_PSOrb_Ints = Int[]
                fixed_PSdist_differences = Float64[]
                corresponding_fixed_OGOrb_locations = []
            end
        end

        for iter=1:num_orbs
            i = argmin(abs.(PS_orbs[iter].energy .- oenergy))
            j = argmin(abs.(PS_orbs[iter].pitch_m .- opitch))
            k = argmin(abs.(PS_orbs[iter].r_m .- or))
            ind = og_orbit_index[i,j,k]

            F_ps_i = 0.0
            if !topological_force
                if ind != 0
                    F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
                    if mislabelled && (PS_orbs[iter].class != og_class[i,j,k])
                        push!(mislabelled_PSOrb_Ints,iter)
                        push!(corresponding_OGOrb_locations,[i,j,k])
                        push!(mislabelledPS_types,PS_orbs[iter].class)
                        push!(mislabelledOS_types,og_class[i,j,k])
                    end
                else 
                    F_ps_i = 0.0
                end
            else 
                dist_E = (PS_orbs[iter].energy - oenergy[i])/dE
                dist_p = (PS_orbs[iter].pitch_m - opitch[j])/dp
                dist_r = (PS_orbs[iter].r_m - or[k])/dr

                if dist_E >= 0.0 #POINT TO THE RIGHT OF GRIDPOINT, THIS IS DISTANCE TO RIGHT
                    e_start_ind=i

                    if i==length(oenergy)
                        E_ints = [e_start_ind]
                        dist_Es = [dist_E]
                    else
                        e_end_ind=i+1
                        E_ints = [e_start_ind,e_end_ind]
                        dist_Es = [dist_E,1-dist_E]
                    end
                else #POINT TO THE LEFT OF GRIDPOINT, DISTANCE TO RIGHT IS abs(dist_E)
                    e_end_ind=i

                    if (i-1)==0 
                        E_ints = [e_end_ind]
                        dist_Es = [abs(dist_E)]
                    else
                        e_start_ind=i-1
                        E_ints = [e_start_ind,e_end_ind]
                        dist_Es = [1-abs(dist_E),abs(dist_E)]
                    end
                end

                if dist_p >= 0.0 #POINT TO THE RIGHT OF GRIDPOINT, THIS IS DISTANCE TO RIGHT
                    p_start_ind=j

                    if j==length(opitch)
                        p_ints = [p_start_ind]
                        dist_ps = [dist_p]
                    else
                        p_end_ind=j+1
                        p_ints = [p_start_ind,p_end_ind]
                        dist_ps = [dist_p,1-dist_p]
                    end
                else #POINT TO THE LEFT OF GRIDPOINT, DISTANCE TO RIGHT IS abs(dist_p)
                    p_end_ind=j

                    if (j-1)==0 
                        p_ints = [p_end_ind]
                        dist_ps = [abs(dist_p)]
                    else
                        p_start_ind=j-1
                        p_ints = [p_start_ind,p_end_ind]
                        dist_ps = [1-abs(dist_p),abs(dist_p)]
                    end
                end

                if dist_r >= 0.0 #POINT TO THE RIGHT OF GRIDPOINT, THIS IS DISTANCE TO RIGHT
                    r_start_ind=k

                    if k==length(or)
                        r_ints = [r_start_ind]
                        dist_rs = [dist_r]
                    else
                        r_end_ind=k+1
                        r_ints = [r_start_ind,r_end_ind]
                        dist_rs = [dist_r,1-dist_r]
                    end
                else #POINT TO THE LEFT OF GRIDPOINT, DISTANCE TO RIGHT IS abs(dist_r)
                    r_end_ind=k

                    if (k-1)==0 
                        r_ints = [r_end_ind]
                        dist_rs = [abs(dist_r)]
                    else
                        r_start_ind=k-1
                        r_ints = [r_start_ind,r_end_ind]
                        dist_rs = [1-abs(dist_r),abs(dist_r)]
                    end
                end

                distances = Float64[]
                locations = Array{Int64,1}[]
                types = Symbol[]

                for (eo,ee) in enumerate(dist_Es)
                    for (po,pp) in enumerate(dist_ps)
                        for (ro,rr) in enumerate(dist_rs)
                            push!(distances,(ee^2+pp^2+rr^2))
                            (ee<0.0||pp<0.0||rr<0.0) && error("Distances incorrectly calculated.")
                            push!(locations,[E_ints[eo],p_ints[po],r_ints[ro]])
                            push!(types,og_class[E_ints[eo],p_ints[po],r_ints[ro]])
                        end
                    end
                end
                
                matching_type_inds = filter(x -> types[x]==PS_orbs[iter].class, 1:length(types))
                locations = locations[matching_type_inds]
                distances = distances[matching_type_inds]

                (PS_orbs[iter].class == :incomplete) && error("Shouldn't be any incomplete orbits.")

                if length(matching_type_inds)==0 #No points in the surrounding grid have the same type -> revert to nearest binning
                    if ind != 0
                        F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
                    else 
                        F_ps_i = 0.0
                    end

                    if mislabelled && (PS_orbs[iter].class != og_class[i,j,k])
                        push!(mislabelled_PSOrb_Ints,iter)
                        push!(corresponding_OGOrb_locations,[i,j,k])
                        push!(mislabelledPS_types,PS_orbs[iter].class)
                        push!(mislabelledOS_types,og_class[i,j,k])
                    end
                else
                    best_match_ind = argmin(distances)

                    indFIXED = og_orbit_index[locations[best_match_ind][1],locations[best_match_ind][2],locations[best_match_ind][3]]
                    F_ps_i = F_os_VEC[indFIXED]*PS_orbs[iter].jacdet

                    if mislabelled && (PS_orbs[iter].class != og_class[i,j,k])
                        push!(mislabelled_PSOrb_Ints,iter)
                        push!(mislabelled_fixed_PSOrb_Ints,iter)
                        push!(corresponding_OGOrb_locations,[i,j,k])
                        push!(mislabelledPS_types,PS_orbs[iter].class)
                        push!(mislabelledOS_types,og_class[i,j,k])
                        push!(corresponding_fixed_OGOrb_locations,locations[best_match_ind])

                        if ind==0
                            push!(fixed_PSdist_differences, PS_orbs[iter].jacdet*F_os_VEC[indFIXED])
                        else
                            push!(fixed_PSdist_differences,PS_orbs[iter].jacdet*(F_os_VEC[indFIXED]-F_os_VEC[ind]))
                        end
                    else
                        if indFIXED != ind 
                            display([i,j,k])
                            display(locations[best_match_ind])
                            error("Topological_force error.")
                        end
                    end
                end
            end

            #Checking Energy Bounds:
            if (PS_orbs[iter].energy > (oenergy[end]+0.5*(oenergy[end]-oenergy[end-1]))) || (PS_orbs[iter].energy < (oenergy[1]-0.5*(oenergy[2]-oenergy[1]))) || (PS_orbs[iter].energy < 0.0)
                F_ps_i = 0.0
            end
            #Checking Pitch Bounds:
            if opitch[end]>opitch[1]
                if (PS_orbs[iter].pitch_m > (opitch[end]+0.5*abs(opitch[end]-opitch[end-1]))) || (PS_orbs[iter].pitch_m < (opitch[1]-0.5*abs(opitch[2]-opitch[1]))) || abs(PS_orbs[iter].pitch_m) > 1.0
                    F_ps_i = 0.0
                end
            else
                if (PS_orbs[iter].pitch_m < (opitch[end]-0.5*abs(opitch[end]-opitch[end-1]))) || (PS_orbs[iter].pitch_m > (opitch[1]+0.5*abs(opitch[2]-opitch[1]))) || abs(PS_orbs[iter].pitch_m) > 1.0
                    F_ps_i = 0.0
                end
            end
            #Rely on wall for R-bounds

            push!(F_ps_VEC,F_ps_i)
            ProgressMeter.next!(p)
        end
    end

    if topological_force && mislabelled
        corresponding_OGOrb_locations_matrix = zeros(Int,length(corresponding_OGOrb_locations),3)
        corresponding_OGOrb_fixed_locations_matrix = zeros(Int,length(corresponding_fixed_OGOrb_locations),3)
        OGclasses = Vector{String}()
        PSclasses = Vector{String}()

        for (io,o) in enumerate(corresponding_OGOrb_locations)
            corresponding_OGOrb_locations_matrix[io,:]=o
            push!(OGclasses,string(mislabelledOS_types[io]))
            push!(PSclasses,string(mislabelledPS_types[io]))
        end
        for (io,o) in enumerate(corresponding_fixed_OGOrb_locations)
            corresponding_OGOrb_fixed_locations_matrix[io,:]=o
        end

        return F_ps_VEC,mislabelled_PSOrb_Ints,mislabelled_fixed_PSOrb_Ints,fixed_PSdist_differences,corresponding_OGOrb_locations_matrix,corresponding_OGOrb_fixed_locations_matrix,PSclasses,OGclasses
    elseif mislabelled
        corresponding_OGOrb_locations_matrix = zeros(Int,length(corresponding_OGOrb_locations),3)
        OGclasses = Vector{String}()
        PSclasses = Vector{String}()

        for (io,o) in enumerate(corresponding_OGOrb_locations)
            corresponding_OGOrb_locations_matrix[io,:]=o
            push!(OGclasses,string(mislabelledOS_types[io]))
            push!(PSclasses,string(mislabelledPS_types[io]))
        end
        return F_ps_VEC,mislabelled_PSOrb_Ints,corresponding_OGOrb_locations_matrix,PSclasses,OGclasses
    else
        return F_ps_VEC
    end
end

function epr2ps_splined(F_os_VEC::AbstractArray{Float64}, orbs::Vector{Orbit}, PS_orbs::Vector{GCEPRCoordinate}, og_orbit_index::Array{Int,3}, og_class::Array{Symbol,3}; verbose=true, distributed=false, k::Int=2, overlap=true, stiffness_factor::Float64=0.0)
    ctr_passing_points = Array{Float64}(undef, 0,3)
    ctr_passing_vals = Float64[]
    trapped_points = Array{Float64}(undef, 0,3)
    trapped_vals = Float64[]
    co_passing_points = Array{Float64}(undef, 0,3)
    co_passing_vals = Float64[]
    

    p = Progress(length(orbs))
    verbose && print("Sorting orbits into types.\n")
    for (io,i) in orbs
        if i.class == :ctr_passing
            ctr_passing_points = vcat(ctr_passing_points,[i.coordinate.energy,i.coordinate.pitch,i.coordinate.r]') #is this efficient?
            push!(ctr_passing_vals,F_os_VEC[io])
        end
        if (i.class == :trapped || i.class == :stagnation || i.class == :potato)
            trapped_points = vcat(trapped_points,[i.coordinate.energy,i.coordinate.pitch,i.coordinate.r]')
            push!(trapped_vals,F_os_VEC[io])
        end
        if (i.class == :co_passing || i.class == :stagnation || i.class == :potato)
            co_passing_points = vcat(co_passing_points,[i.coordinate.energy,i.coordinate.pitch,i.coordinate.r]')
            push!(co_passing_vals,F_os_VEC[io])
        end
        ProgressMeter.next!(p)
    end

    #ctr_passing_points = transpose(ctr_passing_points)
    #trapped_points = transpose(trapped_points)
    #co_passing_points = transpose(co_passing_points)

    ctr_passing_spline = PolyharmonicSpline(k,ctr_passing_points,ctr_passing_vals)
    trapped_spline = PolyharmonicSpline(k,trapped_points,trapped_vals)
    co_passing_spline = PolyharmonicSpline(k,co_passing_points,co_passing_vals)

    verbose && print("Evaluating particle-space distribution values.\n")
    if distributed
        PS_dist = @showprogress @distributed (vcat) for i in PS_orbs
            if i.class == :ctr_passing
                val = i.jacdet*ctr_passing_spline.(i.energy,i.pitch_m,i.r_m)
            elseif i.class == :co_passing
                val = i.jacdet*co_passing_spline.(i.energy,i.pitch_m,i.r_m)
            else    
                val = i.jacdet*trapped_spline.(i.energy,i.pitch_m,i.r_m)
            end

            val
        end
    else
        p = Progress(length(PS_orbs))
        PS_dist = Float64[]
        for i in PS_orbs
            if i.class == :ctr_passing
                val = i.jacdet*ctr_passing_spline.(i.energy,i.pitch_m,i.r_m)
            elseif i.class == :co_passing
                val = i.jacdet*co_passing_spline.(i.energy,i.pitch_m,i.r_m)
            else    
                val = i.jacdet*trapped_spline.(i.energy,i.pitch_m,i.r_m)
            end

            push!(PS_dist,val)
            ProgressMeter.next!(p)
        end
    end

    return PS_dist
end

function ps2epr(F_ps_Weights::AbstractArray{Float64},PS_Grid::PSGrid, og_orbs::Vector{Orbit}; bin_info = false, jac_info = false, matrix_input=false, distributed=false) 
    if distributed 
        F_os_VEC = @showprogress @distributed (vcat) for iter=1:length(og_orbs)
            orb = og_orbs[iter]
            integrated_weight = 0.0

            if bin_info
                path_inds = zeros(Int,(length(orb.path.r)-1),4)
            else    
                path_inds = nothing
            end

            if jac_info
                path_jacs = zeros(Float64,(length(orb.path.r)-1))
            else    
                path_jacs = nothing
            end

            for path_iter = 1:(length(orb.path.r)-1)
                i = argmin(abs.(orb.path.energy[path_iter] .- PS_Grid.energy))
                j = argmin(abs.(orb.path.pitch[path_iter] .- PS_Grid.pitch))
                k = argmin(abs.(orb.path.r[path_iter] .- PS_Grid.r))
                l = argmin(abs.(orb.path.z[path_iter] .- PS_Grid.z))
                ind = PS_Grid.point_index[i,j,k,l]

                if bin_info
                    path_inds[path_iter,:] = [i,j,k,l]
                end
                if jac_info
                    path_jacs[path_iter] = orb.path.jacdets[path_iter]
                end

                if ind != 0
                    matrix_input ? F_os_i = F_ps_Weights[i,j,k,l]*orb.path.jacdets[path_iter] : F_os_i = F_ps_Weights[ind]*orb.path.jacdets[path_iter]
                else 
                    F_os_i = 0.0
                end

                #Checking Energy Bounds:
                if (orb.path.energy[path_iter] > (PS_Grid.energy[end]+0.5*(PS_Grid.energy[end]-PS_Grid.energy[end-1]))) || (orb.path.energy[path_iter] < (PS_Grid.energy[1]-0.5*(PS_Grid.energy[2]-PS_Grid.energy[1]))) || (orb.path.energy[path_iter] < 0.0)
                    F_os_i = 0.0
                end
                #Checking Pitch Bounds:
                if PS_Grid.pitch[end]>PS_Grid.pitch[1]
                    if (orb.path.pitch[path_iter] > (PS_Grid.pitch[end]+0.5*abs(PS_Grid.pitch[end]-PS_Grid.pitch[end-1]))) || (orb.path.pitch[path_iter] < (PS_Grid.pitch[1]-0.5*abs(PS_Grid.pitch[2]-PS_Grid.pitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
                        F_os_i = 0.0
                    end
                else
                    if (orb.path.pitch[path_iter] < (PS_Grid.pitch[end]-0.5*abs(PS_Grid.pitch[end]-PS_Grid.pitch[end-1]))) || (orb.path.pitch[path_iter] > (PS_Grid.pitch[1]+0.5*abs(PS_Grid.pitch[2]-PS_Grid.pitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
                        F_os_i = 0.0
                    end
                end
                #Rely on wall for RZ-bounds

                if path_iter==1
                    integrated_weight += F_os_i*(0.5*orb.path.dt[1]+0.5*orb.path.dt[end-1])/orb.tau_p
                else
                    integrated_weight += F_os_i*(0.5*orb.path.dt[path_iter]+0.5*orb.path.dt[path_iter-1])/orb.tau_p
                end
            end

            (bin_info || jac_info) ? PSDist2EPR_info(integrated_weight,path_inds,path_jacs) : integrated_weight
        end
    else
        if (bin_info || jac_info) 
            F_os_VEC = PSDist2EPR_info[]
        else
            F_os_VEC = Float64[]
        end

        p = Progress(length(og_orbs))

        for iter=1:length(og_orbs)
            orb = og_orbs[iter]
            integrated_weight = 0.0

            if bin_info
                path_inds = zeros(Int,(length(orb.path.r)-1),4)
            else    
                path_inds = nothing
            end

            if jac_info
                path_jacs = zeros(Float64,(length(orb.path.r)-1))
            else    
                path_jacs = nothing
            end

            for path_iter = 1:(length(orb.path.r)-1)
                i = argmin(abs.(orb.path.energy[path_iter] .- PS_Grid.energy))
                j = argmin(abs.(orb.path.pitch[path_iter] .- PS_Grid.pitch))
                k = argmin(abs.(orb.path.r[path_iter] .- PS_Grid.r))
                l = argmin(abs.(orb.path.z[path_iter] .- PS_Grid.z))
                ind = PS_Grid.point_index[i,j,k,l]

                if bin_info
                    path_inds[path_iter,:] = [i,j,k,l]
                end
                if jac_info
                    path_jacs[path_iter] = orb.path.jacdets[path_iter]
                end

                if ind != 0
                    matrix_input ? F_os_i = F_ps_Weights[i,j,k,l]*orb.path.jacdets[path_iter] : F_os_i = F_ps_Weights[ind]*orb.path.jacdets[path_iter]
                else 
                    F_os_i = 0.0
                end

                #Checking Energy Bounds:
                if (orb.path.energy[path_iter] > (PS_Grid.energy[end]+0.5*(PS_Grid.energy[end]-PS_Grid.energy[end-1]))) || (orb.path.energy[path_iter] < (PS_Grid.energy[1]-0.5*(PS_Grid.energy[2]-PS_Grid.energy[1]))) || (orb.path.energy[path_iter] < 0.0)
                    F_os_i = 0.0
                end
                #Checking Pitch Bounds:
                if PS_Grid.pitch[end]>PS_Grid.pitch[1]
                    if (orb.path.pitch[path_iter] > (PS_Grid.pitch[end]+0.5*abs(PS_Grid.pitch[end]-PS_Grid.pitch[end-1]))) || (orb.path.pitch[path_iter] < (PS_Grid.pitch[1]-0.5*abs(PS_Grid.pitch[2]-PS_Grid.pitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
                        F_os_i = 0.0
                    end
                else
                    if (orb.path.pitch[path_iter] < (PS_Grid.pitch[end]-0.5*abs(PS_Grid.pitch[end]-PS_Grid.pitch[end-1]))) || (orb.path.pitch[path_iter] > (PS_Grid.pitch[1]+0.5*abs(PS_Grid.pitch[2]-PS_Grid.pitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
                        F_os_i = 0.0
                    end
                end
                #Rely on wall for RZ-bounds

                if path_iter==1
                    integrated_weight += F_os_i*(0.5*orb.path.dt[1]+0.5*orb.path.dt[end-1])/orb.tau_p
                else
                    integrated_weight += F_os_i*(0.5*orb.path.dt[path_iter]+0.5*orb.path.dt[path_iter-1])/orb.tau_p
                end
            end 
            ProgressMeter.next!(p)

            (bin_info || jac_info) ? push!(F_os_VEC,PSDist2EPR_info(integrated_weight,path_inds,path_jacs)) : push!(F_os_VEC,integrated_weight) 
        end
    end

    if (bin_info || jac_info)
        F_os = Float64[]

        for i=1:length(og_orbs)
            push!(F_os,F_os_VEC[i].orbit_weight)
        end

        return F_os,F_os_VEC
    end
    return F_os_VEC
end

function ps2epr_splined(F_ps_Weights::AbstractArray{Float64},PS_Grid::PSGrid, og_orbs::Vector{Orbit}; k::Int=2, og_ebounds = [0.0,0.0], matrix_input=false, distributed=false)
    energy = PS_Grid.energy
    pitch = PS_Grid.pitch
    r = PS_Grid.r
    z = PS_Grid.z

    if og_ebounds != [0.0,0.0]
        de = abs(energy[2]-energy[1])
        if (max(og_ebounds)>(max(energy)+0.5*de)) || ((min(og_ebounds)<(min(energy)-0.5*de)))
            warn("Your orbit-space grid extends beyond the energy boundaries of your particle-space grid. This will cause extrapolation errors.")
        end
    end

    """
    if distributed
        nenergy = length(energy)
        npitch = length(pitch)
        nr = length(r)
        nz = length(z)

        npoints = nenergy*npitch*nr*nz
        subs = CartesianIndices((nenergy,npitch,nr,nz))

        points = SharedArray{Float64}(undef, length(F_ps_Weights),4)
        shared_PSindex = convert(SharedArray,PS_Grid.point_index)

        @distributed for i = 1:npoints
            if shared_PSindex[subs[i]] != 0 
                ie,ip,ir,iz = Tuple(subs[i])
                point = [energy[ie],pitch[ip],r[ir],z[iz]]'
            end
            
        #pusher = function 
        #pmap()

        #can use PSOrbs, and distributed for loop put the values into a shared array.

    else
        nenergy = length(energy)
        npitch = length(pitch)
        nr = length(r)
        nz = length(z)

        npoints = nenergy*npitch*nr*nz
        subs = CartesianIndices((nenergy,npitch,nr,nz))

        points = Array{Float64}(undef, length(F_ps_Weights),4)

        for i = 1:npoints
            if PS_Grid.point_index[subs[i]] != 0 
                ie,ip,ir,iz = Tuple(subs[i])
                points = vcat(points, [energy[ie],pitch[ip],r[ir],z[iz]]')
            end
        end

        spline = PolyharmonicSpline(k,points,F_ps_Weights)
        
        os_dist = Float64[]
        for i in og_orbs
            integrated_weight = 0.0

            for p in 1:length(i.path.r)
                F_os_i = i.path.jacdets[p]*spline(i.path.energy[p],i.path.pitch[p],i.path.r[p],i.path.z[p])

                if p==1
                    integrated_weight += F_os_i*(0.5*orb.path.dt[1]+0.5*orb.path.dt[end-1])/orb.tau_p
                else
                    integrated_weight += F_os_i*(0.5*orb.path.dt[path_iter]+0.5*orb.path.dt[path_iter-1])/orb.tau_p
                end
            end

            push!(os_dist,integrated_weight)
        end
    end
    """

    #return os_dist
    return nothing
end




        p = Progress(length(PS_orbs))
        PS_dist = Float64[]
        for i in PS_orbs
            if i.class == :ctr_passing
                val = i.jacdet*ctr_passing_spline.(i.energy,i.pitch_m,i.r_m)
            elseif i.class == :co_passing
                val = i.jacdet*co_passing_spline.(i.energy,i.pitch_m,i.r_m)
            else    
                val = i.jacdet*trapped_spline.(i.energy,i.pitch_m,i.r_m)
            end

            push!(PS_dist,val)
            ProgressMeter.next!(p)
        end
    end

#function ps2epr_splined(F_ps_Weights::AbstractArray{Float64},PS_Grid::PSGrid, og_orbs; bin_info = false, jac_info = false, matrix_input=false, distributed=false)

#function ps2epr_sampled(F_ps_Weights::AbstractArray{Float64},PS_Grid::PSGrid, og_orbs; bin_info = false, jac_info = false, matrix_input=false, distributed=false)

function ps2os(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, F_ps::Array{T,4}, energy::AbstractVector, pitch::AbstractVector, R::AbstractVector, Z::AbstractVector, og::OrbitGrid; numOsamples::Int64, progress_prefactor = "", two_pi_mod = false, genuine_PS_weights = false, verbose=false, GCP = GCDeuteron, distributed=true, nbatch = 1_000_000, saveProgress=true, kwargs...) where {T,N}
    if !its4D
        f4d = PS_VectorToMatrix(f,psgrid)
    else
        f4d = f
    end
    
    if genuine_PS_weights #genuine PS_weights don't have to be reshaped by R - R is the jacobian determinant associated with going from cartesian to cylindrical coordinates.
        fr = F_ps
    else    
        fr = F_ps.*reshape(R,(1,1,length(R),1))
    end

    if two_pi_mod 
        two_pi = 2*pi
    else
        two_pi = 1.0
    end

    if verbose
        println("sum(fr): $(sum(fr))")
    end

    if verbose
        println("Acquiring fr, dvols and nfast... ")
    end
    dvols = (2*pi*two_pi) .* get4Dvols(energy, pitch, R, Z)
    nfast = sum(fr.*dvols) #if two_pi_mod = true, integrating out gyro angle as well as toroidal angle

    # Keeping memory usage to minimum
    F_ps = nothing
    dE = nothing
    dp = nothing
    dR = nothing
    dZ = nothing
    dE4D = nothing
    dp4D = nothing
    dR4D = nothing
    dZ4D = nothing

    # Checking if units of R,Z is meters or centimeters. Correct to meters if units is centimeters
    if maximum(R)>100.0 # Assume no tokamak has a major radius larger than 100 meters...
        R = R./100.0
        Z = Z./100.0
    end

    # Handle re-start of ps2os-transformation process, if abruptly cancelled
    if isfile(string(progress_prefactor,"ps2os_progress.jld2"))
        myfile = jldopen(string(progress_prefactor,"ps2os_progress.jld2"),false,false,false,IOStream)
        numOsamples_sofar = deepcopy(myfile["numOsamples"])
        result_sofar = deepcopy(myfile["F_os"])
        numOsamples = numOsamples - numOsamples_sofar

        if length(og.counts) != length(result_sofar)
            error("Loaded orbit-space fast-ion distribution from progress-file is non-compatible with specified orbit grid. Please correct orbit grid, provide valid progress-file or remove progress-file from work directory.")
        end
        close(myfile)
    else
        numOsamples_sofar = 0
        result_sofar = zeros(size(og.counts))
    end

    dims = size(fr) # Tuple
    frdvols_cumsum_vector = cumsum(vec(fr .*dvols)) # Vector
    subs = CartesianIndices(dims) # 4D matrix
    fr = nothing # Memory efficiency
    dvols = nothing # Memory efficiency
    return ps2os_performance(M, wall, frdvols_cumsum_vector, subs, nfast, energy, pitch, R, Z, og; numOsamples=numOsamples, numOsamples_sofar=numOsamples_sofar, result_sofar=result_sofar, distributed=distributed, GCP=GCP, saveProgress=saveProgress, verbose=verbose, kwargs...)
end

"""
ps2os_performance(M, wall, fr, dvols, nfast, energy, pitch, R, Z, og; numOsamples=100_000)

The performance version of part of ps2os(). This function will likely completely replace ps2os() in the near future. It calculates necessary quantities
once instead of for every sample (as ps2os() does). Such as the element-wise product of fr and dvols, and its cumulative sum.
"""
#STUART MODIFIED Energy bounds (Search #^STUART MODIFIED)
function ps2os_performance(M::AbstractEquilibrium, 
                            wall::Union{Nothing,Wall}, 
                            frdvols_cumsum_vector::AbstractVector, 
                            subs::CartesianIndices{4,NTuple{4,Base.OneTo{Int64}}}, 
                            nfast::Union{Int64,Float64}, 
                            energy::AbstractVector, 
                            pitch::AbstractVector, 
                            R::AbstractVector, 
                            Z::AbstractVector, 
                            og::OrbitGrid; 
                            numOsamples::Int64, 
                            numOsamples_sofar=0, 
                            result_sofar=zeros(size(og.counts)), 
                            distributed=true, 
                            GCP=GCDeuteron, 
                            saveProgress=true, 
                            nbatch = 1_000_000, 
                            verbose=false, 
                            kwargs...)
    verbose && println("Pre-computing difference vectors... ")
    dE_vector = vcat(abs.(diff(energy)),abs(energy[end]-energy[end-1]))
    dp_vector = vcat(abs.(diff(pitch)),abs(pitch[end]-pitch[end-1]))
    dR_vector = vcat(abs.(diff(R)),abs(R[end]-R[end-1]))
    dZ_vector = vcat(abs.(diff(Z)),abs(Z[end]-Z[end-1]))

    verbose && println("Going into the distributed part... ")
    if distributed
        subdivide = false
        while numOsamples > nbatch
            subdivide = true
            numOsamples = numOsamples - nbatch
            verbose && println("Samples left: $(numOsamples)")
            result_p = performance_helper(M, wall, nbatch, frdvols_cumsum_vector, subs, dE_vector, dp_vector, dR_vector, dZ_vector, energy, pitch, R, Z, og; GCP=GCP, kwargs...)
            result_sofar .+= result_p
            numOsamples_sofar += nbatch
            if saveProgress
                rm(string(progress_prefactor,"ps2os_progress.jld2"), force=true) #clear the previous file
                myfile = jldopen(string(progress_prefactor,"ps2os_progress.jld2"),true,true,false,IOStream)
                write(myfile,"F_os",result_sofar)
                write(myfile,"numOsamples",numOsamples_sofar)
                close(myfile)
            end
        end
        verbose && println("(Rest) Samples left: $(numOsamples)")
        result_rest = performance_helper(M, wall, numOsamples, frdvols_cumsum_vector, subs, dE_vector, dp_vector, dR_vector, dZ_vector, energy, pitch, R, Z, og; wall=wall, GCP=GCP, kwargs...)
        numOsamples_rest = numOsamples

        if subdivide
            result = result_sofar + result_rest
            numOsamples = numOsamples_sofar + numOsamples_rest
        else
            result = result_rest
            numOsamples = numOsamples_rest
        end
    else # ...if not parallel computing... I wish you good luck.
        for i=numOsamples_sofar+1:numOsamples
            if verbose
                println("Sample number: $(i)")
            end
            # Sample
            p = rand()*frdvols_cumsum_vector[end]
            j = searchsortedfirst(frdvols_cumsum_vector,p,Base.Order.Forward)
            inds = collect(Tuple(subs[j])) # First sample
            r = rand(4) .- 0.5 # 4 stands for the number of dimensions. 0.5 to sample within a hypercube
            E_sample = max(energy[inds[1]] + r[1]*dE_vector[inds[1]], 0.0)
            p_sample = pitch[inds[2]] + r[2]*dp_vector[inds[2]]
            R_sample = R[inds[3]] + r[3]*dR_vector[inds[3]]
            Z_sample = Z[inds[4]] + r[4]*dZ_vector[inds[4]]

            # CHECK IF IT'S A GOOD SAMPLE
            good_sample = checkIfGoodSample(E_sample, p_sample, R_sample, Z_sample, energy, pitch, R, Z)

            if good_sample
                o = get_orbit(M,GCP(E_sample,M.sigma*p_sample,R_sample,Z_sample); store_path=false, wall=wall, kwargs...)
                if (o.coordinate.energy <= (og.energy[end]+0.5*(og.energy[end]-og.energy[end-1]))) && (o.coordinate.energy >= (og.energy[1]-0.5*(og.energy[2]-og.energy[1])))
                    #^STUART MODIFIED if (o.coordinate.energy <= maximum(og.energy) && o.coordinate.energy >= minimum(og.energy))
                    F_os_i = bin_orbits(og,vec([o.coordinate]),weights=vec([1.0]))
                else
                    F_os_i = zeros(length(og.counts))
                end
            else
                F_os_i = zeros(length(og.counts))
            end

            result_sofar .+= F_os_i

            if (i%nbatch)==0 && saveProgress # Every nbatch sample, save
                rm(string(progress_prefactor,"ps2os_progress.jld2"), force=true) #clear the previous file
                myfile = jldopen(string(progress_prefactor,"ps2os_progress.jld2"),true,true,false,IOStream)
                write(myfile,"F_os",result_sofar)
                write(myfile,"numOsamples",i)
                close(myfile)
            end
        end
        result = result_sofar
    end

    if verbose
        println("Number of good samples/All samples: $(sum(result)/numOsamples)")
    end
    rm(string(progress_prefactor,"ps2os_progress.jld2"), force=true) # As in ps2os(), remove the progress file that is no longer needed

    return result, nfast
end

"""
    performance_helper(M, wall, nbatch, frdvols_cumsum_vector, subs, dE_vector, dp_vector, dR_vector, dZ_vector, energy, pitch, R, Z, og;  GCP=GCP, kwargs...)

Help the function ps2os_performance() with acquiring orbit samples when parallel computations are desired.
This is to enable the sampling process to be saved regularly when calculating a large number of samples.
If the sampling process is not saved, then progress will be lost when the super-user of the HPC terminates
the sampling process early, due to misinterpretation of Julia's way of distributed computing.
"""
#STUART MODIFIED Energy bounds (Search #^STUART MODIFIED)
function performance_helper(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, numOsamples::Int64, frdvols_cumsum_vector::Array{Float64,1}, subs::CartesianIndices{4,NTuple{4,Base.OneTo{Int64}}}, dE_vector::Array{Float64,1}, dp_vector::Array{Float64,1}, dR_vector::Array{Float64,1}, dZ_vector::Array{Float64,1}, energy::Array{Float64,1}, pitch::Array{Float64,1}, R::Array{Float64,1}, Z::Array{Float64,1}, og::OrbitGrid; GCP=GCDeuteron, visualizeProgress=true, kwargs...)
    if numOsamples>0.0 # If there are actually a non-zero number of samples left to sample...
        if visualizeProgress
            p = Progress(numOsamples) # Define the progress bar
            channel = RemoteChannel(()->Channel{Bool}(numOsamples),1) # Define the channel from which the progress bar draws data
            result = fetch(@sync begin # Start the distributed computational process, fetch result when done
                @async while take!(channel) # An asynchronous process, with no need for sync, since it simply displays the progress bar
                    ProgressMeter.next!(p)
                end

                @async begin # No internal syncronization needed here either, only external sync needed
                    F_os = @distributed (+) for i=1:numOsamples # Compute one result, and reduce (add) it to a resulting vector F_os

                        # Sample
                        p = rand()*frdvols_cumsum_vector[end]
                        j = searchsortedfirst(frdvols_cumsum_vector,p,Base.Order.Forward)
                        inds = collect(Tuple(subs[j])) # First sample
                        r = rand(4) .- 0.5 # 4 stands for the number of dimensions. 0.5 to sample within a hypercube
                        E_sample = max(energy[inds[1]] + r[1]*dE_vector[inds[1]], 0.0)
                        p_sample = pitch[inds[2]] + r[2]*dp_vector[inds[2]]
                        R_sample = R[inds[3]] + r[3]*dR_vector[inds[3]]
                        Z_sample = Z[inds[4]] + r[4]*dZ_vector[inds[4]]

                        # CHECK IF IT'S A GOOD SAMPLE
                        good_sample = checkIfGoodSample(E_sample, p_sample, R_sample, Z_sample, energy, pitch, R, Z)
                        if good_sample
                            # WHY M.sigma* ??????????????????????????????????????
                            o = get_orbit(M,GCP(E_sample,M.sigma*p_sample,R_sample,Z_sample); store_path=false, wall=wall, kwargs...) # Calculate the orbit
                            if (o.coordinate.energy <= (og.energy[end]+0.5*(og.energy[end]-og.energy[end-1]))) && (o.coordinate.energy >= (og.energy[1]-0.5*(og.energy[2]-og.energy[1])))
                                #^STUART MODIFIED if (o.coordinate.energy <= maximum(og.energy) && o.coordinate.energy >= minimum(og.energy)) # Make sure it's within the energy bounds
                                F_os_i = bin_orbits(og,vec([o.coordinate]),weights=vec([1.0])) # Bin to the orbit grid
                            else
                                F_os_i = zeros(length(og.counts)) # Otherwise, zero
                            end
                        else
                            F_os_i = zeros(length(og.counts)) # Otherwise, zero
                        end
                        put!(channel,true) # Update the progress bar
                        F_os_i # Declare F_os_i as result to add to F_os
                    end
                    put!(channel,false) # Update progress bar
                    F_os # Delcare F_os as done/result, so it can be fetched
                end
            end)
        else
            result = @distributed (+) for i=1:numOsamples
                # Sample
                p = rand()*frdvols_cumsum_vector[end]
                j = searchsortedfirst(frdvols_cumsum_vector,p,Base.Order.Forward)
                inds = collect(Tuple(subs[j])) # First sample
                r = rand(4) .- 0.5 # 4 stands for the number of dimensions. 0.5 to sample within a hypercube
                E_sample = max(energy[inds[1]] + r[1]*dE_vector[inds[1]], 0.0)
                p_sample = pitch[inds[2]] + r[2]*dp_vector[inds[2]]
                R_sample = R[inds[3]] + r[3]*dR_vector[inds[3]]
                Z_sample = Z[inds[4]] + r[4]*dZ_vector[inds[4]]

                # CHECK IF IT'S A GOOD SAMPLE
                good_sample = checkIfGoodSample(E_sample, p_sample, R_sample, Z_sample, energy, pitch, R, Z)
                if good_sample
                    # WHY M.sigma* ??????????????????????????????????????
                    o = get_orbit(M,GCP(E_sample,M.sigma*p_sample,R_sample,Z_sample); store_path=false, wall=wall, kwargs...) # Calculate the orbit
                    if (o.coordinate.energy <= (og.energy[end]+0.5*(og.energy[end]-og.energy[end-1]))) && (o.coordinate.energy >= (og.energy[1]-0.5*(og.energy[2]-og.energy[1])))
                    #^STUART MODIFIED if (o.coordinate.energy <= maximum(og.energy) && o.coordinate.energy >= minimum(og.energy)) # Make sure it's within the energy bounds
                        F_os_i = bin_orbits(og,vec([o.coordinate]),weights=vec([1.0])) # Bin to the orbit grid
                    else
                        F_os_i = zeros(length(og.counts)) # Otherwise, zero
                    end
                else
                    F_os_i = zeros(length(og.counts)) # Otherwise, zero
                end
                F_os_i # Declare F_os_i as result to add to result
            end
        end

        return result
    else # ...otherwise just return a zero vector with the length of the number of valid orbits
        return zeros(length(og.counts))
    end
end

"""
checkIfGoodSample(E_sample, p_sample, R_sample, Z_sample, energy, pitch, R, Z)

The function checks if a sample is within bounds. Returns true if that is the case. Otherwise false. Bounds on samples do not assume equal grid-steps, and pitch values half a step beyond boundaries are included to ensure similar behaviour to ps2epr().
""" 
function checkIfGoodSample(E_sample::Float64, p_sample::Float64, R_sample::Float64, Z_sample::Float64, energy::AbstractArray, pitch::AbstractArray, R::AbstractArray, Z::AbstractArray)

    if pitch[end]>pitch[1]
        if E_sample<=0.0 || p_sample>=(pitch[end]+0.5*(pitch[end]-pitch[end-1])) || p_sample<=(pitch[1]-0.5*(pitch[2]-pitch[1]))|| R_sample>=R[end] || R_sample<=R[1] || Z_sample>=Z[end] || Z_sample<=Z[1] || abs(p_sample)>0.9999
            return false
        else
            return true
        end
    else
        if E_sample<=0.0 || p_sample>=(pitch[1]+0.5*abs(pitch[1]-pitch[2])) || p_sample<=(pitch[end]-0.5*abs(pitch[end-1]-pitch[end]))|| R_sample>=R[end] || R_sample<=R[1] || Z_sample>=Z[end] || Z_sample<=Z[1] || abs(p_sample)>0.9999
            return false
        else
            return true
        end
    end
end



