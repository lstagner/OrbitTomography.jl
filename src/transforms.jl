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

function epr2ps_transform(F_os_VEC::AbstractArray{Float64},oenergy,opitch,or,PS_orbs::Vector{GCEPRCoordinate},og_orbit_index::Array{Int,3},og_class::Array{Symbol,3}; topological_force=true, simple_trans = true, mislabelled=false, distributed=false)
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

function epr2ps_ThinPlateSpline(F_os_VEC::AbstractArray{Float64}, orbs, PS_orbs::Vector{GCEPRCoordinate}, og_orbit_index::Array{Int,3}, og_class::Array{Symbol,3}; verbose=true, distributed=false, k::Int=2, overlap=true, stiffness_factor::Float64=0.0)
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

function ps2epr(F_ps_Weights::AbstractArray{Float64},PS_Grid::PSGrid, og_orbs; bin_info = false, jac_info = false, matrix_input=false, distributed=false)
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