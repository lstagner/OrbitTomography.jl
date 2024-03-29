struct EPDensity{T<:AbstractMatrix,S<:AbstractVector}
    d::T
    detJ::T
    energy::S
    pitch::S
end

struct PSDist2EPR_info
    orbit_weight::Float64
    path_indices::Array{Int,2}
    path_jacobians::Vector{Float64}
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

    return epr2ps_covariance_splined(M,OS.S_inv, f, orbs, sigma;Js=Js,energy=energy,pitch=pitch,r=r,z=z,distributed=distributed, atol=atol, domain_check=domain_check, covariance=covariance, norms=norms,checkpoint=checkpoint, warmstart=warmstart,file=file, kwargs...)
end

"""
    epr2ps(F_os_VEC::Vector{Float64},og,PS_orbs::Vector{GCEPRCoordinate}; sharedArray::Bool=false, topological_force=true, mislabelled=false, distributed=false, rescale_factor=0.0)

Transforms an orbit-space distribution to particle-space by assuming the orbit-space distribution is piecewise-constant. Set topological_force=true to reduce error from  binning across orbit-class boundaries. Runs fastest with sharedArray, distributed = false (default settings). Additional binning information provided by mislabelled=true. The mean distribution value of the transformed distribution is set by rescale_factor (defaults to no rescaling).
"""
function epr2ps(F_os_VEC::Vector{Float64},og,PS_orbs::Vector{GCEPRCoordinate}; sharedArray::Bool=false, topological_force=true, mislabelled=false, distributed=false, rescale_factor=0.0)
    return epr2ps(F_os_VEC::Vector{Float64},og.energy,og.pitch,og.r,PS_orbs,og.orbit_index,og.class; topological_force=topological_force, mislabelled=mislabelled, distributed=distributed, rescale_factor=rescale_factor)
end

"""
    epr2ps(F_os_VEC::Vector{Float64},oenergy::AbstractVector{Float64},opitch::AbstractVector{Float64},or::AbstractVector{Float64},PS_orbs::Vector{GCEPRCoordinate},og_orbit_index::Array{Int,3},og_class::Array{Symbol,3}; sharedArray::Bool=false, topological_force::Bool=true, mislabelled::Bool=false, distributed::Bool=false, rescale_factor=0.0)

Transforms an orbit-space distribution to particle-space by assuming the orbit-space distribution is piecewise-constant. Set topological_force=true to reduce error from  binning across orbit-class boundaries. Runs fastest with sharedArray, distributed = false (default settings). Additional binning information provided by mislabelled=true. The mean distribution value of the transformed distribution is set by rescale_factor (defaults to no rescaling).
"""
function epr2ps(F_os_VEC::Vector{Float64},oenergy::AbstractVector{Float64},opitch::AbstractVector{Float64},or::AbstractVector{Float64},PS_orbs::Vector{GCEPRCoordinate},og_orbit_index::Array{Int,3},og_class::Array{Symbol,3}; sharedArray::Bool=false, topological_force::Bool=true, mislabelled::Bool=false, distributed::Bool=false, rescale_factor=0.0)
    og_class = class_char.(og_class)

    num_orbs = length(PS_orbs)
    mislabelled && (distributed=false)

    dE = oenergy[2]-oenergy[1]
    dp = opitch[2]-opitch[1]
    dr = or[2]-or[1]

    if distributed
        if !sharedArray
            F_ps_VEC = @showprogress @distributed (vcat) for iter in 1:num_orbs
                i = argmin(abs.(PS_orbs[iter].energy .- oenergy))
                j = argmin(abs.(PS_orbs[iter].pitch_m .- opitch))
                k = argmin(abs.(PS_orbs[iter].r_m .- or))
                ind = og_orbit_index[i,j,k]

                F_ps_i = 0.0
                if !topological_force
                    if ind != 0
                        F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
                    else 
                        F_ps_i = 0.0
                    end
                else 
                    if (PS_orbs[iter].class == og_class[i,j,k])
                        F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
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
                        types = Char[]

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

                        (PS_orbs[iter].class == 'i') && error("Shouldn't be any incomplete orbits.")

                        if length(matching_type_inds)==0 #No points in the surrounding grid have the same type -> revert to nearest binning
                            if ind != 0
                                F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
                            else 
                                F_ps_i = 0.0
                            end
                        else
                            best_match_ind = argmin(distances)

                            indFIXED = og_orbit_index[locations[best_match_ind][1],locations[best_match_ind][2],locations[best_match_ind][3]]
                            F_ps_i = F_os_VEC[indFIXED]*PS_orbs[iter].jacdet
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

                F_ps_i
            end
        else 
            F_ps_VEC0 = SharedVector{Float64}(num_orbs)

            @sync @distributed for i in 1:num_orbs
                i = argmin(abs.(PS_orbs[i].energy .- oenergy))
                j = argmin(abs.(PS_orbs[i].pitch_m .- opitch))
                k = argmin(abs.(PS_orbs[i].r_m .- or))
                ind = og_orbit_index[i,j,k]

                F_ps_i = 0.0
                if !topological_force
                    if ind != 0
                        F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
                    else 
                        F_ps_i = 0.0
                    end
                else 
                    if (PS_orbs[iter].class == og_class[i,j,k])
                        F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
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
                        types = Char[]

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

                        (PS_orbs[iter].class == 'i') && error("Shouldn't be any incomplete orbits.")

                        if length(matching_type_inds)==0 #No points in the surrounding grid have the same type -> revert to nearest binning
                            if ind != 0
                                F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
                            else 
                                F_ps_i = 0.0
                            end
                        else
                            best_match_ind = argmin(distances)

                            indFIXED = og_orbit_index[locations[best_match_ind][1],locations[best_match_ind][2],locations[best_match_ind][3]]
                            F_ps_i = F_os_VEC[indFIXED]*PS_orbs[iter].jacdet
                        end
                    end
                end

                #Checking Energy Bounds:
                if (PS_orbs[i].energy > (oenergy[end]+0.5*(oenergy[end]-oenergy[end-1]))) || (PS_orbs[i].energy < (oenergy[1]-0.5*(oenergy[2]-oenergy[1]))) || (PS_orbs[i].energy < 0.0)
                    F_ps_i = 0.0
                end
                #Checking Pitch Bounds:
                if opitch[end]>opitch[1]
                    if (PS_orbs[i].pitch_m > (opitch[end]+0.5*abs(opitch[end]-opitch[end-1]))) || (PS_orbs[i].pitch_m < (opitch[1]-0.5*abs(opitch[2]-opitch[1]))) || abs(PS_orbs[i].pitch_m) > 1.0
                        F_ps_i = 0.0
                    end
                else
                    if (PS_orbs[i].pitch_m < (opitch[end]-0.5*abs(opitch[end]-opitch[end-1]))) || (PS_orbs[i].pitch_m > (opitch[1]+0.5*abs(opitch[2]-opitch[1]))) || abs(PS_orbs[i].pitch_m) > 1.0
                        F_ps_i = 0.0
                    end
                end
                #Rely on wall for R-bounds

                F_ps_VEC0[i] = F_ps_i
            end

            F_ps_VEC = convert(Vector,F_ps_VEC0)
            @everywhere F_ps_VEC0=nothing
        end
    else
        F_ps_VEC = Vector{Float64}(undef, num_orbs)

        if mislabelled 
            mislabelled_PSOrb_Ints = Int[]
            corresponding_OGOrb_locations = []
            mislabelledPS_types = Char[]
            mislabelledOS_types = Char[]
            if topological_force
                mislabelled_fixed_PSOrb_Ints = Int[]
                fixed_PSdist_differences = Float64[]
                corresponding_fixed_OGOrb_locations = []
            end
        end

        @showprogress for iter=1:num_orbs
            i = argmin(abs.(PS_orbs[iter].energy .- oenergy))
            j = argmin(abs.(PS_orbs[iter].pitch_m .- opitch))
            k = argmin(abs.(PS_orbs[iter].r_m .- or))
            ind = og_orbit_index[i,j,k]

            F_ps_i = 0.0
            if !topological_force
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
                if (PS_orbs[iter].class == og_class[i,j,k]) #Nearest point is of same class, take its value straight away
                    F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
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
                    types = Char[]

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

                    (PS_orbs[iter].class == 'i') && error("Shouldn't be any incomplete orbits.")

                    if length(matching_type_inds)==0 #No points in the surrounding grid have the same type -> revert to nearest binning
                        if ind != 0
                            F_ps_i = F_os_VEC[ind]*PS_orbs[iter].jacdet
                        else 
                            F_ps_i = 0.0
                        end

                        if mislabelled
                            push!(mislabelled_PSOrb_Ints,iter)
                            push!(corresponding_OGOrb_locations,[i,j,k])
                            push!(mislabelledPS_types,PS_orbs[iter].class)
                            push!(mislabelledOS_types,og_class[i,j,k])
                        end
                    else #Nearest point isn't of same class, but at least one point in the surrounding grid is. Value of closest same-class point taken
                        best_match_ind = argmin(distances)

                        indFIXED = og_orbit_index[locations[best_match_ind][1],locations[best_match_ind][2],locations[best_match_ind][3]]
                        F_ps_i = F_os_VEC[indFIXED]*PS_orbs[iter].jacdet

                        if mislabelled 
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

            F_ps_VEC[iter] = F_ps_i
        end
    end

    if rescale_factor == 1.0
        F_ps_VEC =  (F_ps_VEC ./ sum(abs.(F_ps_VEC))) .* length(F_ps_VEC)
    elseif rescale_factor != 0.0
        F_ps_VEC =  (F_ps_VEC ./ sum(abs.(F_ps_VEC))) .* rescale_factor
    end

    if topological_force && mislabelled
        corresponding_OGOrb_locations_matrix = zeros(Int,length(corresponding_OGOrb_locations),3)
        corresponding_OGOrb_fixed_locations_matrix = zeros(Int,length(corresponding_fixed_OGOrb_locations),3)
        OGclasses = Vector{Char}()
        PSclasses = Vector{Char}()

        for (io,o) in enumerate(corresponding_OGOrb_locations)
            corresponding_OGOrb_locations_matrix[io,:]=o
            push!(OGclasses,mislabelledOS_types[io])
            push!(PSclasses,mislabelledPS_types[io])
        end
        for (io,o) in enumerate(corresponding_fixed_OGOrb_locations)
            corresponding_OGOrb_fixed_locations_matrix[io,:]=o
        end

        return vec(F_ps_VEC),mislabelled_PSOrb_Ints,mislabelled_fixed_PSOrb_Ints,fixed_PSdist_differences,corresponding_OGOrb_locations_matrix,corresponding_OGOrb_fixed_locations_matrix,PSclasses,OGclasses
    elseif mislabelled
        corresponding_OGOrb_locations_matrix = zeros(Int,length(corresponding_OGOrb_locations),3)
        OGclasses = Vector{Char}()
        PSclasses = Vector{Char}()

        for (io,o) in enumerate(corresponding_OGOrb_locations)
            corresponding_OGOrb_locations_matrix[io,:]=o
            push!(OGclasses,mislabelledOS_types[io])
            push!(PSclasses,mislabelledPS_types[io])
        end
        return vec(F_ps_VEC),mislabelled_PSOrb_Ints,corresponding_OGOrb_locations_matrix,PSclasses,OGclasses
    else
        return vec(F_ps_VEC)
    end
end

"""
    epr2ps_splined(F_os_VEC::Vector{Float64}, orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, PS_orbs::Vector{GCEPRCoordinate}, norms::Vector{Float64}; single_E::Bool=false, verbose=true, distributed=false, k::Int=2, overlap=true, stiffness_factor::Float64=0.0, rescale_factor= 0.0, kwargs...)

Transforms an orbit-space distribution to particle-space using polyharmonic splines. This is very slow for large orbit-space grids - single_E=true calls epr2ps_splined_singleE instead. 
The mean distribution value of the transformed distribution is set by rescale_factor (defaults to no rescaling). k is the degree of the polyharmonic basis function. norms provides relative scaling of the distance between grid points in each of the three dimensions in orbit-space. Defaults to identical scaling for each dimension. 
"""
function epr2ps_splined(F_os_VEC::Vector{Float64}, og, orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, PS_orbs::Vector{GCEPRCoordinate}; verbose=true, distributed=false, k::Int=2, overlap=true, stiffness_factor::Float64=0.0, rescale_factor= 0.0, kwargs...) 
    if length(og.energy)==1
        single_E = true
        norms = [og.pitch[2]-og.pitch[1],og.r[2]-og.r[1]]
    else
        single_E = false
        norms = [og.energy[2]-og.energy[1],og.pitch[2]-og.pitch[1],og.r[2]-og.r[1]]
    end

    return epr2ps_splined(F_os_VEC, orbs, PS_orbs, norms; single_E=single_E, verbose=verbose, distributed=distributed, k=k, overlap=overlap, stiffness_factor=stiffness_factor, rescale_factor=rescale_factor, kwargs...) 
end

"""
    epr2ps_splined(F_os_VEC::Vector{Float64}, orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, PS_orbs::Vector{GCEPRCoordinate}, norms::Vector{Float64}; single_E::Bool=false, verbose=true, distributed=false, k::Int=2, overlap=true, stiffness_factor::Float64=0.0, rescale_factor= 0.0, kwargs...)

Transforms an orbit-space distribution to particle-space using polyharmonic splines. This is very slow for large orbit-space grids - single_E=true calls epr2ps_splined_singleE instead. 
The mean distribution value of the transformed distribution is set by rescale_factor (defaults to no rescaling). k is the degree of the polyharmonic basis function. norms provides relative scaling of the distance between grid points in each of the three dimensions in orbit-space. Defaults to identical scaling for each dimension. 
"""
function epr2ps_splined(F_os_VEC::Vector{Float64}, orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, PS_orbs::Vector{GCEPRCoordinate}, norms::Vector{Float64}; single_E::Bool=false, verbose=true, distributed=false, k::Int=2, overlap=true, stiffness_factor::Float64=0.0, rescale_factor= 0.0, kwargs...) 
    single_E && (return epr2ps_splined_singleE(F_os_VEC, orbs, PS_orbs, norms; verbose=verbose, distributed=distributed, k=k, overlap=overlap, stiffness_factor=stiffness_factor, rescale_factor=rescale_factor, kwargs...))
    ctr_passing_spline, trapped_spline, co_passing_spline = class_splines(orbs,F_os_VEC,k; stiffness_factor=stiffness_factor, single_E=false, norms=norms, kwargs...)

    verbose && print("Evaluating particle-space distribution values.\n")
    if distributed
        PS_dist = @showprogress @distributed (vcat) for i in PS_orbs
            if i.class == 'c'
                val = i.jacdet*ctr_passing_spline.(i.energy,i.pitch_m,i.r_m)
            elseif i.class == 'p'
                val = i.jacdet*co_passing_spline.(i.energy,i.pitch_m,i.r_m)
            else    
                val = i.jacdet*trapped_spline.(i.energy,i.pitch_m,i.r_m)
            end

            val
        end
    else
        PS_dist = Vector{Float64}(undef,length(PS_orbs))
        @showprogress for (io,i) in enumerate(PS_orbs)
            if i.class == 'c'
                val = i.jacdet*ctr_passing_spline.(i.energy,i.pitch_m,i.r_m)
            elseif i.class == 'p'
                val = i.jacdet*co_passing_spline.(i.energy,i.pitch_m,i.r_m)
            else    
                val = i.jacdet*trapped_spline.(i.energy,i.pitch_m,i.r_m)
            end

            PS_dist[io] = val
        end
    end

    if rescale_factor == 1.0
        PS_dist =  (PS_dist ./ sum(abs.(PS_dist))) .* length(PS_dist)
    elseif rescale_factor != 0.0
        PS_dist =  (PS_dist ./ sum(abs.(PS_dist))) .* rescale_factor
    end

    return vec(PS_dist)
end

"""
    epr2ps_splined_singleE(F_os_VEC::Vector{Float64}, orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, PS_orbs::Vector{GCEPRCoordinate}, norms::Vector{Float64}; verbose=true, distributed=false, k::Int=2, overlap=true, stiffness_factor::Float64=0.0, rescale_factor= 0.0, kwargs...)

Transforms an orbit-space distribution to particle-space using polyharmonic splines. One set of splines is used for each energy value. Use identical energy values in the particle-space grid being transformed to, and run ps_cleaner and ps_energyfix upon completion. 
The mean distribution value of the transformed distribution is set by rescale_factor (defaults to no rescaling). k is the degree of the polyharmonic basis function. norms provides relative scaling of the distance between grid points in each of the three dimensions in orbit-space. Defaults to identical scaling for each dimension. 
"""
function epr2ps_splined_singleE(F_os_VEC::Vector{Float64}, orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, PS_orbs::Vector{GCEPRCoordinate}, norms::Vector{Float64}; verbose=true, distributed=false, k::Int=2, overlap=true, stiffness_factor::Float64=0.0, rescale_factor= 0.0, kwargs...) 
    ctr_passing_spline, trapped_spline, co_passing_spline = class_splines(orbs,F_os_VEC,k; stiffness_factor=stiffness_factor, single_E=true, norms=norms, kwargs...)

    verbose && print("Evaluating particle-space distribution values.\n")
    if distributed
        PS_dist = @showprogress @distributed (vcat) for i in PS_orbs
            if i.class == 'c'
                val = i.jacdet*ctr_passing_spline.(i.pitch_m,i.r_m)
            elseif i.class == 'p'
                val = i.jacdet*co_passing_spline.(i.pitch_m,i.r_m)
            else    
                val = i.jacdet*trapped_spline.(i.pitch_m,i.r_m)
            end

            val
        end
    else
        PS_dist = Vector{Float64}(undef,length(PS_orbs))
        @showprogress for (io,i) in enumerate(PS_orbs)
            if i.class == 'c'
                val = i.jacdet*ctr_passing_spline.(i.pitch_m,i.r_m)
            elseif i.class == 'p'
                val = i.jacdet*co_passing_spline.(i.pitch_m,i.r_m)
            else    
                val = i.jacdet*trapped_spline.(i.pitch_m,i.r_m)
            end

            PS_dist[io] = val
        end
    end

    if rescale_factor == 1.0
        PS_dist =  (PS_dist ./ sum(abs.(PS_dist))) .* length(PS_dist)
    elseif rescale_factor != 0.0
        PS_dist =  (PS_dist ./ sum(abs.(PS_dist))) .* rescale_factor
    end

    return vec(PS_dist)
end

"""
    class_splines(orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, F_os_VEC::Vector{Float64}, k::Int; norms::Vector{Float64}=Float64[], single_E::Bool=false, filename_prefactor::String = "", read_save_splines::Bool = false, read_save_centers::Bool = false, spline_read_write_dir::String = "", center_read_write_dir::String = "", stiffness_factor::Float64=0.0, verbose::Bool=true)

Generates polyharmonic splines for each set of arbits after first calling orbsort. Can read save or read the splines straight from file using filename_prefactor, read_save_splines=true, and spline_read_write_dir.
"""
function class_splines(orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, F_os_VEC::Vector{Float64}, k::Int; norms::Vector{Float64}=Float64[], single_E::Bool=false, filename_prefactor::String = "", read_save_splines::Bool = false, read_save_centers::Bool = false, spline_read_write_dir::String = "", center_read_write_dir::String = "", stiffness_factor::Float64=0.0, verbose::Bool=true)  
    single_E && (filename_prefactor = string(filename_prefactor,"SingleE"))
    if isempty(norms)
        spline_filename = string(filename_prefactor,"orb_polysplines.jld2")
    else
        spline_filename = string(filename_prefactor,"orb_NORMpolysplines.jld2")
    end

    !isempty(spline_read_write_dir) && cd(spline_read_write_dir)
    if read_save_splines && isfile(spline_filename)
        verbose && print("Reading splines from file.\n")
        @load spline_filename ctr_passing_spline trapped_spline co_passing_spline
    else
        if !read_save_centers
            verbose && print("Sorting orbits into types.\n")
            ctr_passing_points,trapped_points,co_passing_points,ctr_passing_inds,trapped_inds,co_passing_inds = orbsort(orbs,single_E)

            verbose && print("Calculating splines (RAM intensive).\n")

            if isempty(norms)
                ctr_passing_spline = PolyharmonicSplineInv(k,ctr_passing_points,F_os_VEC[ctr_passing_inds];s=stiffness_factor)
                trapped_spline = PolyharmonicSplineInv(k,trapped_points,F_os_VEC[trapped_inds];s=stiffness_factor)
                co_passing_spline = PolyharmonicSplineInv(k,co_passing_points,F_os_VEC[co_passing_inds];s=stiffness_factor)
            else 
                ctr_passing_spline = PolyharmonicSplineInvNorm(k,ctr_passing_points,F_os_VEC[ctr_passing_inds],norms;s=stiffness_factor)
                trapped_spline = PolyharmonicSplineInvNorm(k,trapped_points,F_os_VEC[trapped_inds],norms;s=stiffness_factor)
                co_passing_spline = PolyharmonicSplineInvNorm(k,co_passing_points,F_os_VEC[co_passing_inds],norms;s=stiffness_factor)
            end
        else 
            !isempty(center_read_write_dir) && cd(center_read_write_dir)
            center_filename = string(filename_prefactor,"sorted_orbs.jld2") #read_orbs

            if isfile(center_filename)
                verbose && print("Reading sorted orbs from file.\n")
                @load center_filename ctr_passing_points trapped_points co_passing_points ctr_passing_inds trapped_inds co_passing_inds
            else
                verbose && print("Sorting orbits into types & printing to file.\n")

                ctr_passing_points,trapped_points,co_passing_points,ctr_passing_inds,trapped_inds,co_passing_inds = orbsort(orbs,single_E)
                @save center_filename ctr_passing_points trapped_points co_passing_points ctr_passing_inds trapped_inds co_passing_inds
            end

            verbose && print("Calculating splines (RAM intensive).\n")
            if isempty(norms)
                ctr_passing_spline = PolyharmonicSplineInv(k,ctr_passing_points,F_os_VEC[ctr_passing_inds];s=stiffness_factor)
                trapped_spline = PolyharmonicSplineInv(k,trapped_points,F_os_VEC[trapped_inds];s=stiffness_factor)
                co_passing_spline = PolyharmonicSplineInv(k,co_passing_points,F_os_VEC[co_passing_inds];s=stiffness_factor)
            else 
                ctr_passing_spline = PolyharmonicSplineInvNorm(k,ctr_passing_points,F_os_VEC[ctr_passing_inds],norms;s=stiffness_factor)
                trapped_spline = PolyharmonicSplineInvNorm(k,trapped_points,F_os_VEC[trapped_inds],norms;s=stiffness_factor)
                co_passing_spline = PolyharmonicSplineInvNorm(k,co_passing_points,F_os_VEC[co_passing_inds],norms;s=stiffness_factor)
            end
        end

        if read_save_splines
            !isempty(spline_read_write_dir) && cd(spline_read_write_dir)
            verbose && print("Printing splines to file.\n")
            @save spline_filename ctr_passing_spline trapped_spline co_passing_spline
        end
    end

    return ctr_passing_spline, trapped_spline, co_passing_spline
end

"""
    ps_polyharmonic_spline(PS_orbs::Vector{GCEPRCoordinate}, F_ps_Weights::Vector{Float64};  fpsorbs_2_matrix=x->psorbs_2_matrix(x), k::Int=2, verbose::Bool=true, read_save_prefactor::String = "", read_save_centers::Bool = false, read_save_spline::Bool = false, stiffness_factor::Float64=0.0)

Generates polyharmonic spline for a vector of GCEPRCoordinates and their corresponding distribution values. Can read save or read the spline straight from file using read_save_prefactor and read_save_spline=true.
"""
function ps_polyharmonic_spline(PS_orbs::Vector{GCEPRCoordinate}, F_ps_Weights::Vector{Float64};  fpsorbs_2_matrix=x->psorbs_2_matrix(x), k::Int=2, verbose::Bool=true, read_save_prefactor::String = "", read_save_centers::Bool = false, read_save_spline::Bool = false, stiffness_factor::Float64=0.0)

    spline_filename = string(read_save_prefactor,"ps_polyspline.jld2")

    if read_save_spline && isfile(spline_filename)
        verbose && print("Reading particle-space spline from file.\n")
        @load spline_filename spline
    else
        psorb_matrix_filename = string(read_save_prefactor,"psorb_centers.jld2")

        if read_save_centers && isfile(psorb_matrix_filename)
            verbose && print("Reading spline centers from file.\n")
            @load psorb_matrix_filename points
        else
            verbose && print("Calculating spline centers.\n")

            points = fpsorbs_2_matrix(PS_orbs)

            if read_save_centers
                verbose && print("Printing spline centers to file.\n")
                @save psorb_matrix_filename points
            end
        end

        verbose && print("Generating particle-space spline (RAM intensive).\n")
        spline = PolyharmonicSpline(k,points,F_ps_Weights;s=stiffness_factor)

        if read_save_spline
            verbose && print("Printing spline to file.\n")
            @save spline_filename spline
        end
    end

    return spline
end

"""
    orbsort(orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit{Float64}},Vector{Orbit}},single_E::Bool)

Sorts vector of orbits into three sets of classes, and converts their coordinates into a matrix to generate polyharmonic splines. Three sets of classes are 1: counter passing, 2: co-passing, potato and stagnation, and 3: trapped, potato and stagnation. 
"""
function orbsort(orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit{Float64}},Vector{Orbit}},single_E::Bool)  
    single_E && (return orbsort_singleE(orbs)) 

    ctr_passing_points = SVector{3, Float64}[]
    ctr_passing_inds = Int[]
    trapped_points = SVector{3, Float64}[]
    trapped_inds = Int[]
    co_passing_points = SVector{3, Float64}[]
    co_passing_inds = Int[]

    ctr_passing_num = 0
    trapped_num = 0
    co_passing_num = 0

    @showprogress for (io,i) in enumerate(orbs)
        if i.class == :ctr_passing 
            push!(ctr_passing_points,SVector{3}(i.coordinate.energy,i.coordinate.pitch,i.coordinate.r)) #is this efficient?
            push!(ctr_passing_inds,io)
            ctr_passing_num += 1
        end
        if (i.class == :trapped || i.class == :stagnation || i.class == :potato) 
            push!(trapped_points,SVector{3}(i.coordinate.energy,i.coordinate.pitch,i.coordinate.r))
            push!(trapped_inds,io)
            trapped_num += 1
        end
        if (i.class == :co_passing || i.class == :stagnation || i.class == :potato) 
            push!(co_passing_points,SVector{3}(i.coordinate.energy,i.coordinate.pitch,i.coordinate.r))
            push!(co_passing_inds,io)
            co_passing_num += 1
        end
        
    end

    ctr_passing_points = copy(reshape(reinterpret(Float64, ctr_passing_points), (3,ctr_passing_num)))
    trapped_points =  copy(reshape(reinterpret(Float64, trapped_points), (3,trapped_num)))
    co_passing_points =  copy(reshape(reinterpret(Float64, co_passing_points), (3,co_passing_num)))

    return ctr_passing_points,trapped_points,co_passing_points,ctr_passing_inds,trapped_inds,co_passing_inds
end

"""
    orbsort_v2(orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit{Float64}},Vector{Orbit}},single_E::Bool)

Alternate version of orbsort, where the three sets of classes are 1: counter passing and stagnation (pitch < 0), 2: co-passing, potato and stagnation (pitch > 0), and 3: trapped, potato and stagnation (pitch > 0). Introduces more error than orbsort.
"""
function orbsort_v2(orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit{Float64}},Vector{Orbit}},single_E::Bool)  
    single_E && (return orbsort_singleE(orbs)) 

    ctr_passing_points = SVector{3, Float64}[]
    ctr_passing_inds = Int[]
    trapped_points = SVector{3, Float64}[]
    trapped_inds = Int[]
    co_passing_points = SVector{3, Float64}[]
    co_passing_inds = Int[]

    ctr_passing_num = 0
    trapped_num = 0
    co_passing_num = 0

    @showprogress for (io,i) in enumerate(orbs)
        if (i.class == :ctr_passing || (i.coordinate.pitch<0.0 && i.class == :stagnation))
            push!(ctr_passing_points,SVector{3}(i.coordinate.energy,i.coordinate.pitch,i.coordinate.r)) #is this efficient?
            push!(ctr_passing_inds,io)
            ctr_passing_num += 1
        end
        if (i.class == :trapped || (i.class == :stagnation && i.coordinate.pitch>0.0) || i.class == :potato)
            push!(trapped_points,SVector{3}(i.coordinate.energy,i.coordinate.pitch,i.coordinate.r))
            push!(trapped_inds,io)
            trapped_num += 1
        end
        if (i.class == :co_passing || (i.class == :stagnation && i.coordinate.pitch>0.0) || i.class == :potato)
            push!(co_passing_points,SVector{3}(i.coordinate.energy,i.coordinate.pitch,i.coordinate.r))
            push!(co_passing_inds,io)
            co_passing_num += 1
        end
        
    end

    ctr_passing_points = copy(reshape(reinterpret(Float64, ctr_passing_points), (3,ctr_passing_num)))
    trapped_points =  copy(reshape(reinterpret(Float64, trapped_points), (3,trapped_num)))
    co_passing_points =  copy(reshape(reinterpret(Float64, co_passing_points), (3,co_passing_num)))

    return ctr_passing_points,trapped_points,co_passing_points,ctr_passing_inds,trapped_inds,co_passing_inds
end

"""
    orbsort_singleE(orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit{Float64}},Vector{Orbit}})

Sorts vector of orbits into three sets of classes, and converts their coordinates (minus energy) into a matrix to generate polyharmonic splines. Three sets of classes are 1: counter passing, 2: co-passing, potato and stagnation, and 3: trapped, potato and stagnation. 
"""
function orbsort_singleE(orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit{Float64}},Vector{Orbit}})  
    ctr_passing_points = SVector{2, Float64}[]
    ctr_passing_inds = Int[]
    trapped_points = SVector{2, Float64}[]
    trapped_inds = Int[]
    co_passing_points = SVector{2, Float64}[]
    co_passing_inds = Int[]

    ctr_passing_num = 0
    trapped_num = 0
    co_passing_num = 0

    @showprogress for (io,i) in enumerate(orbs)
        if i.class == :ctr_passing
            push!(ctr_passing_points,SVector{2}(i.coordinate.pitch,i.coordinate.r)) #is this efficient?
            push!(ctr_passing_inds,io)
            ctr_passing_num += 1
        end
        if (i.class == :trapped || i.class == :stagnation || i.class == :potato)
            push!(trapped_points,SVector{2}(i.coordinate.pitch,i.coordinate.r))
            push!(trapped_inds,io)
            trapped_num += 1
        end
        if (i.class == :co_passing || i.class == :stagnation || i.class == :potato)
            push!(co_passing_points,SVector{2}(i.coordinate.pitch,i.coordinate.r))
            push!(co_passing_inds,io)
            co_passing_num += 1
        end
        
    end

    ctr_passing_points = copy(reshape(reinterpret(Float64, ctr_passing_points), (2,ctr_passing_num)))
    trapped_points =  copy(reshape(reinterpret(Float64, trapped_points), (2,trapped_num)))
    co_passing_points =  copy(reshape(reinterpret(Float64, co_passing_points), (2,co_passing_num)))

    return ctr_passing_points,trapped_points,co_passing_points,ctr_passing_inds,trapped_inds,co_passing_inds
end

"""
    orbsort_singleE_v2(orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit{Float64}},Vector{Orbit}})

Alternate version of orbsort_singleE, where the three sets of classes are 1: counter passing and stagnation (pitch < 0), 2: co-passing, potato and stagnation (pitch > 0), and 3: trapped, potato and stagnation (pitch > 0). Introduces more error than orbsort.
"""
function orbsort_singleE_v2(orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit{Float64}},Vector{Orbit}})  
    ctr_passing_points = SVector{2, Float64}[]
    ctr_passing_inds = Int[]
    trapped_points = SVector{2, Float64}[]
    trapped_inds = Int[]
    co_passing_points = SVector{2, Float64}[]
    co_passing_inds = Int[]

    ctr_passing_num = 0
    trapped_num = 0
    co_passing_num = 0

    @showprogress for (io,i) in enumerate(orbs)
        if (i.class == :ctr_passing || (i.coordinate.pitch<0.0 && i.class == :stagnation))
            push!(ctr_passing_points,SVector{2}(i.coordinate.pitch,i.coordinate.r)) #is this efficient?
            push!(ctr_passing_inds,io)
            ctr_passing_num += 1
        end
        if (i.class == :trapped || (i.class == :stagnation && i.coordinate.pitch>0.0) || i.class == :potato)
            push!(trapped_points,SVector{2}(i.coordinate.pitch,i.coordinate.r))
            push!(trapped_inds,io)
            trapped_num += 1
        end
        if (i.class == :co_passing || (i.class == :stagnation && i.coordinate.pitch>0.0) || i.class == :potato)
            push!(co_passing_points,SVector{2}(i.coordinate.pitch,i.coordinate.r))
            push!(co_passing_inds,io)
            co_passing_num += 1
        end
        
    end

    ctr_passing_points = copy(reshape(reinterpret(Float64, ctr_passing_points), (2,ctr_passing_num)))
    trapped_points =  copy(reshape(reinterpret(Float64, trapped_points), (2,trapped_num)))
    co_passing_points =  copy(reshape(reinterpret(Float64, co_passing_points), (2,co_passing_num)))

    return ctr_passing_points,trapped_points,co_passing_points,ctr_passing_inds,trapped_inds,co_passing_inds
end

"""
    epr2ps_covariance_splined(M::AbstractEquilibrium, S_inv::AbstractArray{Float64}, F_os_VEC::Vector, orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, psgrid::PSGrid, sigma;
                            distributed=false, atol=1e-3, domain_check= (xx,yy) -> true,
                            covariance=:local, norms=S3(1.0,1.0,1.0),
                            checkpoint=true, warmstart=false,file="eprz_progress.jld2",  rescale_factor= 0.0, kwargs...) 

Transforms an orbit-space distribution to particle-space using the covariance spline with PSGrid as an input. See epr2ps_covariance_splined below for full description.
"""
function epr2ps_covariance_splined(M::AbstractEquilibrium, S_inv::AbstractArray{Float64}, F_os_VEC::Vector, orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, psgrid::PSGrid, sigma;
                            distributed=false, atol=1e-3, domain_check= (xx,yy) -> true,
                            covariance=:local, norms=S3(1.0,1.0,1.0),
                            checkpoint=true, warmstart=false,file="eprz_progress.jld2",  rescale_factor= 0.0, kwargs...) 

    Js = Array{Vector{Float64}}(undef, length(orbs))
    for (io,o) in enumerate(orbs) 
        Js[io] = o.path.jacdets
    end

    eprz_density = epr2ps_covariance_splined(M, S_inv, F_os_VEC, orbs, sigma; Js=Js, energy=psgrid.energy, pitch=psgrid.pitch, r=psgrid.r, z=psgrid.z,distributed=distributed, atol=atol, domain_check=domain_check, covariance=covariance, norms=norms, checkpoint=checkpoint, warmstart=warmstart, file=file, kwargs...)
    F_ps_VEC = ps_MatrixToVector(eprz_density.d, psgrid)

    if rescale_factor == 1.0
        F_ps_VEC =  (F_ps_VEC ./ sum(abs.(F_ps_VEC))) .* length(F_ps_VEC)
    elseif rescale_factor != 0.0
        F_ps_VEC =  (F_ps_VEC ./ sum(abs.(F_ps_VEC))) .* rescale_factor
    end

    return F_ps_VEC
end

"""
    epr2ps_covariance_splined(M::AbstractEquilibrium, S_inv::AbstractArray{Float64}, F_os_VEC::Vector, orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, sigma;
                            Js=Vector{Vector{Float64}}[],
                            energy=range(1.0,80.0,length=25),
                            pitch=range(-0.99,0.99,length=25),
                            r = range(limits(M)[1]...,length=25),
                            z = range(limits(M)[2]...,length=25),
                            distributed=false, atol=1e-3, domain_check= (xx,yy) -> true,
                            covariance=:local, norms=S3(1.0,1.0,1.0),
                            checkpoint=true, warmstart=false,file="eprz_progress.jld2", kwargs...)

Transforms an orbit-space distribution to particle-space using the covariance spline. This involves calculating a particle-space point’s orbit, and its ‘similarity’ with all other orbits in the orbit-grid; the similarity between two orbits is a measure of how alike the orbit paths are in particle-space, and it is found by integrating both orbit paths in particle-space while passing the distance in 4D {E, p, R, Z} space between the two particles through a gaussian kernel (see function get_covariance, line 310 in covariance.jl).
Requires input of characteristic distances in E,p,R,Z space for the gaussian kernel. Input as 4-component vector sigma.
"""
function epr2ps_covariance_splined(M::AbstractEquilibrium, S_inv::AbstractArray{Float64}, F_os_VEC::Vector, orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, sigma;
                            Js=Vector{Vector{Float64}}[],
                            energy=range(1.0,80.0,length=25),
                            pitch=range(-0.99,0.99,length=25),
                            r = range(limits(M)[1]...,length=25),
                            z = range(limits(M)[2]...,length=25),
                            distributed=false, atol=1e-3, domain_check= (xx,yy) -> true,
                            covariance=:local, norms=S3(1.0,1.0,1.0),
                            checkpoint=true, warmstart=false,file="eprz_progress.jld2", kwargs...) 

    if isempty(Js) && (!isempty(orbs[1].path.jacdets))
        Js = Array{Vector{Float64}}(undef, length(orbs))
        for (io,o) in enumerate(orbs) 
            Js[io] = o.path.jacdets
        end
    end

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

    @showprogress for I in inds
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
            lJs = Array{Vector{Float64}}(undef, length(lorbs))
            for (io,o) in enumerate(lorbs)
                try
                    lJs[io] = get_jacobian(M, o)
                catch
                    lJs[io] = zeros(2)
                end
            end
        end

        if covariance == :local
            Si = get_covariance_matrix(M, lorbs, orbs, sigma, Js_1=lJs, Js_2=Js,
                                       distributed=distributed, atol=atol)
        else
            Si = get_global_covariance_matrix(lorbs, orbs, sigma, norms=norms)
        end

        f_ep = reshape(max.(Si*(S_inv*F_os_VEC),0.0),nenergy,npitch)
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

"""
    ps_cleaner(F_ps_Weights::Vector{Float64},  psgrid::PSGrid; zbounds = [0.0,0.0], PS_orbs=nothing, class_specific::Bool=false, threshold::Float64=0.0, num0::Int=0, rescale_factor::Number = 1.0, E_range::Int=0, range::Int=1)

Run on a distribution after transforming from orbit-space to particle-space.
Smooths over large transformed values by assigning them the average of their nearest neighbours in particle-space. This is either done automatically to the largest n values by setting num0=n, or automatically to all values above threshold, or manually (default setting).
Sets negative distibution values and NaNs to 0 - these are caused by the polyharmonic spline transform. Removes the influence of orbits with Z_m far off the main axis, including orbits that appear in the divertor region. The mean distribution value is set by rescale_factor (defaults to no rescaling).
"""
function ps_cleaner(F_ps_Weights::Vector{Float64},  psgrid::PSGrid; zbounds = [0.0,0.0], PS_orbs=nothing, class_specific::Bool=false, threshold::Float64=0.0, num0::Int=0, rescale_factor::Number = 1.0, E_range::Int=0, range::Int=1) where {T}
    F_ps_Weights = [isnan(i) ? 0.0 : i for i in F_ps_Weights] #Removing Nans
    F_ps_Weights = [i < 0.0 ? 0.0 : i for i in F_ps_Weights]  #Removing negative values
    if zbounds!=[0.0,0.0] #Removing disconnected regions
        isnothing(PS_orbs) && error("Input ps_orbs to implement zbounds.")
        F_ps_Weights = [(PS_orbs[i].z_m < minimum(zbounds) || PS_orbs[i].z_m > maximum(zbounds)) ? 0.0 : F_ps_Weights[i] for i in 1:length(PS_orbs)]  
    end

    sorted = sort(F_ps_Weights,rev=true)
    if num0 != 0
        num = num0
        print("Largest $(num) entries will be cleaned.\n")
    elseif threshold == 0.0
        display(sorted)
        print("Enter number of outliers to clean:\n")
        sleep(1)
        num = readline()
        num = parse(Int64, num)

        print("Largest $(num) entries will be cleaned.\n")
    else
        threshold = threshold*mean(F_ps_Weights)
        num=0
        for i in 1:length(F_ps_Weights) 
            (sorted[i] < threshold) && break
            num=i
        end

        print("Largest $(num) entries will be cleaned.\n")
    end

    old_values = zeros(Float64,num)
    new_values = zeros(Float64,num)
    cleaned_args = []
    surrounding_classes = []
    surrounding_values = []
    uncleaned = copy(F_ps_Weights);

    F_ps_matrix = ps_VectorToMatrix(F_ps_Weights,psgrid);
    @showprogress for i in 1:num
        all_inds = CartesianIndices((length(psgrid.energy),length(psgrid.pitch),length(psgrid.r),length(psgrid.z)))
        args = argmax(F_ps_matrix)
        max_type = psgrid.class[args[1],args[2],args[3],args[4]]

        e1 = max(args[1]-E_range,1)
        p1 = max(args[2]-range,1)
        r1 = max(args[3]-range,1)
        z1 = max(args[4]-range,1)

        e2 = min(args[1]+E_range,length(psgrid.energy))
        p2 = min(args[2]+range,length(psgrid.pitch))
        r2 = min(args[3]+range,length(psgrid.r))
        z2 = min(args[4]+range,length(psgrid.z))

        #args[1]-E_range:args[1]+E_range,args[2]-range:args[2]+range,args[3]-range:args[3]+range,args[4]-range:args[4]+range
        #becomes e1:e2,p1:p2,r1:r2,z1:z2

        if class_specific 
            elligibles = filter(x -> (psgrid.class[x[1],x[2],x[3],x[4]]==max_type)&&(x!=args), all_inds[e1:e2,p1:p2,r1:r2,z1:z2])
        else
            elligibles = filter(x -> (psgrid.class[x[1],x[2],x[3],x[4]]!='i')&&(x!=args), all_inds[e1:e2,p1:p2,r1:r2,z1:z2])
        
        end
        new_val = mean(F_ps_matrix[elligibles]) 

        old_values[i] = maximum(F_ps_matrix)
        new_values[i] = new_val
        push!(cleaned_args, (argmax(F_ps_Weights),args))
        push!(surrounding_values, F_ps_matrix[e1:e2,p1:p2,r1:r2,z1:z2])
        push!(surrounding_classes, psgrid.class[e1:e2,p1:p2,r1:r2,z1:z2])

        F_ps_matrix[args[1],args[2],args[3],args[4]] = new_val
        F_ps_Weights[argmax(F_ps_Weights)] = new_val
    end

    F_ps_Weights = F_ps_Weights .* (rescale_factor*length(F_ps_Weights)/sum(F_ps_Weights))
    uncleaned = uncleaned .* (rescale_factor*length(uncleaned)/sum(uncleaned))

    return F_ps_Weights,old_values,new_values,surrounding_classes,surrounding_values,uncleaned
end

"""
    ps_energyfix(F_ps_Weights::Vector{Float64}, F_os_VEC::Vector{Float64}, og::Union{OrbitGrid{T},OrbitGrid},  psgrid::PSGrid; F_ps_Weights0::Vector{Float64}=Float64[], ps_Efix::Bool=false, info::Bool=false, rescale_factor::Number = 1.0) where {T}

Run on a distribution after transforming from orbit-space to particle-space energy-slice by energy-slice. Rescales the different energy slices according to the orbit-space energy distribution. Can also re-scale according to the energy distribution of a reference particle-space distribution. The mean distribution value is set by rescale_factor (defaults to no rescaling).
"""
function ps_energyfix(F_ps_Weights::Vector{Float64}, F_os_VEC::Vector{Float64}, og::Union{OrbitGrid{T},OrbitGrid},  psgrid::PSGrid; F_ps_Weights0::Vector{Float64}=Float64[], ps_Efix::Bool=false, info::Bool=false, rescale_factor::Number = 1.0) where {T}
    length(psgrid.energy) != length(og.energy) && throw(DimensionMismatch())
    scales = zeros(Float64,length(psgrid.energy))
    scalestest = zeros(Float64,length(psgrid.energy))

    for i in 1:length(psgrid.energy)
        orb_e_inds = filter(x->x!=0, og.orbit_index[i,:,:]);
        ps_e_inds = filter(x->x!=0, psgrid.point_index[i,:,:,:]);

        (ps_Efix && !isempty(F_ps_Weights0)) ? (scale_factor = mean(F_ps_Weights0[ps_e_inds])) : (scale_factor = mean(F_os_VEC[orb_e_inds]))

        F_ps_Weights[ps_e_inds] = F_ps_Weights[ps_e_inds] .* (scale_factor*length(ps_e_inds)/sum(F_ps_Weights[ps_e_inds]))

        scales[i]=scale_factor
        scalestest[i]=mean(F_ps_Weights[ps_e_inds])
    end
    F_ps_Weights = [isnan(x) ? 0.0 : x for x in F_ps_Weights]
    F_ps_Weights = F_ps_Weights .* (rescale_factor*length(F_ps_Weights)/sum(F_ps_Weights))

    info && (return F_ps_Weights,scales,scalestest)
    return F_ps_Weights
end

"""
    ps2epr(F_ps_Weights::AbstractArray{Float64},PS_Grid::PSGrid, og_orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}; bin_info = false, jac_info = false,  distributed=false, rescale_factor=0.0)

Transforms an EPRZ distribution to 3D orbit-space by assuming it's piecewise constant. Slightly slower and less accurate than ps2epr_bsplined. bin_info=true provides information on the number of bins each orbit in orbit-space spans in particle-space. jac_info=true outputs the average jacobian along each orbit-space orbit's path. Returns the orbit-space distribution in vector form.
"""
function ps2epr(F_ps_Weights::AbstractArray{Float64},PS_Grid::PSGrid, og_orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}; bin_info = false, jac_info = false,  distributed=false, rescale_factor=0.0) 
    length(size(F_ps_Weights))>1 ? (matrix_input=true) :  (matrix_input=false)

    if distributed 
        psenergy = PS_Grid.energy
        pspitch = PS_Grid.pitch
        psr = PS_Grid.r
        psz = PS_Grid.z
        pspoint_index = PS_Grid.point_index

        @eval @everywhere begin
            psenergy = $psenergy
            pspitch = $pspitch
            psr = $psr
            psz = $psz
            pspoint_index = $pspoint_index
        end

        F_os_VEC = @showprogress @distributed (vcat) for iter=1:length(og_orbs)
            orb = og_orbs[iter]
            integrated_weight = 0.0

            if bin_info
                path_inds = zeros(Int,(length(orb.path.r)-1),4)
            else    
                path_inds = zeros(Int,0,0)
            end

            if jac_info
                path_jacs = zeros(Float64,(length(orb.path.r)-1))
            else    
                path_jacs = zeros(Float64,0)
            end

            for path_iter = 1:(length(orb.path.r)-1)
                i = argmin(abs.(orb.path.energy[path_iter] .- psenergy))
                j = argmin(abs.(orb.path.pitch[path_iter] .- pspitch))
                k = argmin(abs.(orb.path.r[path_iter] .- psr))
                l = argmin(abs.(orb.path.z[path_iter] .-psz))
                ind = pspoint_index[i,j,k,l]

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
                if (orb.path.energy[path_iter] > (psenergy[end]+0.5*(psenergy[end]-psenergy[end-1]))) || (orb.path.energy[path_iter] < (psenergy[1]-0.5*(psenergy[2]-psenergy[1]))) || (orb.path.energy[path_iter] < 0.0)
                    F_os_i = 0.0
                end
                #Checking Pitch Bounds:
                if pspitch[end]>pspitch[1]
                    if (orb.path.pitch[path_iter] > (pspitch[end]+0.5*abs(pspitch[end]-pspitch[end-1]))) || (orb.path.pitch[path_iter] < (pspitch[1]-0.5*abs(pspitch[2]-pspitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
                        F_os_i = 0.0
                    end
                else
                    if (orb.path.pitch[path_iter] < (pspitch[end]-0.5*abs(pspitch[end]-pspitch[end-1]))) || (orb.path.pitch[path_iter] > (pspitch[1]+0.5*abs(pspitch[2]-pspitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
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
            F_os_VEC = Vector{PSDist2EPR_info}(undef,length(og_orbs))
        else
            F_os_VEC = Vector{Float64}(undef,length(og_orbs))
        end

        @showprogress for iter=1:length(og_orbs)
            orb = og_orbs[iter]
            integrated_weight = 0.0

            if bin_info
                path_inds = zeros(Int,(length(orb.path.r)-1),4)
            else    
                path_inds = zeros(Int,0,0)
            end

            if jac_info
                path_jacs = zeros(Float64,(length(orb.path.r)-1))
            else    
                path_jacs = zeros(Float64,0)
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

            (bin_info || jac_info) ? F_os_VEC[iter] = PSDist2EPR_info(integrated_weight,path_inds,path_jacs) : F_os_VEC[iter] = integrated_weight 
        end
    end

    if (bin_info || jac_info)
        F_os = Vector{Float64}(undef,length(og_orbs))

        for i=1:length(og_orbs)
            F_os[i] = F_os_VEC[i].orbit_weight
        end

        if rescale_factor == 1.0
            F_os =  (F_os ./ sum(abs.(F_os))) .* length(F_os)
        elseif rescale_factor != 0.0
            F_os =  (F_os ./ sum(abs.(F_os))) .* rescale_factor
        end

        return vec(F_os),F_os_VEC
    end

    if rescale_factor == 1.0
        F_os_VEC =  (F_os_VEC ./ sum(abs.(F_os_VEC))) .* length(F_os_VEC)
    elseif rescale_factor != 0.0
        F_os_VEC =  (F_os_VEC ./ sum(abs.(F_os_VEC))) .* rescale_factor
    end

    return vec(F_os_VEC)
end

"""
    ps2epr_polysplined(F_ps_Weights::Vector{Float64}, PS_orbs::Vector{GCEPRCoordinate}, og_orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}; distributed=false, rescale_factor=0.0, verbose=true, kwargs...)

Transforms an EPRZ distribution to 3D orbit-space, using polyharmonic splines. Fatally slow, use ps2epr_bsplined instead.
"""
function ps2epr_polysplined(F_ps_Weights::Vector{Float64}, PS_orbs::Vector{GCEPRCoordinate}, og_orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}; distributed=false, rescale_factor=0.0, verbose=true, kwargs...)  #vers=2 slightly faster for small batches, will confirm which better large scale on cluster
    num_psorbs = length(PS_orbs)

    spline = ps_polyharmonic_spline(PS_orbs, F_ps_Weights; verbose=verbose, kwargs...)

    verbose && print("Calculating orbit-space distribution.\n")

    if distributed
        os_dist = @showprogress @distributed (vcat) for i in og_orbs
            integrated_weight = 0.0

            for p in 1:length(i.path.r)
                F_os_i = i.path.jacdets[p]*spline(i.path.energy[p],i.path.pitch[p],i.path.r[p],i.path.z[p])

                if p==1
                    integrated_weight += F_os_i*(0.5*i.path.dt[1]+0.5*i.path.dt[end-1])/i.tau_p
                else
                    integrated_weight += F_os_i*(0.5*i.path.dt[p]+0.5*i.path.dt[p-1])/i.tau_p
                end
            end

            integrated_weight
        end
    else
        os_dist =  Vector{Float64}(undef,length(og_orbs))

        @showprogress for (io,i) in enumerate(og_orbs)
            integrated_weight = 0.0

            for p in 1:length(i.path.r)
                F_os_i = i.path.jacdets[p]*spline(i.path.energy[p],i.path.pitch[p],i.path.r[p],i.path.z[p])

                if p==1
                    integrated_weight += F_os_i*(0.5*i.path.dt[1]+0.5*i.path.dt[end-1])/i.tau_p
                else
                    integrated_weight += F_os_i*(0.5*i.path.dt[p]+0.5*i.path.dt[p-1])/i.tau_p
                end
            end

            os_dist[io] = integrated_weight
        end
    end

    if rescale_factor == 1.0
        os_dist =  (os_dist ./ sum(abs.(os_dist))) .* length(os_dist)
    elseif rescale_factor != 0.0
        os_dist =  (os_dist ./ sum(abs.(os_dist))) .* rescale_factor
    end
    return vec(os_dist)
end

"""
    ps2epr_bsplined(F_ps_Weights::Vector{Float64}, psgrid::PSGrid, og_orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, og::Union{OrbitGrid{T},OrbitGrid}; spline_type = BSpline(Cubic(Flat(OnCell()))), distributed::Bool=false, rescale_factor=0.0, use_slice_E::Bool=true, verbose=true, psgridEprefactor = "", gcvalid_check::Bool=false)

Transforms an EPRZ distribution to 3D orbit-space, using an ongrid bspline (see Interpolations.jl).
Fastest and most accurate ps2epr transform. distributed=false is faster. The mean distribution value is set by rescale_factor (defaults to no rescaling). use_slice_E creates a 3D spline for each energy value. This isn't necessary for all but the largest grids.
"""
function ps2epr_bsplined(F_ps_Weights::Vector{Float64}, psgrid::PSGrid, og_orbs::Union{Vector{Orbit{Float64, EPRCoordinate{Float64}}},Vector{Orbit},Vector{Orbit{Float64}}}, og::Union{OrbitGrid{T},OrbitGrid}; spline_type = BSpline(Cubic(Flat(OnCell()))), distributed::Bool=false, rescale_factor=0.0, use_slice_E::Bool=true, verbose=true, psgridEprefactor = "", gcvalid_check::Bool=false) where {T}
    F_ps_WeightsMatrix = ps_VectorToMatrix(F_ps_Weights,psgrid)

    if distributed
        @everywhere psenergy = psgrid.energy
        @everywhere pspitch = psgrid.pitch

        if use_slice_E
            length(og.energy)!=length(psgrid.energy) && throw(DimensionMismatch())
            orig_orbit_index = og.orbit_index

            weights_inds = @showprogress @distributed (hcat) for slice_E in 1:length(og.energy)
                F_ps_WeightsMatrixE = F_ps_WeightsMatrix[slice_E,:,:,:]
                og_orbsE,ogE = energy_slice(og,og_orbs;ind=slice_E)
                orbE_inds = filter(x->x!=0,orig_orbit_index[slice_E,:,:])
                if isfile(string(psgridEprefactor,"(E",slice_E,")PSGrid.jld2")) && isfile(string(psgridEprefactor,"(E",slice_E,")GCEPRCoords.jld2"))
                    psgridE = read_PSGrid(string(psgridEprefactor,"(E",slice_E,")PSGrid.jld2"))
                    psorbsE,gcvalid_checkE,vacuum,drift = read_GCEPRCoords(string(psgridEprefactor,"(E",slice_E,")GCEPRCoords.jld2"))
                else
                    psorbsE, psgridE, gcvalid_checkE = energy_slice(psgrid,psgrid,gcvalid_check,slice_E)
                end

                os_dist =  Array{Float64}(undef,2,length(og_orbsE))

                psgrid_pitch = range(psgridE.pitch[1],psgridE.pitch[end],length(psgridE.pitch))
                psgrid_r = range(psgridE.r[1],psgridE.r[end],length(psgridE.r))
                psgrid_z = range(psgridE.z[1],psgridE.z[end],length(psgridE.z))
                itp =  Interpolations.scale(interpolate(F_ps_WeightsMatrixE, spline_type),psgrid_pitch,psgrid_r,psgrid_z)
                sitp = extrapolate(itp,Flat())

                for iter=1:length(og_orbsE)
                    orb = og_orbsE[iter]
                    integrated_weight = 0.0
        
                    for path_iter = 1:(length(orb.path.r)-1)
                        F_os_i = orb.path.jacdets[path_iter]*sitp(orb.path.pitch[path_iter],orb.path.r[path_iter],orb.path.z[path_iter])


                        #Checking Energy Bounds:
                        if (orb.path.energy[path_iter] > (psenergy[end]+0.5*(psenergy[end]-psenergy[end-1]))) || (orb.path.energy[path_iter] < (psenergy[1]-0.5*(psenergy[2]-psenergy[1]))) || (orb.path.energy[path_iter] < 0.0)
                            F_os_i = 0.0
                        end
                        #Checking Pitch Bounds:
                        if pspitch[end]>pspitch[1]
                            if (orb.path.pitch[path_iter] > (pspitch[end]+0.5*abs(pspitch[end]-pspitch[end-1]))) || (orb.path.pitch[path_iter] < (pspitch[1]-0.5*abs(pspitch[2]-pspitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
                                F_os_i = 0.0
                            end
                        else
                            if (orb.path.pitch[path_iter] < (pspitch[end]-0.5*abs(pspitch[end]-pspitch[end-1]))) || (orb.path.pitch[path_iter] > (pspitch[1]+0.5*abs(pspitch[2]-pspitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
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

                    os_dist[:,iter] = [orbE_inds[iter],integrated_weight]
                end 

                os_dist
            end

            os_dist = zeros(length(og_orbs))
            @inbounds for i in 1:length(og_orbs)
                os_dist[Int(weights_inds[1,i])] = weights_inds[2,i]
            end

        else
            psgrid_energy = range(psgrid.energy[1],psgrid.energy[end],length(psgrid.energy)) 
            psgrid_pitch = range(psgrid.pitch[1],psgrid.pitch[end],length(psgrid.pitch))
            psgrid_r = range(psgrid.r[1],psgrid.r[end],length(psgrid.r))
            psgrid_z = range(psgrid.z[1],psgrid.z[end],length(psgrid.z))
            itp = Interpolations.scale(interpolate(F_ps_WeightsMatrix, spline_type),psgrid_energy,psgrid_pitch,psgrid_r,psgrid_z)
            sitp = extrapolate(itp,Flat())

            os_dist = @showprogress @distributed (vcat) for iter=1:length(og_orbs)
                orb = og_orbs[iter]
                integrated_weight = 0.0

                for path_iter = 1:(length(orb.path.r)-1)
                    F_os_i = orb.path.jacdets[path_iter]*sitp(orb.path.energy[path_iter],orb.path.pitch[path_iter],orb.path.r[path_iter],orb.path.z[path_iter])

                    #Checking Energy Bounds:
                    if (orb.path.energy[path_iter] > (psenergy[end]+0.5*(psenergy[end]-psenergy[end-1]))) || (orb.path.energy[path_iter] < (psenergy[1]-0.5*(psenergy[2]-psenergy[1]))) || (orb.path.energy[path_iter] < 0.0)
                        F_os_i = 0.0
                    end
                    #Checking Pitch Bounds:
                    if pspitch[end]>pspitch[1]
                        if (orb.path.pitch[path_iter] > (pspitch[end]+0.5*abs(pspitch[end]-pspitch[end-1]))) || (orb.path.pitch[path_iter] < (pspitch[1]-0.5*abs(pspitch[2]-pspitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
                            F_os_i = 0.0
                        end
                    else
                        if (orb.path.pitch[path_iter] < (pspitch[end]-0.5*abs(pspitch[end]-pspitch[end-1]))) || (orb.path.pitch[path_iter] > (pspitch[1]+0.5*abs(pspitch[2]-pspitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
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

                integrated_weight
            end
        end
    else
        psenergy = psgrid.energy
        pspitch = psgrid.pitch

        psgrid_energy = range(psgrid.energy[1],psgrid.energy[end],length(psgrid.energy)) 
        psgrid_pitch = range(psgrid.pitch[1],psgrid.pitch[end],length(psgrid.pitch))
        psgrid_r = range(psgrid.r[1],psgrid.r[end],length(psgrid.r))
        psgrid_z = range(psgrid.z[1],psgrid.z[end],length(psgrid.z))
        itp=Interpolations.scale(interpolate(F_ps_WeightsMatrix, spline_type),psgrid_energy,psgrid_pitch,psgrid_r,psgrid_z)

        sitp = extrapolate(itp,Flat())

        os_dist = zeros(length(og_orbs))
        @showprogress for iter=1:length(og_orbs)
            orb = og_orbs[iter]
            integrated_weight = 0.0

            for path_iter = 1:(length(orb.path.r)-1)
                F_os_i = orb.path.jacdets[path_iter]*sitp(orb.path.energy[path_iter],orb.path.pitch[path_iter],orb.path.r[path_iter],orb.path.z[path_iter])

                #Checking Energy Bounds:
                if (orb.path.energy[path_iter] > (psenergy[end]+0.5*(psenergy[end]-psenergy[end-1]))) || (orb.path.energy[path_iter] < (psenergy[1]-0.5*(psenergy[2]-psenergy[1]))) || (orb.path.energy[path_iter] < 0.0)
                    F_os_i = 0.0
                end
                #Checking Pitch Bounds:
                if pspitch[end]>pspitch[1]
                    if (orb.path.pitch[path_iter] > (pspitch[end]+0.5*abs(pspitch[end]-pspitch[end-1]))) || (orb.path.pitch[path_iter] < (pspitch[1]-0.5*abs(pspitch[2]-pspitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
                        F_os_i = 0.0
                    end
                else
                    if (orb.path.pitch[path_iter] < (pspitch[end]-0.5*abs(pspitch[end]-pspitch[end-1]))) || (orb.path.pitch[path_iter] > (pspitch[1]+0.5*abs(pspitch[2]-pspitch[1]))) || abs(orb.path.pitch[path_iter]) > 1.0
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

            os_dist[iter] = integrated_weight
        end
    end

    os_dist = [x<0.0 ? 0.0 : x for x in os_dist]

    if rescale_factor == 1.0
        os_dist =  (os_dist ./ sum(abs.(os_dist))) .* length(os_dist)
    elseif rescale_factor != 0.0
        os_dist =  (os_dist ./ sum(abs.(os_dist))) .* rescale_factor
    end
    return vec(os_dist)
end

"""
    psorbs_2_matrix(PS_orbs::Vector{GCEPRCoordinate})

Converts a vector of PS_orbs to a matrix using a for loop.
"""
function psorbs_2_matrix(PS_orbs::Vector{GCEPRCoordinate}) #No difference vs psorbs_2_matrix_INV
    points = Array{Float64}(undef,length(PS_orbs),4)
    for i = 1:length(PS_orbs)
        points[i,:] = [PS_orbs[i].energy,PS_orbs[i].pitch,PS_orbs[i].r,PS_orbs[i].z]
    end
    return points
end

"""
    psorbs_2_matrix_INV(PS_orbs::Vector{GCEPRCoordinate})

Converts a vector of PS_orbs to a matrix using a for loop reversing indices order. 
"""
function psorbs_2_matrix_INV(PS_orbs::Vector{GCEPRCoordinate})
    points = Array{Float64}(undef,4,length(PS_orbs))
    for i = 1:length(PS_orbs)
        points[:,i] = [PS_orbs[i].energy,PS_orbs[i].pitch,PS_orbs[i].r,PS_orbs[i].z]
    end
    return points
end

"""
    psorbs_2_matrix_StaticArray(PS_orbs::Vector{GCEPRCoordinate})

Converts a vector of PS_orbs to a matrix by concatenating a vector of static arrays (slow). 
"""
function psorbs_2_matrix_StaticArray(PS_orbs::Vector{GCEPRCoordinate})
    points = SVector{4, Float64}[]

    for i in PS_orbs
        push!(points,SVector{4}(i.energy,i.pitch,i.r,i.z))
    end

    return copy(reshape(reinterpret(Float64, points), (4,length(PS_orbs))))
end

"""
    psorbs_2_matrix_Darray(PS_orbs::Vector{GCEPRCoordinate})

Converts a vector of PS_orbs to a matrix by using DistributedArrays. Works on cluster over multiple nodes.
"""
function psorbs_2_matrix_Darray(PS_orbs::Vector{GCEPRCoordinate}) #works slow on cluster
    num_psorbs = length(PS_orbs)

    points = dzeros(Float64,(num_psorbs,4),workers(),[nworkers(),1])

    psorbs_2_matrix_DarrayInner(PS_orbs,points)

    points0 = convert(Array,points)
    close(points)

    return points0
end

"""
    psorbs_2_matrix_DarrayInner(PS_orbs::Vector{GCEPRCoordinate},points::DArray{Float64,2})

Inner component of psorbs_2_matrix_Darray. Fills out local component of Darray.
"""
function psorbs_2_matrix_DarrayInner(PS_orbs::Vector{GCEPRCoordinate},points::DArray{Float64,2})
    @sync @distributed for j = 1:nworkers()
        local_part = localpart(points)
        for (io,o) in enumerate(DistributedArrays.localindices(points)[1])
            local_part[io,:] = [PS_orbs[o].energy,PS_orbs[o].pitch,PS_orbs[o].r,PS_orbs[o].z]
        end
    end
end


"""
    psorbs_2_matrix_DistributedForLoop(PS_orbs::Vector{GCEPRCoordinate}) 

Converts a vector of PS_orbs to a matrix using a distributed for loop. Slower than linear for loop.
"""
function psorbs_2_matrix_DistributedForLoop(PS_orbs::Vector{GCEPRCoordinate}) 
    points = @distributed (vcat) for i = 1:length(PS_orbs)
        [PS_orbs[i].energy,PS_orbs[i].pitch,PS_orbs[i].r,PS_orbs[i].z]'
    end
    return points
end

"""
    psorbs_2_matrix_DarraySPMD2(PS_orbs::Vector{GCEPRCoordinate}, points::DArray{Float64,2}) 

Converts a vector of PS_orbs to a matrix using the DistributedArrays SPMD2 distributed kernel functionality. While not faster than a linear for loop for this application, this is a good introduction to this method.
To run this function, call the following:
"
    @everywhere using DistributedArrays
    points = dzeros(Float64,(length(PS_orbs),4),workers(),[nworkers(),1])
    SPMD.spmd(OrbitTomography.psorbs_2_matrix_DarraySPMD2,PS_orbs,points)
"
"""
function psorbs_2_matrix_DarraySPMD2(PS_orbs::Vector{GCEPRCoordinate}, points::DArray{Float64,2}) #Fastest distributed method on cluster -> randomly crashed on cluster (killed)
    local_part = points[:L]
    for (io,o) in enumerate(DistributedArrays.localindices(points)[1])
        local_part[io,:] = [PS_orbs[o].energy,PS_orbs[o].pitch,PS_orbs[o].r,PS_orbs[o].z]
    end
end


"""
    psorbs_2_matrix_SharedArray(PS_orbs::Vector{GCEPRCoordinate})

Converts a vector of PS_orbs to a matrix using a SharedArray. This is fast but requires a local distributed environment (all CPUs connected to the same node, no ssh tunneling). If not, breaks down (see https://stackoverflow.com/questions/64802561/julia-sharedarray-with-remote-workers-becomes-a-0-element-array)
"""
function psorbs_2_matrix_SharedArray(PS_orbs::Vector{GCEPRCoordinate})
    num_psorbs = length(PS_orbs)

    points = SharedArray{Float64}(num_psorbs,4)
    @sync @distributed for i = 1:num_psorbs
        points[i,:] = [PS_orbs[i].energy,PS_orbs[i].pitch,PS_orbs[i].r,PS_orbs[i].z]
    end

    points0 = convert(Array,points)
    @everywhere points=nothing

    return points0
end

"""
    ps2epr_sampled(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, F_ps_Weights::AbstractArray{Float64}, psgrid::PSGrid, og::OrbitGrid;...

Samples an EPRZ distribution to 3D orbit-space using ps2os_performance, using a PSGrid as an input.
genuine_PS_weights = false will scale EPRZ distribution by R, the jacobian of transform from cartesian to cylindrical coordinates.

Returns the orbit-space distribution in vector form, as well as the number of particles in the original orbit-space distribution (volume x distribution).
"""
function ps2epr_sampled(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, F_ps_Weights::AbstractArray{Float64}, psgrid::PSGrid, og::OrbitGrid; numOsamples::Int64, progress_prefactor = "_", two_pi_mod = false, genuine_PS_weights = false, verbose=false, GCP = GCDeuteron, distributed=true, nbatch = 1_000_000, saveProgress=true, kwargs...) 
    if length(size(F_ps_Weights)) == 1
        F_ps_Weights = ps_VectorToMatrix(F_ps_Weights,psgrid)
    end

    return ps2epr_sampled(M, wall, F_ps_Weights, psgrid.energy, psgrid.pitch, psgrid.r, psgrid.z, og; numOsamples=numOsamples, progress_prefactor = progress_prefactor, two_pi_mod = two_pi_mod, genuine_PS_weights = genuine_PS_weights, verbose=verbose, GCP = GCP, distributed=distributed, nbatch = nbatch, saveProgress=saveProgress, kwargs...) 
end

"""
    ps2epr_sampled(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, F_ps_Weights::AbstractArray{Float64}, energy::AbstractVector{Float64}, pitch::AbstractVector{Float64}, R::AbstractVector{Float64}, Z::AbstractVector{Float64}, og::OrbitGrid;...
    
Samples an EPRZ distribution to 3D orbit-space using ps2os_performance.
genuine_PS_weights = false will scale EPRZ distribution by R, the jacobian of transform from cartesian to cylindrical coordinates.

Returns the orbit-space distribution in vector form, as well as the number of particles in the original orbit-space distribution (volume x distribution).
"""
function ps2epr_sampled(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, F_ps_Weights::AbstractArray{Float64}, energy::AbstractVector{Float64}, pitch::AbstractVector{Float64}, R::AbstractVector{Float64}, Z::AbstractVector{Float64}, og::OrbitGrid; numOsamples::Int64, progress_prefactor = "_", two_pi_mod = false, genuine_PS_weights = true, verbose=false, GCP = GCDeuteron, distributed=true, nbatch = 1_000_000, saveProgress=true, kwargs...) 
    if length(size(F_ps_Weights)) == 1
        error("Call ps2epr_sampled(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, F_ps_Weights::AbstractArray{Float64}, psgrid::PSGrid, og::OrbitGrid) with PSGrid.")
    end

    if genuine_PS_weights #genuine PS_weights don't have to be reshaped by R (R is the jacobian determinant associated with going from cartesian to cylindrical coordinates).
        fr = F_ps_Weights
    else    
        fr = F_ps_Weights.*reshape(R,(1,1,length(R),1))
    end
    F_ps_Weights = nothing

    if two_pi_mod #if two_pi_mod = true, integrating out gyro angle as well as toroidal angle
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
    nfast = sum(fr.*dvols) 

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
    return ps2os_performance(M, wall, frdvols_cumsum_vector, subs, nfast, energy, pitch, R, Z, og; numOsamples=numOsamples, numOsamples_sofar=numOsamples_sofar, result_sofar=result_sofar, distributed=distributed, GCP=GCP, saveProgress=saveProgress, progress_prefactor=progress_prefactor, verbose=verbose, kwargs...)
end

"""
    ps2os_performance(M, wall, fr, dvols, nfast, energy, pitch, R, Z, og; numOsamples=100_000)
    
Working function that samples an EPRZ distribution to 3D orbit-space. 
#STUART MODIFIED Energy bounds (Search #^STUART MODIFIED)
"""
function ps2os_performance(M::AbstractEquilibrium, 
                            wall::Union{Nothing,Wall}, 
                            frdvols_cumsum_vector::AbstractVector, 
                            subs::CartesianIndices{4,NTuple{4,Base.OneTo{Int64}}}, 
                            nfast::Union{Int64,Float64}, 
                            energy::AbstractVector{Float64}, 
                            pitch::AbstractVector{Float64}, 
                            R::AbstractVector{Float64}, 
                            Z::AbstractVector{Float64}, 
                            og::OrbitGrid; 
                            numOsamples::Int64, 
                            numOsamples_sofar=0, 
                            result_sofar=zeros(size(og.counts)), 
                            distributed=true, 
                            GCP=GCDeuteron, 
                            saveProgress=true, 
                            nbatch = 1_000_000, 
                            verbose=false, 
                            progress_prefactor="",
                            rescale_factor = 0.0,
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
                    #^STUART MODIFIED, old one: if (o.coordinate.energy <= maximum(og.energy) && o.coordinate.energy >= minimum(og.energy)) 
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

    if rescale_factor == 1.0
        result .=  (result ./ sum(abs.(result))) .* length(result)
    elseif rescale_factor != 0.0
        result .=  (result ./ sum(abs.(result))) .* rescale_factor
    end

    return result, nfast
end

"""
    performance_helper(M, wall, nbatch, frdvols_cumsum_vector, subs, dE_vector, dp_vector, dR_vector, dZ_vector, energy, pitch, R, Z, og;  GCP=GCP, kwargs...)

Help the function ps2os_performance() with acquiring orbit samples when parallel computations are desired.
This is to enable the sampling process to be saved regularly when calculating a large number of samples.
If the sampling process is not saved, then progress will be lost when the super-user of the HPC terminates
the sampling process early, due to misinterpretation of Julia's way of distributed computing.

#STUART MODIFIED Energy bounds (Search #^STUART MODIFIED)
"""
function performance_helper(M::AbstractEquilibrium, wall::Union{Nothing,Wall}, numOsamples::Int64, frdvols_cumsum_vector::Array{Float64,1}, subs::CartesianIndices{4,NTuple{4,Base.OneTo{Int64}}}, dE_vector::Array{Float64,1}, dp_vector::Array{Float64,1}, dR_vector::Array{Float64,1}, dZ_vector::Array{Float64,1}, energy::AbstractVector{Float64}, pitch::AbstractVector{Float64}, R::AbstractVector{Float64}, Z::AbstractVector{Float64}, og::OrbitGrid; GCP=GCDeuteron, visualizeProgress=true, kwargs...)
    if numOsamples>0.0 # If there are actually a non-zero number of samples left to sample...
        if visualizeProgress
            F_os = @showprogress @distributed (+) for i=1:numOsamples # Compute one result, and reduce (add) it to a resulting vector F_os
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
                    o = get_orbit(M,GCP(E_sample,M.sigma*p_sample,R_sample,Z_sample); store_path=false, wall=wall, kwargs...) # Calculate the orbit
                    if (o.coordinate.energy <= (og.energy[end]+0.5*(og.energy[end]-og.energy[end-1]))) && (o.coordinate.energy >= (og.energy[1]-0.5*(og.energy[2]-og.energy[1]))) # Make sure it's within the energy bounds
                    #^STUART MODIFIED, old one: if (o.coordinate.energy <= maximum(og.energy) && o.coordinate.energy >= minimum(og.energy))  
                        F_os_i = bin_orbits(og,vec([o.coordinate]),weights=vec([1.0])) # Bin to the orbit grid
                    else
                        F_os_i = zeros(length(og.counts)) # Otherwise, zero
                    end
                else
                    F_os_i = zeros(length(og.counts)) # Otherwise, zero
                end
                F_os_i # Declare F_os_i as result to add to F_os
            end
        else
            F_os = @distributed (+) for i=1:numOsamples
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
                    o = get_orbit(M,GCP(E_sample,M.sigma*p_sample,R_sample,Z_sample); store_path=false, wall=wall, kwargs...) # Calculate the orbit
                    if (o.coordinate.energy <= (og.energy[end]+0.5*(og.energy[end]-og.energy[end-1]))) && (o.coordinate.energy >= (og.energy[1]-0.5*(og.energy[2]-og.energy[1]))) # Make sure it's within the energy bounds
                    #^STUART MODIFIED, old one: if (o.coordinate.energy <= maximum(og.energy) && o.coordinate.energy >= minimum(og.energy)) 
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

        return F_os
    else # ...otherwise just return a zero vector with the length of the number of valid orbits
        return zeros(length(og.counts))
    end
end

"""
checkIfGoodSample(E_sample, p_sample, R_sample, Z_sample, energy, pitch, R, Z)

The function checks if a sample is within bounds. Returns true if that is the case. Otherwise false. Bounds on samples do not assume equal grid-steps, and pitch values half a step beyond boundaries are included to ensure similar behaviour to ps2epr().
""" 
function checkIfGoodSample(E_sample::Float64, p_sample::Float64, R_sample::Float64, Z_sample::Float64, energy::AbstractVector{Float64}, pitch::AbstractVector{Float64}, R::AbstractVector{Float64}, Z::AbstractVector{Float64})

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

"""
    get4Dvols(E, p, R, Z)

This function will calculate the volume of all the hyper-voxels pertaining to the 4D grid. It assumes the hyper-voxels pertaining to the
upper-end (edge) grid-points have the same volumes as the hyper-voxels just inside of them. It return a 4D array, containing all the hyper-voxel
volumes. The 4D array will have size()=(length(E), length(p), length(R), length(Z)). The function assumes a rectangular 4D grid.
"""
function get4Dvols(E, p, R, Z)

    # Safety-check to ensure vectors
    if !(1==length(size(E))==length(size(p))==length(size(R))==length(size(Z)))
        throw(ArgumentError("Energy, pitch, R, Z inputs are not all vectors. Please correct and re-try."))
    end

    vols = zeros(length(E), length(p), length(R), length(Z))
    for Ei=1:length(E)
        if Ei==length(E)
            dE = abs(E[end]-E[end-1])
        else
            dE = abs(E[Ei+1]-E[Ei])
        end
        for pi=1:length(p)
            if pi==length(p)
                dp = abs(p[end]-p[end-1])
            else
                dp = abs(p[pi+1]-p[pi])
            end
            for Ri=1:length(R)
                if Ri==length(R)
                    dR = abs(R[end]-R[end-1])
                else
                    dR = abs(R[Ri+1]-R[Ri])
                end
                for Zi=1:length(Z)
                    if Zi==length(Z)
                        dZ = abs(Z[end]-Z[end-1])
                    else
                        dZ = abs(Z[Zi+1]-Z[Zi])
                    end

                    vols[Ei,pi,Ri,Zi] = dE*dp*dR*dZ
                end
            end
        end
    end

    return vols
end

