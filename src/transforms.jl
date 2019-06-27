struct EPDensity{T<:AbstractMatrix,S<:AbstractVector}
    d::T
    detJ::T
    energy::S
    pitch::S
end

function local_distribution(M::AxisymmetricEquilibrium, grid::OrbitGrid, f::Vector, r, z;
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

function local_distribution(M::AxisymmetricEquilibrium, OS::OrbitSystem, f, orbs, sigma, r, z;
                            Js=Vector{Vector{Float64}}[],
                            energy=range(1.0,80.0,length=25),
                            pitch=range(-0.99,0.99,length=25),
                            distributed=false, atol=1e-3,
                            covariance=:local, norms=S3(1.0,1.0,1.0), kwargs...)

    nenergy = length(energy)
    npitch = length(pitch)
    m = orbs[1].coordinate.m
    q = orbs[1].coordinate.q
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

function rz_profile(M::AxisymmetricEquilibrium, OS::OrbitSystem, f::Vector, orbs, sigma;
                            Js=Vector{Vector{Float64}}[],
                            energy=range(1.0,80.0,length=25),
                            pitch=range(-0.99,0.99,length=25),
                            r = range(extrema(M.r)...,length=25),
                            z = range(extrema(M.z)...,length=25),
                            distributed=false, atol=1e-3, domain_check= (xx,yy) -> true,
                            covariance=:local, norms=S3(1.0,1.0,1.0),
                            checkpoint=true, warmstart=false,file="rz_progress.jld2", kwargs...)

    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)
    m = orbs[1].coordinate.m
    q = orbs[1].coordinate.q

    f_rz = zeros(nr,nz)

    if warmstart && isfile(file) && (filesize(file) != 0)
        @load file ir_start iz_start f_rz
        ir = ir_start
        iz = iz_start
    else
        ir_start = 1
        iz_start = 1
        ir = 1
        iz = 1
    end

    if checkpoint
        touch(file)
    end

    for j=iz:nz
        for i=ir:nr
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
            ir_start = i + 1
            if checkpoint
                @save file ir_start iz_start f_rz
            end
       end
       iz_start = j + 1
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

function eprz_distribution(M::AxisymmetricEquilibrium, OS::OrbitSystem, f::Vector, orbs, sigma;
                            Js=Vector{Vector{Float64}}[],
                            energy=range(1.0,80.0,length=25),
                            pitch=range(-0.99,0.99,length=25),
                            r = range(extrema(M.r)...,length=25),
                            z = range(extrema(M.z)...,length=25),
                            distributed=false, atol=1e-3, domain_check= (xx,yy) -> true,
                            covariance=:local, norms=S3(1.0,1.0,1.0),
                            checkpoint=true, warmstart=false,file="eprz_progress.jld2", kwargs...)

    nenergy = length(energy)
    npitch = length(pitch)
    nr = length(r)
    nz = length(z)
    m = orbs[1].coordinate.m
    q = orbs[1].coordinate.q

    f_eprz = zeros(nenergy,npitch,nr,nz)

    if warmstart && isfile(file) && (filesize(file) != 0)
        @load file ir_start iz_start f_eprz
        ir = ir_start
        iz = iz_start
    else
        ir_start = 1
        iz_start = 1
        ir = 1
        iz = 1
    end

    if checkpoint
        touch(file)
    end

    for j=iz:nz
        for i=ir:nr
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
            ir_start = i + 1
            if checkpoint
                @save file ir_start iz_start f_eprz
            end
        end
        iz_start = j + 1
    end

    return EPRZDensity(f_eprz,energy,pitch,r,z)
end
