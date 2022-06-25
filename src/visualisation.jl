"""
    plot_PS_distribution_drop_dims(psgrid,f;...)

This function plots an (energy, pitch) and an (R,Z) plot of a particle-space distirbution, averaging over hidden dimensions.
auto_normalise sets the average distribution value to 1. bluesreds prints a variation from 0 in a blue/red scale, good for mapping the difference calculated between two distributions.
Include wall and overplot_boundary=true to get the tokamak wall in the RZ plot. Set colorbar = true for a colorbar.
Use filename_prefactor to set a path and a unique name for the distribution when printing (need savefig=true).
"""
function plot_PS_distribution_drop_dims(psgrid,f;auto_normalise=true,pngg=false,LMARG=50px, Mmarker=:white, M = nothing, wall=nothing, cgrad_ = 0, scale = 0, bluesreds = false, its4D = false, savefig=true, filename_prefactor="",fs = "medium", vert=false,cmax=[],fig=nothing,ax=nothing,picsize=600,colorbar=false, overplot_boundary = true, overplot_magnetic_axis = true)
    if bluesreds
        auto_normalise=false
    else 
        if auto_normalise
            f = f .* (length(f)/sum(f))
        end
    end
    
    if !its4D
        f4d = ps_VectorToMatrix(f,psgrid)
    else
        f4d = f
    end

    ne=length(psgrid.energy)
    np=length(psgrid.pitch)
    nr=length(psgrid.r)
    nz=length(psgrid.z)

    npoints = ne*np*nr*nz

    subs = CartesianIndices((ne,np,nr,nz))

    dE = abs(psgrid.energy[2]-psgrid.energy[1]) # Assume equidistant os-psgrid
    dp = abs(psgrid.pitch[2] - psgrid.pitch[1]) # Assume equidistant os-psgrid
    dr = abs(psgrid.r[2]-psgrid.r[1]) # Assume equidistant os-psgrid
    dz = abs(psgrid.z[2]-psgrid.z[1]) # Assume equidistant os-psgrid

    epr_pdf = dropdims(sum(f4d,dims=4),dims=4)/nz #*dz # Integrate out the z dimension
    #epz_pdf = dropdims(sum(f4d,dims=3),dims=3)/nr #*dr # Integrate out the r dimension
    #erz_pdf = dropdims(sum(f4d,dims=2),dims=2)/np #*dp # Integrate out the p dimension
    prz_pdf = dropdims(sum(f4d,dims=1),dims=1)/ne #*dE # Integrate out the E dimension

    ep_pdf = dropdims(sum(epr_pdf,dims=3),dims=3)/nr #*dr
    rz_pdf = dropdims(sum(prz_pdf,dims=1),dims=1)/np #*dp

    if length(cmax) == 0 # If no maximum for the color scale is defined, use the maximum of the functions
        cmax = zeros(2)
        cmax[1] = maximum(ep_pdf)
        cmax[2] = maximum(rz_pdf)
        #cmax[3] = maximum(er_pdf)
    end

    if colorbar==false
        legend = :none
    else
        legend = true
    end

    if scale != 0
        cgrad1 = cgrad(:default, scale=scale)
    else
        cgrad1 = cgrad(:default)
    end

    if cgrad_ != 0
        cgrad1=cgrad_
    end

    if typeof(M)!=Nothing
        raxis = M.axis[1]
        zaxis = M.axis[2]
    end

    normalised_ratio = (maximum(psgrid.z)-minimum(psgrid.z))/(maximum(psgrid.r)-minimum(psgrid.r))
    E_P_normalised_ratio = (maximum(psgrid.energy)-minimum(psgrid.energy))/2.0
    
    erange = maximum(psgrid.energy)-minimum(psgrid.energy)
    prange = maximum(psgrid.pitch)-minimum(psgrid.pitch)
    rrange = maximum(psgrid.r)-minimum(psgrid.r)
    zrange = maximum(psgrid.z)-minimum(psgrid.z)

    if !bluesreds
        ep_plot = Plots.heatmap(psgrid.energy, psgrid.pitch, ep_pdf', thickness_scaling = 1.2,colorbar=colorbar, aspect_ratio = normalised_ratio*E_P_normalised_ratio,xlims = (minimum(psgrid.energy),maximum(psgrid.energy)),  ylims = (-1.0,1.0), bottom_margin = 15px,left_margin=10px,right_margin=10px, dpi =600 , xlabel="Energy (KeV)", ylabel="Pitch", c=cgrad1,fontfamily="Computer Modern",legend=legend)
        #savefig && Plots.savefig(string(filename_prefactor,"EP_heatmap"))
        annotate!(-0.05*erange+minimum(psgrid.energy),-0.08*prange+minimum(psgrid.pitch), text("a.", :black, :right))
        rz_plot = Plots.heatmap(psgrid.r, psgrid.z, rz_pdf',thickness_scaling = 1.2,colorbar=colorbar, bottom_margin = 15px,left_margin=10px,right_margin=10px, ylims = (minimum(psgrid.z),maximum(psgrid.z)),xlims = (minimum(psgrid.r),maximum(psgrid.r)), dpi =600 , xlabel="R (m)", ylabel="Z (m)",  c=cgrad1,fontfamily="Computer Modern",legend=legend,aspect_ratio=:equal)
        annotate!(-0.05*rrange+minimum(psgrid.r),-0.08*zrange+minimum(psgrid.z), text("b.", :black, :right))
        if overplot_boundary && typeof(wall)!=Nothing
            rz_plot = Plots.plot!(wall.r, wall.z,  thickness_scaling = 1.1,ylims = (minimum(psgrid.z),maximum(psgrid.z)),xlims = (minimum(psgrid.r),maximum(psgrid.r)),  bottom_margin = 15px,left_margin=10px,right_margin=10px,  dpi =600 ,  label="tokamak wall",background_color_legend=:lightgrey , color=:white,aspect_ratio=:equal)
        end
        if overplot_magnetic_axis && typeof(M)!=Nothing
            rz_plot = Plots.scatter!([raxis],[zaxis], thickness_scaling = 1.1,ylims = (minimum(psgrid.z),maximum(psgrid.z)),xlims = (minimum(psgrid.r),maximum(psgrid.r)),  bottom_margin = 15px,left_margin=10px,right_margin=10px,  dpi =600 ,markershape = :x, markersize=3, label="magnetic axis",legend=:bottomright,color=Mmarker,aspect_ratio=:equal)
        end
    else
        ep_plot = Plots.heatmap(psgrid.energy, psgrid.pitch, ep_pdf', thickness_scaling = 1.2, colorbar=colorbar, aspect_ratio = normalised_ratio*E_P_normalised_ratio, ylims = (-1.0,1.0), xlims = (minimum(psgrid.energy),maximum(psgrid.energy)), bottom_margin = 15px,left_margin=10px,right_margin=10px,  dpi =600 , xlabel="Energy (KeV)", ylabel="Pitch", c = :bluesreds, clims = (-maximum(abs.(ep_pdf)),maximum(abs.(ep_pdf))) ,fontfamily="Computer Modern" ,legend=legend)
        #savefig && Plots.savefig(string(filename_prefactor,"EP_heatmap"))
        annotate!(-0.05*erange+minimum(psgrid.energy),-0.08*prange+minimum(psgrid.pitch), text("a.", :black, :right))
        rz_plot = Plots.heatmap(psgrid.r, psgrid.z, rz_pdf',thickness_scaling = 1.2,colorbar=colorbar, bottom_margin = 15px,left_margin=10px, ylims = (minimum(psgrid.z),maximum(psgrid.z)),xlims = (minimum(psgrid.r),maximum(psgrid.r)),right_margin=10px,  dpi =600 , xlabel="R (m)", ylabel="Z (m)", c = :bluesreds, clims = (-maximum(abs.(rz_pdf)),maximum(abs.(rz_pdf))) ,fontfamily="Computer Modern" ,legend=legend,aspect_ratio=:equal)
        annotate!(-0.05*rrange+minimum(psgrid.r),-0.08*zrange+minimum(psgrid.z), text("b.", :black, :right))
        if overplot_boundary && typeof(wall)!=Nothing
            rz_plot = Plots.plot!(wall.r, wall.z, thickness_scaling = 1.1, ylims = (minimum(psgrid.z),maximum(psgrid.z)),xlims = (minimum(psgrid.r),maximum(psgrid.r)) ,bottom_margin = 15px,left_margin=10px,right_margin=10px,  dpi =600 ,  label="tokamak wall",background_color_legend=:lightgrey ,color=:white,aspect_ratio=:equal)
        end #xticks=false,yticks=false,
        if overplot_magnetic_axis && typeof(M)!=Nothing
            rz_plot = Plots.scatter!([raxis],[zaxis], thickness_scaling = 1.1, ylims = (minimum(psgrid.z),maximum(psgrid.z)),xlims = (minimum(psgrid.r),maximum(psgrid.r)),  bottom_margin = 15px,left_margin=10px,right_margin=10px,  dpi =600 ,markershape = :x,markersize=3, label="magnetic axis", legend=:bottomright, color=Mmarker,aspect_ratio=:equal)
        end
    end 

    Plots.plot(ep_plot,rz_plot,bottom_margin = 30px,left_margin=LMARG,right_margin=30px, dpi =600 ,  size = (picsize*1.5,picsize),fontfamily="Computer Modern")
    if savefig 
        if pngg
            Plots.savefig(string(filename_prefactor,"PSDist.png"))
        else
            Plots.savefig(string(filename_prefactor,"PSDist.pdf"))
        end
    end
end

"""
    plot_PS_distribution_EP(psgrid,f;...)

This function plots an (energy, pitch) plot of a particle-space distirbution, averaging over hidden RZ dimensions.
rint1, rint2, zint1, zint2 provide ranges over which to average the distribution. 
To use a specific RZ point, set rint1 = rint2 and zint1 = zint2 to the same, desired value.

Set colorbar = true for a colorbar. biglim_bool=true sets  the color limits from the original distribution, while biglim_bool=false sets them using the RZ-averaged distribution.
Use filename_prefactor to set a path and a unique name for the distribution when printing (need savefig=true).
"""
function plot_PS_distribution_EP(psgrid,f; colorbar=false, biglim_bool = true, rints = Int[],zint1 = 0, zint2 = 0, rint1 = 0, rint2 = 0, savefig=true, filename_prefactor="",fs = "medium", vert=false,cmax=[],fig=nothing,ax=nothing,picsize=4000)
    if length(size(f))==1
        f4d = ps_VectorToMatrix(f,psgrid)
    else
        f4d = f
    end

    #if zint1 == zint2 == 1
    #    error("enter proper zints")
    #end
    #if rint1 == rint2 == 1
    #    error("enter proper rints")
    #end

    colorscheme = cgrad([:white,:cyan,:blue,:green,:orange,:red,:darkred],scale = :log10)

    ne=length(psgrid.energy)
    np=length(psgrid.pitch)
    nr=length(psgrid.r)
    nz=length(psgrid.z)

    npoints = ne*np*nr*nz

    z_avg = zeros(Float64,ne,np,nr)
    rz_avg = zeros(Float64,ne,np)

    if (rint1 == rint2) && (zint1 == zint2) && (zint1 != 0)
        rz_avg = f4d[:,:,rint1,zint1]
        ep_plot = Plots.heatmap(psgrid.energy, psgrid.pitch, rz_avg', colorbar=colorbar, ylims = (-1.0,1.0), bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel="Energy (KeV)", ylabel="Pitch", c = colorscheme)
        savefig && Plots.savefig(string(filename_prefactor,"EPDist_$(zint1)__$(rint1)"))
        return nothing
    end

    if (rint1 == rint1 == 0) && (zint1 != 0)
        olddir = pwd()
    
        !isdir(string(filename_prefactor,"_z$(zint1)")) && mkdir(string(filename_prefactor,"_z$(zint1)"))
        cd(string(filename_prefactor,"_z$(zint1)"))

        biglim0 = maximum(abs.(f4d[:,:,:,zint1]))
    
        if length(rints) != 0
            for ioo in rints
                rz_avg = f4d[:,:,ioo,zint1]
                biglim_bool ? (bl = biglim0) : (bl = maximum(abs.(rz_avg)))
                ep_plot = Plots.heatmap(psgrid.energy, psgrid.pitch, rz_avg', colorbar=colorbar, ylims = (-1.0,1.0), bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel="Energy (KeV)", ylabel="Pitch", c = colorscheme, clims = (0.0,bl))
                savefig && Plots.savefig(string(filename_prefactor,"EPDist_z$(zint1)__r$(ioo)"))
            end
        else
            for ioo = 1:nr
                rz_avg = f4d[:,:,ioo,zint1]
                biglim_bool ? (bl = biglim0) : (bl = maximum(abs.(rz_avg)))
                ep_plot = Plots.heatmap(psgrid.energy, psgrid.pitch, rz_avg', colorbar=colorbar, ylims = (-1.0,1.0), bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel="Energy (KeV)", ylabel="Pitch", c = colorscheme,clims = (0.0,bl))
                savefig && Plots.savefig(string(filename_prefactor,"EPDist_z$(zint1)__r$(ioo)"))
            end
        end


        cd(olddir)
        return nothing
    end

    if (rint1 == rint2 == zint1 == zint2 == 0) 
        rint1 = 1
        zint1 = 1
        rint2 = nr
        zint2 = nz
    end

    for i in 1:ne
        for j in 1:np
            for k in 1:nr
                z_avg[i,j,k] = mean(f4d[i,j,k,zint1:zint2])
            end
        end
    end

    for i in 1:ne
        for j in 1:np
            rz_avg[i,j] = mean(z_avg[i,j,rint1:rint2])
        end
    end

    subs = CartesianIndices((ne,np,nr,nz))

    if length(cmax) == 0 # If no maximum for the color scale is defined, use the maximum of the functions
        #cmax = zeros(1)
        cmax = maximum(rz_avg)
        #cmax[2] = maximum(rz_pdf)
        #cmax[3] = maximum(er_pdf)
    end

    ep_plot = Plots.heatmap(psgrid.energy, psgrid.pitch, rz_avg', colorbar=colorbar, ylims = (-1.0,1.0), bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel="Energy (KeV)", ylabel="Pitch", c = colorscheme)

    savefig && Plots.savefig(string(filename_prefactor,"EPDist_$(zint1)_$(zint2)__$(rint1)_$(rint2)"))
end

function map_orbits_no_scaling(grid::OrbitGrid, f::Vector, os_equidistant::Bool)
    if length(grid.counts) != length(f)
        throw(ArgumentError("Incompatible sizes"))
    end

    if os_equidistant
        return [i == 0 ? zero(f[1]) : f[i] for i in grid.orbit_index]
    else
        return [i == 0 ? zero(f[1]) : f[i] for i in grid.orbit_index] 
    end
end

"""
    plot_OG_distribution_drop_dims(ogrid,f;...)

This function plots an (energy, pitch_m), a (pitch_m,r_m) and an (energy, r_m) plot of an orbit-space distribution, averaging over hidden dimensions.
auto_normalise sets the average distribution value to 1. bluesreds prints a variation from 0 in a blue/red scale, good for mapping the difference calculated between two distributions.
Use filename_prefactor to set a path and a unique name for the distribution when printing (need savefig=true).
"""
function plot_OG_distribution_drop_dims(ogrid,f;auto_normalise = true,NE213_fix=false,slice_color=:green, energy_slice = nothing, pngg=false,ep_dimcut=[0,0],spec_size=(600,1.0*600), spec_layout = @layout([a b;c{0.7h}]), cgrad_ = 0, scale = 0, bluesreds = false, its3D = false, savefig=true, filename_prefactor="",fs = "medium", vert=false,cmax=[],fig=nothing,ax=nothing,picsize=600,colorbar=false)
    if bluesreds
        auto_normalise=false
    else 
        if auto_normalise
            f = f .* (length(f)/sum(f))
        end
    end
    
    if !its3D
        f3d = map_orbits_no_scaling(ogrid,f,true)
        #dorb = abs((ogrid.r[2]-ogrid.r[1])*(ogrid.energy[2]-ogrid.energy[1])*(ogrid.pitch[2]-ogrid.pitch[1]))
        #f3d = f3d .* (grid.counts[i]*dorb)
    else
        f3d = f
    end
        
    ne=length(ogrid.energy)
    np=length(ogrid.pitch)
    nr=length(ogrid.r)

    filename_prefactor_orbit = string(ne,"x",np,"x",nr,"_")

    npoints = ne*np*nr

    subs = CartesianIndices((ne,np,nr))

    dE = abs(ogrid.energy[2]-ogrid.energy[1]) # Assume equidistant os-ogrid
    dp = abs(ogrid.pitch[2] - ogrid.pitch[1]) # Assume equidistant os-ogrid
    dr = abs(ogrid.r[2]-ogrid.r[1]) # Assume equidistant os-ogrid

    ep_pdf = dropdims(sum(f3d,dims=3),dims=3)/nr #*dr # Integrate out the r dimension
    er_pdf = dropdims(sum(f3d,dims=2),dims=2)/np #*dp # Integrate out the p dimension
    pr_pdf = dropdims(sum(f3d,dims=1),dims=1)/ne #*dE # Integrate out the E dimension

    if ep_dimcut[1]>0
        for i in 1:ep_dimcut[1]
            ep_pdf[i,:].=0.0
            er_pdf[i,:].=0.0
        end

    end

    if ep_dimcut[2]>0
        for i in 1:ep_dimcut[2]
            ep_pdf[:,i].=0.0
        end
        for i in np-ep_dimcut[1]:ne
            ep_pdf[:,i].=0.0
        end
    end

    #if length(cmax) == 0 # If no maximum for the color scale is defined, use the maximum of the functions
        #cmax = zeros(2)
        #cmax[1] = maximum(ep_pdf)
        #cmax[2] = maximum(rz_pdf)
        #cmax[3] = maximum(er_pdf)
    #end
    
    if scale != 0
        cgrad1 = cgrad(:default, scale=scale)
    else
        cgrad1 = cgrad(:default)
    end

    if cgrad_ != 0
        cgrad1=cgrad_
    end

    if colorbar==false
        legend = :none
    else
        legend = true
    end

    enrange = maximum(ogrid.energy)-minimum(ogrid.energy)
    pirange = maximum(ogrid.pitch)-minimum(ogrid.pitch)
    rrange = maximum(ogrid.r)-minimum(ogrid.r)

    if !bluesreds
        if !NE213_fix
            p1 = Plots.heatmap(ogrid.energy, ogrid.pitch, ep_pdf',bottom_margin = 10px,c=cgrad1, left_margin=10px,right_margin=10px,xlabel="Energy (KeV)", ylabel=L"p_m",dpi =600,fontfamily="Computer Modern",legend=false)
            annotate!(-0.08*enrange+minimum(ogrid.energy),-0.22*pirange+minimum(ogrid.pitch), text("a.", :black, :right))
            p2 = Plots.heatmap(ogrid.r,ogrid.energy,er_pdf,bottom_margin = 10px,c=cgrad1, left_margin=10px,right_margin=10px,xlabel=L"R_m  (m)", ylabel="Energy (KeV)",dpi =600,fontfamily="Computer Modern",legend=false)
            annotate!(-0.08*rrange+minimum(ogrid.r),-0.22*enrange+minimum(ogrid.energy), text("b.", :black, :right))
            
            #if colorbar
            #    pr_pdf = pr_pdf .* (1/maximum(pr_pdf))
            #end
            p3 = Plots.heatmap(ogrid.r,ogrid.pitch, pr_pdf, ylims=(-1.0,1.0),clims = (0,maximum(pr_pdf)), bottom_margin = 10px,c=cgrad1, left_margin=10px,right_margin=10px,xlabel=L"R_m  (m)", ylabel=L"p_m",dpi =600,fontfamily="Computer Modern",legend=legend,xlims = (minimum(ogrid.r),maximum(ogrid.r)), aspect_ratio=0.32)
            annotate!(p3,-0.055*rrange+minimum(ogrid.r),-0.08*pirange+minimum(ogrid.pitch), text("c.", :black, :right))
            typeof(energy_slice)!=Nothing && (p3 = Overplot_topological_contour2(ogrid,energy_slice,boundary_color=slice_color))
            
            #annotate!(p3,3.2,0, text("c.", :white, :right))
        else
            p1 = Plots.heatmap(ogrid.r,ogrid.pitch, pr_pdf, ylims=(-1.0,1.0), bottom_margin = 10px,c=cgrad1, left_margin=10px,right_margin=10px,xlabel=L"R_m  (m)", ylabel=L"p_m",dpi =600,fontfamily="Computer Modern",legend=false,xlims = (minimum(ogrid.r),maximum(ogrid.r)), aspect_ratio=0.32)
            annotate!(-0.08*enrange+minimum(ogrid.energy),-0.22*pirange+minimum(ogrid.pitch), text("a.", :black, :right))
            p2 = Plots.heatmap(ogrid.r,ogrid.energy,er_pdf,bottom_margin = 10px,c=cgrad1, left_margin=10px,right_margin=10px,xlabel=L"R_m  (m)", ylabel="Energy (KeV)",dpi =600,fontfamily="Computer Modern",legend=false)
            annotate!(-0.08*rrange+minimum(ogrid.r),-0.22*enrange+minimum(ogrid.energy), text("b.", :black, :right))
            
            #if colorbar
            #    ep_pdf = ep_pdf .* (1/maximum(ep_pdf))
            #end
            p3 = Plots.heatmap(ogrid.energy, ogrid.pitch, ep_pdf',clims = (0,maximum(ep_pdf)), bottom_margin = 10px,c=cgrad1, left_margin=10px,right_margin=10px,xlabel="Energy (KeV)", ylabel=L"p_m",dpi =600,fontfamily="Computer Modern",legend=legend)
            
            annotate!(p3,-0.055*rrange+minimum(ogrid.r),-0.08*pirange+minimum(ogrid.pitch), text("c.", :black, :right))

            typeof(energy_slice)!=Nothing && (p3 = Overplot_topological_contour2(ogrid,energy_slice,boundary_color=slice_color))
            
            #annotate!(p3,3.2,0, text("c.", :white, :right))
        end
    else
        p1 = Plots.heatmap(ogrid.energy, ogrid.pitch, ep_pdf',bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel="Energy (KeV)", ylabel=L"p_m", c = :bluesreds, clims = (-maximum(abs.(ep_pdf)),maximum(abs.(ep_pdf))) ,dpi =600,fontfamily="Computer Modern",legend=false)
        #p2 = Plots.heatmap(ogrid.energy,ogrid.r, er_pdf',bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel="Energy (KeV)", ylabel="Rm (m)", c = :bluesreds, clims = (-maximum(abs.(er_pdf)),maximum(abs.(er_pdf))) ,dpi =600,fontfamily="Computer Modern" )
        annotate!(-0.08*enrange+minimum(ogrid.energy),-0.22*pirange+minimum(ogrid.pitch), text("a.", :black, :right))
        p2 = Plots.heatmap(ogrid.r,ogrid.energy, er_pdf,bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel=L"R_m  (m)", ylabel="Energy (KeV)", c = :bluesreds, clims = (-maximum(abs.(er_pdf)),maximum(abs.(er_pdf))) ,dpi =600,fontfamily="Computer Modern" ,legend=false)
        annotate!(-0.08*rrange+minimum(ogrid.r),-0.22*enrange+minimum(ogrid.energy), text("b.", :black, :right))
        
        #if colorbar
        #    pr_pdf = pr_pdf .* (1/length(ogrid.energy))
        #end

        p3 = Plots.heatmap(ogrid.r,ogrid.pitch, pr_pdf, ylims=(-1.0,1.0), bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel=L"R_m  (m)", ylabel=L"p_m", c = :bluesreds, xlims = (minimum(ogrid.r),maximum(ogrid.r)), dpi =600,fontfamily="Computer Modern" ,legend=legend,aspect_ratio=0.32,clim = (-maximum(abs.(pr_pdf)),maximum(abs.(pr_pdf))))
        annotate!(p3,-0.055*rrange+minimum(ogrid.r),-0.08*pirange+minimum(ogrid.pitch), text("c.", :black, :right))

        #p3 = Overplot_topological_contour2(ogrid,energy_slice,boundary_color=slice_color)
        typeof(energy_slice)!=Nothing && (p3 = Overplot_topological_contour2(ogrid,energy_slice,standalone=false,boundary_color=slice_color))
        #topoMap_tot = Overplot_topological_contour2(ogrid,energy_slice,standalone=false,boundary_color=slice_color)
        #Plots.contour!(p3,energy_slice.r,energy_slice.pitch,topoMap_tot,dpi=600,legend=false,levels = [0.00005,0.00015,0.00025,0.00035,0.00045,0.00055,0.00065,0.00075,0.00085,0.00095],c=:black)
    
    end

    Plots.plot(p1,p2,p3,layout=spec_layout,size = spec_size,bottom_margin = 15px,left_margin=25px,right_margin=20px,dpi =600,fontfamily="Computer Modern",legend=legend)
    
    if pngg
        savefig && Plots.savefig(string(filename_prefactor,filename_prefactor_orbit,"Dist.png"))
    else
        savefig && Plots.savefig(string(filename_prefactor,filename_prefactor_orbit,"Dist.pdf"))
    end
end

"""
    plot_OG_distribution_ALL_E(ogrid,f;...)

This function plots a series of (r_m, pitch_m) plots of an orbit-space distribution at different energy values. Use energy_inds to specify specific energies, otherwise defaults to plotting the whole energy range.
bluesreds prints a variation from 0 in a blue/red scale, good for mapping the difference calculated between two distributions.
"""
function plot_OG_distribution_ALL_E(ogrid,f; thickness_scaling0=1.4, custom_lim=0,colorbar=true, rel_scale = true, bluesreds = false,energy_inds = Int64[], its3D = false, savefig=true, filename_prefactor="",fs = "medium", vert=false,cmax=[],fig=nothing,ax=nothing,picsize=4000)
    if !its3D
        f3d = map_orbits_no_scaling(ogrid,f,true)
    else
        f3d = f
    end

    biglim = rel_scale

    ne=length(ogrid.energy)
    np=length(ogrid.pitch)
    nr=length(ogrid.r)

    new_dir=false
    if length(energy_inds)==0 
        energy_inds = 1:ne
        new_dir=true
    elseif length(energy_inds)>5 
        new_dir=true
    end

    filename_prefactor_orbit = string(ne,"x",np,"x",nr,"_")

    npoints = ne*np*nr

    subs = CartesianIndices((ne,np,nr))

    dE = abs(ogrid.energy[2]-ogrid.energy[1]) # Assume equidistant os-ogrid
    dp = abs(ogrid.pitch[2] - ogrid.pitch[1]) # Assume equidistant os-ogrid
    dr = abs(ogrid.r[2]-ogrid.r[1]) # Assume equidistant os-ogrid

    #ep_pdf = dropdims(sum(f3d,dims=3),dims=3)*dr # Integrate out the r dimension
    #er_pdf = dropdims(sum(f3d,dims=2),dims=2)*dp # Integrate out the p dimension
    #pr_pdf = dropdims(sum(f3d,dims=1),dims=1)*dE # Integrate out the E dimension

    if new_dir
        olddir = pwd()
        
        !isdir(string(filename_prefactor_orbit,filename_prefactor,"_All_Energies")) && mkdir(string(filename_prefactor_orbit,filename_prefactor,"_All_Energies"))
        cd(string(filename_prefactor_orbit,filename_prefactor,"_All_Energies"))
    end

    biglim0 = maximum(abs.(f3d))

    if !bluesreds
        for (io,i) in enumerate(energy_inds)
            if custom_lim!=0
                Plots.heatmap(ogrid.r,ogrid.pitch, f3d[i,:,:],colorbar=colorbar, thickness_scaling = thickness_scaling0, bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel=L"R_m  (m)", ylabel=L"p_m",aspect_ratio=0.32,xlims = (minimum(ogrid.r),maximum(ogrid.r)),ylims=(-1.0,1.0), clims = (0,custom_lim));
            else
                Plots.heatmap(ogrid.r,ogrid.pitch, f3d[i,:,:],colorbar=colorbar, thickness_scaling = thickness_scaling0, bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel=L"R_m (m)", ylabel=L"p_m",aspect_ratio=0.32,xlims = (minimum(ogrid.r),maximum(ogrid.r)),ylims=(-1.0,1.0));
            end
            Plots.savefig(string(filename_prefactor_orbit,filename_prefactor,"MeV_",ogrid.energy[i],"Rm_vs_pm.png"))
        end
    else
        for (io,i) in enumerate(energy_inds)
            biglim || (biglim0 = maximum(abs.(f3d[i,:,:])))
            Plots.heatmap(ogrid.r,ogrid.pitch, f3d[i,:,:],colorbar=colorbar,clim = (-maximum(abs.(pr_pdf)),maximum(abs.(pr_pdf))), thickness_scaling = thickness_scaling0, bottom_margin = 10px,left_margin=10px,right_margin=10px,xlabel=L"R_m  (m)", ylabel=L"p_m", c = :bluesreds, clims = (-biglim0,biglim0) ,xlims = (minimum(ogrid.r),maximum(ogrid.r)),ylims=(-1.0,1.0),aspect_ratio=0.32);
            Plots.savefig(string(filename_prefactor_orbit,filename_prefactor,"MeV_",ogrid.energy[i],"Rm_vs_pm.png"))
        end
    end

    
    new_dir && cd(olddir)
end

"""
    Overplot_topological_contour(ogrid,f,ogrid_Energy_Ind,energy_slice;...)

This function plots a (r_m, pitch_m) plot of an orbit-space distribution (averaging over Energy), and overplots the topological boundary of a high-resolution, single-energy orbit grid called energy_slice. 
bluesreds prints a variation from 0 in a blue/red scale, good for mapping the difference calculated between two distributions.
"""
function Overplot_topological_contour(ogrid,f,ogrid_Energy_Ind,energy_slice; colorbar=false, boundary_color = :black, thickness_scaling0=1.4, bluesreds= false, custom_lim=0, its3D = false, res = 0,energy_slice_ind = 1, distributed = false, distinguishLost = false, distinguishIncomplete = false, filename_prefactor = "", overlay = false, linewidth = 30, xres = 5000, yres = 3000, plot_centers = false, kwargs...)
    if !its3D
        f3d = map_orbits_no_scaling(ogrid,f,true)
    else
        f3d = f
    end

    length(energy_slice.energy) == 1 && (energy_slice_ind=1)

    if res==0
        npitch = length(energy_slice.pitch)
        nr = length(energy_slice.r)

        subs = CartesianIndices((npitch,nr))
        npoints = npitch*nr

        topoMap_tot = zeros(npitch,nr)

        for l=1:npoints
            i,j = Tuple(subs[l])
            class = energy_slice.class[energy_slice_ind,i,j]

            if (class == :lost) && distinguishLost
                topoMap_tot[i,j] =  7
            elseif (class == :incomplete) && distinguishIncomplete
                topoMap_tot[i,j] =  6
            elseif class == :trapped
                topoMap_tot[i,j] =  2
            elseif class == :co_passing
                topoMap_tot[i,j] =  3
            elseif class == :stagnation
                topoMap_tot[i,j] =  1
            elseif class == :potato
                topoMap_tot[i,j] =  5
            elseif class == :ctr_passing
                topoMap_tot[i,j] =  4
            else
                topoMap_tot[i,j] =  9
            end
        end
        levels = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5] 


    else
        pm_array = range(-0.999,0.999,length=res)
        Rm_array = range(M.axis[1],maximum(wall.r),length=res)

        subs = CartesianIndices((res,res))
        npoints = res*res

        topoMap_tot = zeros(res,res)

        

        @showprogress for l=1:npoints
            i,j = Tuple(subs[l])
            topoMap_tot[i,j] = assign_topological_height(energy_slice, pm_array[i], Rm_array[j],max_difference=true)
        end

        levels = range(-5,15,step = 0.5)
        levels = []
        
        filename_prefactor = string(filename_prefactor,"ArbRes",res,"_")
        
    end

    #biglim0 = maximum(abs.(f3d[ogrid_Energy_Ind,:,:]))
    #Plots.heatmap(ogrid.r,ogrid.pitch, f3d[ogrid_Energy_Ind,:,:], xlabel="Rm [m]", ylabel="pm", title="Energy Slice: $(energy_slice.energy[energy_slice_ind]) MeV, Distribution Energy: $(ogrid.energy[ogrid_Energy_Ind]) MeV", size=(xres,yres), margin=25mm, c = :bluesreds, clims = (-biglim0,biglim0)  )

    if !bluesreds
        if custom_lim!=0
            Plots.heatmap(ogrid.r,ogrid.pitch, f3d[ogrid_Energy_Ind,:,:], dpi=600, colorbar=colorbar, thickness_scaling = thickness_scaling0, bottom_margin = 10px,left_margin=10px,right_margin=10px, title = "Energy: $(round(ogrid.energy[ogrid_Energy_Ind],digits=3)) KeV", xlabel=L"R_m  (m)", ylabel=L"p_m",aspect_ratio=0.32,xlims = (minimum(ogrid.r),maximum(ogrid.r)),ylims=(-1.0,1.0), clims = (0,custom_lim));
        else
            Plots.heatmap(ogrid.r,ogrid.pitch, f3d[ogrid_Energy_Ind,:,:], dpi=600, colorbar=colorbar, thickness_scaling = thickness_scaling0, bottom_margin = 10px,left_margin=10px,right_margin=10px, title = "Energy: $(round(ogrid.energy[ogrid_Energy_Ind],digits=3)) KeV", xlabel=L"R_m (m)", ylabel=L"p_m",aspect_ratio=0.32,xlims = (minimum(ogrid.r),maximum(ogrid.r)),ylims=(-1.0,1.0));
        end
    else
        biglim || (biglim0 = maximum(abs.(f3d[ogrid_Energy_Ind,:,:])))
        Plots.heatmap(ogrid.r,ogrid.pitch, f3d[ogrid_Energy_Ind,:,:],colorbar=colorbar, thickness_scaling = thickness_scaling0, bottom_margin = 10px,left_margin=10px,right_margin=10px, title = "Energy: $(round(ogrid.energy[ogrid_Energy_Ind],digits=3)) KeV", xlabel=L"R_m  (m)", ylabel=L"p_m", c = :bluesreds, clims = (-biglim0,biglim0) ,xlims = (minimum(ogrid.r),maximum(ogrid.r)),ylims=(-1.0,1.0),aspect_ratio=0.32);
    end

    if res==0
        Plots.contour!(energy_slice.r,energy_slice.pitch,topoMap_tot; dpi=600,  levels=levels,c=boundary_color,kwargs...)
    else
        Plots.contour!(Rm_array,pm_array,topoMap_tot; dpi=600,  levels=levels,c=boundary_color,kwargs...)
    end

    if plot_centers
        npitch = length(ogrid.pitch)
        nr = length(ogrid.r)
    
        #subs = CartesianIndices((npitch,nr))

        xcentres = zeros(Float64,npitch*nr)
        ycentres = zeros(Float64,npitch*nr)

        for i=0:(npitch-1)
            xcentres[(1+i*npitch):(i*npitch+nr)] .= ogrid.pitch[i+1]
            ycentres[(1+i*npitch):(i*npitch+nr)] = ogrid.r[1:nr]
        end

        filename_prefactor = string("Gridded",filename_prefactor)
 
        Plots.scatter!(ycentres,xcentres,dpi=600,  markersize = 0.2,color=:white)
    end

    png(string(filename_prefactor,"topoContour_MeV",ogrid.energy[ogrid_Energy_Ind]))

    return nothing
end

"""
    Overplot_topological_contour2(ogrid,energy_slice;...)

This function plots the topological boundary of a high-resolution, single-energy orbit grid called energy_slice in the (r_m, pithc_m) space. If standalone=true, it will plot over an existing plot, otherwise it makes a standalone plot.
"""
function Overplot_topological_contour2(ogrid,energy_slice; standalone=false, colorbar=false, boundary_color = :black, thickness_scaling0=1.4, custom_lim=0, its3D = false, res = 0,energy_slice_ind = 1, distributed = false, distinguishLost = false, distinguishIncomplete = false, filename_prefactor = "", overlay = false, linewidth = 30, xres = 3000, yres = 3000, plot_centers = false, kwargs...)

    length(energy_slice.energy) == 1 && (energy_slice_ind=1)

    npitch = length(energy_slice.pitch)
    nr = length(energy_slice.r)

    subs = CartesianIndices((npitch,nr))
    npoints = npitch*nr

    topoMap_tot = zeros(npitch,nr)

    for l=1:npoints
        i,j = Tuple(subs[l])
        class = energy_slice.class[energy_slice_ind,i,j]

        if (class == :lost) && distinguishLost
            topoMap_tot[i,j] =  0.0007
        elseif (class == :incomplete) && distinguishIncomplete
            topoMap_tot[i,j] =  0.0006
        elseif class == :trapped
            topoMap_tot[i,j] =  0.0002
        elseif class == :co_passing
            topoMap_tot[i,j] =  0.0003
        elseif class == :stagnation
            topoMap_tot[i,j] =  0.0001
        elseif class == :potato
            topoMap_tot[i,j] =  0.0005
        elseif class == :ctr_passing
            topoMap_tot[i,j] =  0.0004
        else
            topoMap_tot[i,j] =  0.0009
        end
    end
    levels = [0.00005,0.00015,0.00025,0.00035,0.00045,0.00055,0.00065,0.00075,0.00085,0.00095]

    #biglim0 = maximum(abs.(f3d[ogrid_Energy_Ind,:,:]))
    #Plots.heatmap(ogrid.r,ogrid.pitch, f3d[ogrid_Energy_Ind,:,:], xlabel="Rm [m]", ylabel="pm", title="Energy Slice: $(energy_slice.energy[energy_slice_ind]) MeV, Distribution Energy: $(ogrid.energy[ogrid_Energy_Ind]) MeV", size=(xres,yres), margin=25mm, c = :bluesreds, clims = (-biglim0,biglim0)  )
    if standalone
        overplot = Plots.contour(energy_slice.r,energy_slice.pitch,topoMap_tot; dpi=600, colorbar = false, levels=levels,c=boundary_color,kwargs...)
    else
        overplot = Plots.contour!(energy_slice.r,energy_slice.pitch,topoMap_tot; dpi=600, colorbar = false, levels=levels,c=boundary_color,kwargs...)
    end
    if plot_centers
        npitch = length(ogrid.pitch)
        nr = length(ogrid.r)
    
        #subs = CartesianIndices((npitch,nr))

        xcentres = zeros(Float64,npitch*nr)
        ycentres = zeros(Float64,npitch*nr)

        for i=0:(npitch-1)
            xcentres[(1+i*npitch):(i*npitch+nr)] .= ogrid.pitch[i+1]
            ycentres[(1+i*npitch):(i*npitch+nr)] = ogrid.r[1:nr]
        end

        filename_prefactor = string("Gridded",filename_prefactor)
 
        overplot = Plots.scatter!(ycentres,xcentres,dpi=600,  markersize = 0.2,color=:white)
    end

    return overplot
    #return topoMap_tot
end

"""
    plot_Energy_Dist(ogrid,f;...)

This function plots a 1D energy distribution of an orbit-space distirbution, averaging over the other two variables.
"""
function plot_Energy_Dist(ogrid::OrbitGrid,f;overlay = false, its3D=false,renorm = true)
    if !its3D
        f3d = map_orbits_no_scaling(ogrid,f,true)
    else
        f3d = f
    end
        
    ne=length(ogrid.energy)
    np=length(ogrid.pitch)
    nr=length(ogrid.r)

    filename_prefactor_orbit = string(ne,"x",np,"x",nr,"_")

    npoints = ne*np*nr

    subs = CartesianIndices((ne,np,nr))

    dE = abs(ogrid.energy[2]-ogrid.energy[1]) # Assume equidistant os-ogrid
    dp = abs(ogrid.pitch[2] - ogrid.pitch[1]) # Assume equidistant os-ogrid
    dr = abs(ogrid.r[2]-ogrid.r[1]) # Assume equidistant os-ogrid

    ep_pdf = dropdims(sum(f3d,dims=3),dims=3)*dr # Integrate out the r dimension
    er_pdf = dropdims(sum(f3d,dims=2),dims=2)*dp # Integrate out the p dimension
    #pr_pdf = dropdims(sum(f3d,dims=1),dims=1)*dE # Integrate out the E dimension

    e_pdf1 = dropdims(sum(ep_pdf,dims=2),dims=2)*dp # Integrate out the r dimension
    #e_pdf2 = dropdims(sum(er_pdf,dims=2),dims=2)*dr # Integrate out the r dimension

    #display(e_pdf1)
    #display(e_pdf2)

    if renorm
        e_pdf1 = e_pdf1 .* (length(e_pdf1)/sum(e_pdf1))
    end

    overlay ? Plots.plot!(ogrid.energy,e_pdf1) : Plots.plot(ogrid.energy,e_pdf1)
end

"""
    plot_Energy_Dist(psgrid,f;...)

This function plots a 1D energy distribution of a particle-space distirbution, averaging over the other three variables.
"""
function plot_Energy_Dist(psgrid::PSGrid,f;overlay = false, its4D=false, renorm = true)
    if !its4D
        f4d = ps_VectorToMatrix(f,psgrid)
    else
        f4d = f
    end

    ne=length(psgrid.energy)
    np=length(psgrid.pitch)
    nr=length(psgrid.r)
    nz=length(psgrid.z)

    if ne==np==nr==nz
        filename_prefactor = string("S",np,"_")
    else 
        filename_prefactor_orbit = string("S_",ne,"x",np,"x",nr,"x",nz,"_")
    end

    npoints = ne*np*nr*nz

    #subs = CartesianIndices((ne,np,nr,nz))
    dE = abs(psgrid.energy[2]-psgrid.energy[1]) # Assume equidistant os-psgrid
    dp = abs(psgrid.pitch[2] - psgrid.pitch[1]) # Assume equidistant os-psgrid
    dr = abs(psgrid.r[2]-psgrid.r[1]) # Assume equidistant os-psgrid
    dz = abs(psgrid.z[2]-psgrid.z[1]) # Assume equidistant os-psgrid

    epr_pdf = dropdims(sum(f4d,dims=4),dims=4)*dz # Integrate out the z dimension
    #epz_pdf = dropdims(sum(f4d,dims=3),dims=3)*dr # Integrate out the r dimension
    #erz_pdf = dropdims(sum(f4d,dims=2),dims=2)*dp # Integrate out the p dimension
    #prz_pdf = dropdims(sum(f4d,dims=1),dims=1)*dE # Integrate out the E dimension

    ep_pdf = dropdims(sum(epr_pdf,dims=3),dims=3)*dr
    #rz_pdf = dropdims(sum(prz_pdf,dims=1),dims=1)*dp

    e_pdf1 = dropdims(sum(ep_pdf,dims=2),dims=2)*dp 

    #display(e_pdf1)
    #display(e_pdf2)
    if renorm
        e_pdf1 = e_pdf1 .* (length(e_pdf1)/sum(e_pdf1))
    end

    overlay ? Plots.plot!(psgrid.energy,e_pdf1) : Plots.plot(psgrid.energy,e_pdf1)
end

function plot_topological_boundary(ogrid; energy_ind = 1)
    length(ogrid.energy) == 1 && (energy_ind=1)
    npitch = length(ogrid.pitch)
    nr = length(ogrid.r)

    subs = CartesianIndices((npitch,nr))
end

function assign_topological_height(ogrid, pitch, Rm; energy_ind = 1, distinguishLost = false, distinguishIncomplete = false, max_difference = false)
    length(ogrid.energy) == 1 && (energy_ind=1)
    j = argmin(abs.(pitch .- ogrid.pitch))
    k = argmin(abs.(Rm .- ogrid.r))
    class = ogrid.class[energy_ind,j,k]

    if max_difference
        if class == :trapped
            return 10
        elseif class == :co_passing
            return 0
        elseif class == :stagnation
            return 5
        elseif class == :potato
            return 5
        elseif class == :ctr_passing
            return 0
        else
            return 10
        end
    else
        if (class == :lost) && distinguishLost
            return 7
        elseif (class == :incomplete) && distinguishIncomplete
            return 6
        elseif class == :trapped
            return 2
        elseif class == :co_passing
            return 3
        elseif class == :stagnation
            return 1
        elseif class == :potato
            return 5
        elseif class == :ctr_passing
            return 4
        else
            return 9
        end
    end
end

function plot_topological_heatmap(ogrid; energy_ind = 1, distributed = false, distinguishLost = false, distinguishIncomplete = false, filename_prefactor = "")
    length(ogrid.energy) == 1 && (energy_ind=1)
    npitch = length(ogrid.pitch)
    nr = length(ogrid.r)

    subs = CartesianIndices((npitch,nr))
    npoints = npitch*nr

    topoMap_tot = zeros(npitch,nr)

    for l=1:npoints
        i,j = Tuple(subs[l])
        class = ogrid.class[energy_ind,i,j]

        if (class == :lost) && distinguishLost
            topoMap_tot[i,j] =  7
        elseif (class == :incomplete) && distinguishIncomplete
            topoMap_tot[i,j] =  6
        elseif class == :trapped
            topoMap_tot[i,j] =  2
        elseif class == :co_passing
            topoMap_tot[i,j] =  3
        elseif class == :stagnation
            topoMap_tot[i,j] =  1
        elseif class == :potato
            topoMap_tot[i,j] =  5
        elseif class == :ctr_passing
            topoMap_tot[i,j] =  4
        else
            topoMap_tot[i,j] =  9
        end
    end

    Plots.heatmap(ogrid.r,ogrid.pitch,topoMap_tot,color=:Set1_9,legend=false,xlabel="Rm [m]", ylabel="pm", title="Energy: $(ogrid.energy[energy_ind]) KeV")
    png(string(filename_prefactor,"topoMap_JET_96100_53,0s_$(ogrid.energy[energy_ind])MeV"))
end

function plot_topological_contour(ogrid; res = 0,energy_ind = 1, distributed = false, distinguishLost = false, distinguishIncomplete = false, filename_prefactor = "", overlay = false, linewidth = 30, xres = 5000, yres = 3000, kwargs...)
    length(ogrid.energy) == 1 && (energy_ind=1)

    if res==0
        npitch = length(ogrid.pitch)
        nr = length(ogrid.r)

        subs = CartesianIndices((npitch,nr))
        npoints = npitch*nr

        topoMap_tot = zeros(npitch,nr)

        for l=1:npoints
            i,j = Tuple(subs[l])
            class = ogrid.class[energy_ind,i,j]

            if (class == :lost) && distinguishLost
                topoMap_tot[i,j] =  7
            elseif (class == :incomplete) && distinguishIncomplete
                topoMap_tot[i,j] =  6
            elseif class == :trapped
                topoMap_tot[i,j] =  2
            elseif class == :co_passing
                topoMap_tot[i,j] =  3
            elseif class == :stagnation
                topoMap_tot[i,j] =  1
            elseif class == :potato
                topoMap_tot[i,j] =  5
            elseif class == :ctr_passing
                topoMap_tot[i,j] =  4
            else
                topoMap_tot[i,j] =  9
            end
        end
        if !overlay
            Plots.contour(ogrid.r,ogrid.pitch,topoMap_tot; levels=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5],c=:black, colorbar=false,legend=false,xlabel="Rm [m]", ylabel="pm", title="Energy: $(ogrid.energy[energy_ind]) KeV", size=(xres,yres),margin=25mm, kwargs...)
            png(string(filename_prefactor,"topoContour_JET_96100_53,0s_$(ogrid.energy[energy_ind])MeV"))
        else
            Plots.contour!(ogrid.r,ogrid.pitch,topoMap_tot; levels=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5],c=:black, colorbar=false,legend=false,xlabel="Rm [m]", ylabel="pm", title="Energy: $(ogrid.energy[energy_ind]) KeV", size=(xres,yres), margin=25mm,kwargs...)
        end
    else
        pm_array = range(-0.999,0.999,length=res)
        Rm_array = range(M.axis[1],maximum(wall.r),length=res)

        subs = CartesianIndices((res,res))
        npoints = res*res

        topoMap_tot = zeros(res,res)

        @showprogress for l=1:npoints
            i,j = Tuple(subs[l])
            topoMap_tot[i,j] = assign_topological_height(ogrid, pm_array[i], Rm_array[j])
        end

        levels = range(-1,11,step = 0.005)
        levels = []
        
        filename_prefactor = string(filename_prefactor,"ArbRes",res,"_")
        if !overlay
            Plots.contour(Rm_array,pm_array,topoMap_tot;levels=levels,c=:black, colorbar=false,legend=false,xlabel="Rm [m]", ylabel="pm", title="Energy: $(ogrid.energy[energy_ind]) KeV", size=(xres,yres),margin=25mm, kwargs...)
            png(string(filename_prefactor,"topoContour_JET_96100_53,0s_$(ogrid.energy[energy_ind])MeV"))
        else
            Plots.contour!(ogrid.r,ogrid.pitch,topoMap_tot;levels=levels,c=:black, colorbar=false,legend=false,xlabel="Rm [m]", ylabel="pm", title="Energy: $(ogrid.energy[energy_ind]) KeV", size=(xres,yres), margin=25mm,kwargs...)
        end
    end

    return nothing
end

function plot_topological_contour2(ogrid; energy_ind = 1, distributed = false, distinguishLost = false, distinguishIncomplete = false)
    length(ogrid.energy) == 1 && (energy_ind=1)
    npitch = length(ogrid.pitch)
    nr = length(ogrid.r)

    subs = CartesianIndices((npitch,nr))
    npoints = npitch*nr

    topoMap_tot = zeros(npitch,nr)

    for l=1:npoints
        i,j = Tuple(subs[l])
        

        class = ogrid.class[energy_ind,i,j]

        if (class == :lost) && distinguishLost
            topoMap_tot[i,j] =  7
        elseif (class == :incomplete) && distinguishIncomplete
            topoMap_tot[i,j] =  6
        elseif class == :trapped
            topoMap_tot[i,j] =  2
        elseif class == :co_passing
            topoMap_tot[i,j] =  3
        elseif class == :stagnation
            topoMap_tot[i,j] =  1
        elseif class == :potato
            topoMap_tot[i,j] =  5
        elseif class == :ctr_passing
            topoMap_tot[i,j] =  4
        else
            topoMap_tot[i,j] =  9
        end
    end

    #Plots.heatmap(ogrid.r,ogrid.pitch,topoMap_tot,color=:Set1_9,legend=false,xlabel="Rm [m]", ylabel="pm", title="Energy: $(ogrid.energy[energy_ind]) MeV")
    #png("topoMap_JET_96100_53,0s_$(ogrid.energy[energy_ind])MeV")
end