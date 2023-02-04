function Theory(cosmology::Cosmology,
                names, types, pairs,
                idx, files;
                Nuisances=Dict())
    
    nui_type =  eltype(valtype(Nuisances))
    if !(nui_type <: Float64) & (nui_type != Any)
        if nui_type != Real
            cosmology.settings.cosmo_type = nui_type
        end
    end
    
    tracers =  Dict{String}{Tracer}()
    ntracers = length(names)
    @inbounds for i in 1:ntracers
        name = names[i]
        t_type = types[i]
        if t_type == "galaxy_density"
            zs_mean, nz_mean = files[string("nz_", name)]
            b = get(Nuisances, string(name, "_", "b"), 1.0)
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            dz = get(Nuisances, string(name, "_", "dz"), 0.0)
            tracer = NumberCountsTracer(cosmology, zs .- dz, nz;
                                        b=b)
        elseif t_type == "galaxy_shear"
            zs_mean, nz_mean = files[string("nz_", name)]
            m = get(Nuisances, string(name, "_", "m"), 0.0)
            IA_params = [get(Nuisances, "A_IA", 0.0),
                         get(Nuisances, "alpha_IA", 0.0)]
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            dz = get(Nuisances, string(name, "_", "dz"), 0.0)
            tracer = WeakLensingTracer(cosmology, zs .- dz, nz;
                                       m=m, IA_params=IA_params)
            
        elseif t_type == "cmb_convergence"
            tracer = CMBLensingTracer(cosmology)

        else
            print("Not implemented")
            tracer = nothing
        end
        merge!(tracers, Dict(name => tracer))
    end

    npairs = length(pairs)
    total_len = last(idx)
    cls = zeros(cosmology.settings.cosmo_type, total_len)
    @inbounds Threads.@threads :static for i in 1:npairs
        name1, name2 = pairs[i]
        ls = files[string("ls_", name1, "_", name2)]
        tracer1 = tracers[name1]
        tracer2 = tracers[name2]
        cls[idx[i]+1:idx[i+1]] = angularCℓs(cosmology, tracer1, tracer2, ls)
    end
    
    return cls
end

function Theory(cosmology::Cosmology,
                instructions, files;
                Nuisances=Dict())
    
    names = instructions.names
    types = instructions.types
    pairs = instructions.pairs
    idx = instructions.idx
    
    return Theory(cosmology::Cosmology,
                  names, types, pairs,
                  idx, files;
                  Nuisances=Nuisances)
 end

function TheoryFast(cosmo::Cosmology,
                    names, types, pairs,
                    idx, files;
                    Nuisances=Dict())
    sett = cosmo.settings

    nui_type =  eltype(valtype(Nuisances))
    if !(nui_type <: Float64) & (nui_type != Any)
        if nui_type != Real
            sett.cosmo_type = nui_type
        end
    end

    chis = cosmo.chi(sett.zs_t)

    ntracers = length(names)
    tracers =  Dict{String}{Tracer}()
    W = zeros(sett.cosmo_type, ntracers, sett.nz_t)
    F = zeros(Float64, ntracers, sett.nℓ)
    @inbounds for i in 1:ntracers
        name = names[i]
        t_type = types[i]
        if t_type == "galaxy_density"
            zs_mean, nz_mean = files[string("nz_", name)]
            b = get(Nuisances, string(name, "_", "b"), 1.0)
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            dz = get(Nuisances, string(name, "_", "dz"), 0.0)
            tracer = NumberCountsTracer(cosmo, zs .- dz, nz;
                                        b=b)
        elseif t_type == "galaxy_shear"
            zs_mean, nz_mean = files[string("nz_", name)]
            m = get(Nuisances, string(name, "_", "m"), 0.0)
            IA_params = [get(Nuisances, "A_IA", 0.0),
                         get(Nuisances, "alpha_IA", 0.0)]
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            dz = get(Nuisances, string(name, "_", "dz"), 0.0)
            tracer = WeakLensingTracer(cosmo, zs .- dz, nz;
                                       m=m, IA_params=IA_params)

        elseif t_type == "cmb_convergence"
            tracer = CMBLensingTracer(cosmo)

        else
            print("Not implemented")
            tracer = nothing
        end
        merge!(tracers, Dict(name => tracer))
        W[i, :] .= tracer.wint(cosmo.chi(cosmo.zs_t))
        F[i, :] .= tracer.F(sett.ℓs)
    end

    C_ℓij = angularCℓsFast(cosmo, W, F)

    Cls_ij_itps = Dict{String}{AbstractExtrapolation}()
    @inbounds for i ∈ axes(C_ℓij, 2)
        @inbounds for j ∈ axes(C_ℓij, 3)
            name_i = names[i]
            name_j = names[j]
            cl_name = string("Cl_", name_i, name_j)
            itp = cubic_spline_interpolation(sett.ℓs, C_ℓij[:,i,j])
            merge!(Cls_ij_itps, Dict(cl_name =>itp))
        end
    end

    npairs = length(pairs)
    total_len = last(idx)
    theory = zeros(sett.cosmo_type, total_len)
    @inbounds for i in 1:npairs
        name1, name2 = pairs[i]
        cl_name = string("Cl_", name1, name2)
        itp = Cls_ij_itps[cl_name]
        ls = files[string("ls_", name1, "_", name2)]
        theory[idx[i]+1:idx[i+1]] = itp(ls)
    end

    return theory
end

function TheoryFast(cosmology::Cosmology,
                    instructions, files;
                    Nuisances=Dict())

    names = instructions.names
    types = instructions.types
    idx = instructions.idx
    pairs = instructions.pairs

    return TheoryFast(cosmology::Cosmology,
                      names, types, pairs,
                      idx, files;
                      Nuisances=Nuisances)
 end