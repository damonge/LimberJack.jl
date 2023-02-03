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

    chis = cosmo.chi(zs_t)

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
            tracer = CMBLensingTracer(cosmology)

        else
            print("Not implemented")
            tracer = nothing
        end
        merge!(tracers, Dict(name => tracer))
        W[i, :] .= tracer.wint(chis)
        F[i, :] .= tracer.F(sett.ℓs)
    end

    P = zeros(Float64, sett.nz_t, sett.nℓ)
    for z ∈ axes(sett.zs_t, 1)
        for ℓ ∈  axes(sett.ℓs, 1)
            P[z, ℓ] = nonlin_Pk(cosmo, (sett.ℓs[ℓ]+0.5)/chis[z], sett.zs_t[z])
        end
    end
    Ezs = Ez(cosmo, sett.zs_t)
    chis = cosmo.chi(sett.zs_t)
    dz = (sett.zs_t[2] - sett.zs_t[1])
    SimpsonWeights = SimpsonWeightArray(sett.nz_t)
    C_ℓij = zeros(sett.cosmo_type, sett.nℓ, ntracers, ntracers)
    for ℓ ∈ axes(C_ℓij, 1)
        for i ∈ axes(C_ℓij, 2)
            for j ∈ axes(C_ℓij, 3)
                for z ∈ axes(sett.zs_t, 1)
                    integrand = (W[i, z] * W[j, z] * P[z, ℓ]) / (Ezs[z] * chis[z]^2)
                    C_ℓij[ℓ,i,j] += integrand * SimpsonWeights[z] * dz
                end
                C_ℓij[ℓ,i,j] *= CLIGHT_HMPC * F[i, ℓ] * F[j, ℓ]
            end
        end
    end

    Cls_ij_itps = Dict{String}{AbstractExtrapolation}()
    for i ∈ axes(C_ℓij, 2)
        for j ∈ axes(C_ℓij, 3)
            name_i = names[i]
            name_j = names[j]
            cl_name = string("Cl_", name_i, name_j)
            itp = linear_interpolation(sett.ℓs, C_ℓij[:,i,j])
            merge!(Cls_ij_itps, Dict(cl_name =>itp))
        end
    end

    npairs = length(pairs)
    total_len = last(idx)
    theory = zeros(sett.cosmo_type, total_len)
    for i in 1:npairs
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