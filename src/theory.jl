function get_nzs(nz_path, tracer_name)
    nzs = npzread(string(nz_path, "nz_", tracer_name, ".npz"))
    zs = nzs["z"]
    nz = nzs["dndz"]
    cov = get(nzs, "cov", zeros(length(zs)))
    return zs, nz, cov
end

function Theory(cosmology::Cosmology,
                tracers_names, pairs,
                idx, files;
                Nuisances=Dict())

    ntracers = length(tracers_names)
    tracers = []
    
    nui_type =  eltype(valtype(Nuisances))
    if !(nui_type <: Float64) & (nui_type != Any)
        if nui_type != Real
            cosmology.settings.cosmo_type = nui_type
        end
    end
    
    tracers =  Dict{String}{Tracer}()
    
    @inbounds for name in tracers_names
        n = length(name)
        t_type = name[n:n]
        if t_type == "0"
            nzs = files[string("nz_", name)]
            nzs = [nzs[i,:] for i in 1:size(nzs,1)]
            zs_mean, nz_mean = nzs[1], nzs[2]

            b = get(Nuisances, string(name, "_", "b"), 1.0)
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            #nz .*=  nz .> 0 # Enforce positive
            dzi = get(Nuisances, string(name, "_", "dz"), 0.0)
            zs = zs_mean .+ dzi  # Opposite sign in KiDS
            #zs .*=  zs .> 0 # Enforce positive
            tracer = NumberCountsTracer(cosmology, zs, nz;
                                        b=b)
        elseif t_type == "e"
            nzs = files[string("nz_", name)]
            nzs = [nzs[i,:] for i in 1:size(nzs,1)]
            zs_mean, nz_mean = nzs[1], nzs[2]

            mb = get(Nuisances, string(name, "_", "mb"), 0.0)
            IA_params = [get(Nuisances, "A_IA", 0.0),
                         get(Nuisances, "alpha_IA", 0.0)]
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            #nz .*=  nz .> 0 # Enforce positive
            dzi = get(Nuisances, string(name, "_", "dz"), 0.0)
            zs = zs_mean .+ dzi  # Opposite sign in KiDS
            #zs .*=  zs .> 0 # Enforce positive
            tracer = WeakLensingTracer(cosmology, zs, nz;
                                       mb=mb, IA_params=IA_params)
            
        elseif t_type == "k"
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
