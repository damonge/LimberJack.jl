function get_nzs(nz_path, tracer_type, bin)
    nzs = npzread(string(nz_path, "nz_", tracer_type, bin, ".npz"))
    zs = nzs["z"]
    nz = nzs["dndz"]
    cov = get(nzs, "cov", zeros(length(zs)))
    return zs, nz, cov
end      

function Theory(cosmology::Cosmology, files;
                nz_path="../data/DESY1_cls/fiducial_nzs/",
                Nuisances=Dict())
    
    tracers_names = [eachrow(files["tracers"])...]
    pairs = [eachrow(files["pairs"])...]
    pairs_ids = [eachrow(files["pairs_ids"])...]
    
    nui_type = valtype(Nuisances)
    if !(nui_type <: Float64) & (nui_type != Any)
        if nui_type != Real
            cosmology.settings.cosmo_type = nui_type
        end
    end
    
    ntracers = length(tracers_names)
    tracers = []
    for i in 1:ntracers
        tracer = tracers_names[i]
        tracer_type = tracer[1]
        bin = tracer[2]
        zs_mean, nz_mean, cov = get_nzs(nz_path, tracer_type, bin)
        if tracer_type == 1
            b = get(Nuisances, string("b", bin), 1.0)
            nz = get(Nuisances, string("nz_g", bin), nz_mean)
            dzi = get(Nuisances, string("dz_g", bin), 0.0)
            zs = zs_mean .- dzi
            sel = zs .> 0.
            tracer = NumberCountsTracer(cosmology, zs[sel], nz[sel];
                                        b=b)
        elseif tracer_type == 2
            mb = get(Nuisances, string("mb", bin), 0.0)
            IA_params = [get(Nuisances, "A_IA", 0.0),
                         get(Nuisances, "alpha_IA", 0.0)]
            nz = get(Nuisances, string("nz_k", bin), nz_mean)
            dzi = get(Nuisances, string("dz_k", bin), 0.0)
            zs = zs_mean .- dzi
            sel = zs .> 0.
            tracer = WeakLensingTracer(cosmology, zs[sel], nz[sel];
                                       mb=mb, IA_params=IA_params)
        else
            print("Not implemented")
            trancer = nothing
        end
        push!(tracers, tracer)
        
    end
    
    npairs = length(pairs)
    idx = files["idx"]
    total_len = last(idx)
    cls = zeros(cosmology.settings.cosmo_type, total_len)
    @inbounds Threads.@threads for i in 1:npairs
        pair = pairs[i]
        ids = pairs_ids[i]
        ls = files[string("ls_", pair[1], pair[2], pair[3], pair[4])]
        tracer1 = tracers[ids[1]]
        tracer2 = tracers[ids[2]]
        cls[idx[i]+1:idx[i+1]] = angularCℓs(cosmology, tracer1, tracer2, ls)
    end
    
    return cls
    
end
