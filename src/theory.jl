function get_nzs(nz_path, tracer_name)
    nzs = npzread(string(nz_path, "nz_", tracer_name, ".npz"))
    zs = nzs["z"]
    nz = nzs["dndz"]
    cov = get(nzs, "cov", zeros(length(zs)))
    return zs, nz, cov
end

function Theory(cosmology::Cosmology,
                tracers_names, pairs,
                pairs_ids, idx;
                nz_path="data/FD/nzs/",
                Nuisances=Dict())

    nui_type = valtype(Nuisances)
    if !(nui_type <: Float64) & (nui_type != Any)
        if nui_type != Real
            cosmology.settings.cosmo_type = nui_type
        end
    end

    ntracers = length(tracers_names)
    tracers = []
    for name in tracers_names
        n = length(name)
        t_type = name[n:n]
        zs_mean, nz_mean, cov = get_nzs(nz_path, name)
        if t_type == "0"
            b = get(Nuisances, string(name, "_", "b"), 1.0)
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            dzi = get(Nuisances, string(name, "_", "dz"), 0.0)
            zs = zs_mean .+ dzi  # Opposite sign in KiDS
            sel = zs .> 0.
            tracer = NumberCountsTracer(cosmology, zs[sel], nz[sel];
                                        b=b)
        elseif t_type == "e"
            mb = get(Nuisances, string(name, "_", "mb"), 0.0)
            IA_params = [get(Nuisances, "A_IA", 0.0),
                         get(Nuisances, "alpha_IA", 0.0)]
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            dzi = get(Nuisances, string(name, "_", "dz"), 0.0)
            zs = zs_mean .+ dzi  # Opposite sign in KiDS
            sel = zs .> 0.
            tracer = WeakLensingTracer(cosmology, zs[sel], nz[sel];
                                       mb=mb, IA_params=IA_params)
        else
            print("Not implemented")
            trancer = nothing
        end
        push!(tracers, tracer)

    end

    println("Hello World")

    npairs = length(pairs)
    total_len = last(idx)
    cls = zeros(cosmology.settings.cosmo_type, total_len)
    @inbounds Threads.@threads for i in 1:npairs
        name1, name2 = pairs[i]
        id1, id2 = pairs_ids[i]
        ls = files[string("ls_", name1, "_", name2)]
        ls = pyconvert(Vector{Float64}, ls)
        tracer1 = tracers[id1]
        tracer2 = tracers[id2]
        cls[idx[i]+1:idx[i+1]] = angularCℓs(cosmology, tracer1, tracer2, ls)
    end

    return cls
end
