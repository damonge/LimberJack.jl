struct cls_meta{T<:AbstractVector}
    tracers::T
    pairs::T
    pairs_ids::T
end

cls_meta(file) = begin
    tracers = [c[:] for c in eachrow(file["tracers"])]
    pairs = [c[:] for c in eachrow(file["pairs"])]
    pairs_ids = [c[:] for c in eachrow(file["pairs_ids"])]
    cls_meta(tracers, pairs, pairs_ids)
end

struct Theory
    tracers
    cls
end

Theory(cosmology::Cosmology, cls_meta, files;
       Nuisances=Dict()) = begin
    
    nui_type = valtype(Nuisances)
    if !(nui_type <: Float64) & (nui_type != Any)
        if nui_type != Real
            cosmology.settings.cosmo_type = nui_type
        end
    end
    
    ntracers = length(cls_meta.tracers)
    tracers = []
    for i in 1:ntracers
        tracer = cls_meta.tracers[i]
        tracer_type = tracer[1]
        bin = tracer[2]
        nzs = files[string("nz_", tracer_type, bin)]
        nz = vec(nzs[2:2, :])
        zs = vec(nzs[1:1, :])
        if tracer_type == 1
            bias = get(Nuisances, string("b", bin), 1.0)
            dzi = get(Nuisances, string("dz_g", bin), 0.0)
            zs_shift =  zs .- dzi
            sel = zs_shift .> 0.
            tracer = NumberCountsTracer(cosmology, zs_shift[sel], nz[sel];
                                        bias=bias)
        elseif tracer_type == 2
            mbias = get(Nuisances, string("m", bin), 0.0)
            dzi = get(Nuisances, string("dz_k", bin), 0.0)
            IA_params = [get(Nuisances, "A_IA", 0.0),
                         get(Nuisances, "alpha_IA", 0.0)]
            zs_shift =  zs .- dzi
            sel = zs_shift .> 0.
            tracer = WeakLensingTracer(cosmology, zs_shift[sel], nz[sel];
                                       mbias=mbias, IA_params=IA_params)
        else
            print("Not implemented")
            trancer = nothing
        end
        push!(tracers, tracer)
        
    end
    
    npairs = length(cls_meta.pairs)
    idx = files["idx"]
    total_len = last(idx)
    cls = zeros(cosmology.settings.cosmo_type, total_len)
    @inbounds Threads.@threads for i in 1:npairs
        pair = cls_meta.pairs[i]
        ids = cls_meta.pairs_ids[i]
        ls = files[string("ls_", pair[1], pair[2], pair[3], pair[4])]
        tracer1 = tracers[ids[1]]
        tracer2 = tracers[ids[2]]
        cls[idx[i]+1:idx[i+1]] = angularCℓs(cosmology, tracer1, tracer2, ls)
    end
    
    Theory(tracers, cls)
    
end
