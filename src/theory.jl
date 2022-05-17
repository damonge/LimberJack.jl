struct cls_meta{T<:AbstractVector}
    tracers::T
    pairs::T
    pairs_ids::T
end

function cls_meta(file)
    tracers = [c[:] for c in eachrow(file["tracers"])]
    pairs = [c[:] for c in eachrow(file["pairs"])]
    pairs_ids = [c[:] for c in eachrow(file["pairs_ids"])]
    cls_meta(tracers, pairs, pairs_ids)
end

function fill_NuisancePars(params_dict)
    nui_names = keys(params_dict)
    
    if "b0" in nui_names
        b0 = params_dict["b0"]
    else 
        b0 = 1.0
    end
    
    if "b1" in nui_names
        b1 = params_dict["b1"]
    else
        b1 = 1.0
    end
    
    if "b2" in nui_names
        b2 = params_dict["b2"]
    else 
        b2 = 1.0
    end
    
    if "b3" in nui_names    
        b3 = params_dict["b3"]
    else 
        b3 = 1.0
    end
    
    if "b4" in nui_names
        b4 = params_dict["b4"]
    else 
        b4 = 1.0
    end 

    if "dz_g0" in nui_names
        dz_g0 = params_dict["dz_g0"]
    else 
        dz_g0 = 0.0
    end

     if "dz_g1" in nui_names
        dz_g1 = params_dict["dz_g1"]
    else 
        dz_g1 = 0.0
    end
    
    if "dz_g2" in nui_names
        dz_g2 = params_dict["dz_g2"]
    else 
        dz_g2 = 0.0
    end
    
    if "dz_g3" in nui_names
        dz_g3 = params_dict["dz_g3"]
    else 
        dz_g3 = 0.0
    end
    
    if "dz_g4" in nui_names
        dz_g4 = params_dict["dz_g4"]
    else 
        dz_g4 = 0.0
    end

    if "dz_k0" in nui_names
        dz_k0 = params_dict["dz_k0"]
    else 
        dz_k0 = 0.0
    end
    
    if "dz_k1" in nui_names
        dz_k1 = params_dict["dz_k1"]
    else 
        dz_k1 = 0.0
    end
    
        if "dz_k2" in nui_names
        dz_k2 = params_dict["dz_k2"]
    else 
        dz_k2 = 0.0
    end
    
    if "dz_k3" in nui_names
        dz_k3 = params_dict["dz_k3"]
    else 
        dz_k3 = 0.0
    end

    if "m0" in nui_names
        m0 = params_dict["m0"]
    else 
        m0 = -1.0
    end
    
    if "m1" in nui_names
        m1 = params_dict["m1"]
    else 
        m1 = -1.0
    end
    
    if "m2" in nui_names
        m2 = params_dict["m2"]
    else 
        m2 = -1.0
    end
    
    if "m3" in nui_names
        m3 = params_dict["m3"]
    else 
        m3 = -1.0
    end
    
    if "A_IA" in nui_names
        A_IA = params_dict["A_IA"]
    else 
        A_IA = 0.0
    end
    
    if "alpha_IA" in nui_names
        alpha_IA = params_dict["alpha_IA"]
    else 
        alpha_IA = 0.0
    end
    
    nuisances = Dict("b0" => b0,
                     "b1" => b1,
                     "b2" => b2,
                     "b3" => b3,
                     "b4" => b4,
                     "dz_g0" => dz_g0,
                     "dz_g1" => dz_g1,
                     "dz_g2" => dz_g2,
                     "dz_g3" => dz_g3,
                     "dz_g4" => dz_g4,
                     "dz_k0" => dz_k0,
                     "dz_k1" => dz_k1,
                     "dz_k2" => dz_k2,
                     "dz_k3" => dz_k3,
                     "m0" => m0,
                     "m1" => m1,
                     "m2" => m2,
                     "m3" => m3,
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA)
    
    return nuisances
end

function Theory(cosmology, Nuisances, cls_meta, files)
    # OPT: move these loops outside the lkl
    tracers = []
    Nuisances = fill_NuisancePars(Nuisances)
    for tracer in cls_meta.tracers
        tracer_type = tracer[1]
        bin = tracer[2]
        nzs = files[string("nz_", tracer_type, bin)]
        nz = vec(nzs[2:2, :])
        zs = vec(nzs[1:1, :])
        
        if tracer_type == 1
            bias = Nuisances[string("b", bin)]
            dzi = Nuisances[string("dz_g", bin)]
            zs .=  zs .- dzi
            sel = zs .> 0.
            tracer = NumberCountsTracer(cosmology, zs[sel], nz[sel]; bias=bias)
        elseif tracer_type == 2
            mbias = Nuisances[string("m", bin)]
            dzi = Nuisances[string("dz_k", bin)]
            IA_params = [Nuisances["A_IA"], Nuisances["alpha_IA"]]
            zs .=  zs .- dzi
            sel = zs .> 0.
            tracer = WeakLensingTracer(cosmology, zs[sel], nz[sel];
                                       mbias=mbias, IA_params=IA_params)
        else
            print("Not implemented")
            trancer = nothing
        end
        push!(tracers, tracer)
    end
    npairs = length(cls_meta.pairs)
    Cls = []
    @inbounds for i in 1:npairs
        pair = cls_meta.pairs[i]
        ids = cls_meta.pairs_ids[i]
        ls = files[string("ls_", pair[1], pair[2], pair[3], pair[4])]
        tracer1 = tracers[ids[1]]
        tracer2 = tracers[ids[2]]
        Cl = [angularCℓ(cosmology, tracer1, tracer2, l) for l in ls]
        push!(Cls, Cl)
    end
    return Cls
end

function Theory_parallel(cosmology, Nuisances, cls_meta, files)
    # OPT: move these loops outside the lkl
    Nuisances = fill_NuisancePars(Nuisances)
    ntracers = length(cls_meta.tracers)
    tracers = Array{Any}(undef, ntracers)
    @inbounds Threads.@threads for i in 1:ntracers
        tracer = cls_meta.tracers[i]
        tracer_type = tracer[1]
        bin = tracer[2]
        nzs = files[string("nz_", tracer_type, bin)]
        nz = vec(nzs[2:2, :])
        zs = vec(nzs[1:1, :])
        
        if tracer_type == 1
            bias = Nuisances[string("b", bin)]
            dzi = Nuisances[string("dz_g", bin)]
            tracer = NumberCountsTracer(cosmology, zs .- dzi, nz; bias=bias)
        elseif tracer_type == 2
            mbias = Nuisances[string("m", bin)]
            dzi = Nuisances[string("dz_k", bin)]
            IA_params = [Nuisances["A_IA"], Nuisances["alpha_IA"]]
            tracer = WeakLensingTracer(cosmology, zs .- dzi, nz;
                                       mbias=mbias, IA_params=IA_params)
        else
            print("Not implemented")
            trancer = nothing
        end
        tracers[i] = tracer
        
    end

    npairs = length(cls_meta.pairs)
    Cls = Vector{Vector{Union{Real, ForwardDiff.Dual{Nothing, Float64, 2}}}}(undef, npairs)
    @inbounds Threads.@threads for i in 1:npairs
        pair = cls_meta.pairs[i]
        ids = cls_meta.pairs_ids[i]
        ls = files[string("ls_", pair[1], pair[2], pair[3], pair[4])]
        tracer1 = tracers[ids[1]]
        tracer2 = tracers[ids[2]]
        Cls[i] = angularCℓs(cosmology, tracer1, tracer2, ls)
    end
    return Cls 
end