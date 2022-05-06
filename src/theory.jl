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

function Theory(cosmology, Nuisances, cls_meta, files)
    # OPT: move these loops outside the lkl
    tracers = []
    for tracer in cls_meta.tracers
        tracer_type = tracer[1]
        bin = tracer[2]
        nzs = files[string("nz_", tracer_type, bin)]
        nz = vec(nzs[2:2, :])
        
        if tracer_type == 1
            bias = string("b", bin)
            dzi = string("dz_g", bin)
            zs = @.(vec(nzs[1:1, :]) - Nuisances[dzi])
            tracer = NumberCountsTracer(cosmology, zs, nz, Nuisances[bias])
        elseif tracer_type == 2
            dzi = string("dz_k", bin)
            zs = @.(vec(nzs[1:1, :]) - Nuisances[dzi])
            A_IA = string("A_IA")
            alpha_IA = string("alpha_IA")
            IA = [A_IA, alpha_IA]
            tracer = WeakLensingTracer(cosmology, zs, nz, IA)
        else
            print("Not implemented")
            trancer = nothing
        end
        push!(tracers, tracer)
    end
    #Cls = Vector{Real}[]
    npairs = length(cls_meta.pairs)
    #Cls = Vector{Vector}(undef, npairs)
    Cls = []
    #@inbounds Threads.@threads for i in 1:npairs
    @inbounds for i in 1:npairs
        pair = cls_meta.pairs[i]
        ids = cls_meta.pairs_ids[i]
        ls = files[string("ls_", pair[1], pair[2], pair[3], pair[4])]
        tracer1 = tracers[ids[1]]
        tracer2 = tracers[ids[2]]
        #Cl = zeros(length(ls))
        #@inbounds for i in 1:length(ls)
        #    Cl[i] = angularCℓ(cosmology, tracer1, tracer2, ls[i]) 
        #end
        #Cls[i] = [angularCℓ(cosmology, tracer1, tracer2, l) for l in ls::Vector{Float64}]
        Cl = [angularCℓ(cosmology, tracer1, tracer2, l) for l in ls::Vector{Float64}]
        push!(Cls, Cl)
    end
    return Cls
    
end
