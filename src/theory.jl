struct cls_meta
    tracers::Vector
    pairs::Vector
    pairs_ids::Vector
end

function cls_meta(file)
    tracers = [c[:] for c in eachrow(file["tracers"])]
    pairs = [c[:] for c in eachrow(file["pairs"])]
    pairs_ids = [c[:] for c in eachrow(file["pairs_ids"])]
    cls_meta(tracers, pairs, pairs_ids)
end

struct Theory
    tracers
    Cls
end

function Theory(cosmology, Nuisances, cls_meta, files)
    # OPT: move these loops outside the lkl
    ntracers = length(cls_meta.tracers)
    tracers = Array{Any}(undef, ntracers ) 
    for i in 1:ntracers
        tracer_info = cls_meta.tracers[i]
        tracer_type = tracer_info[1]
        bin = tracer_info[2]
        nzs = files[string("nz_", tracer_type, bin)]
        zs = vec(nzs[1:1, :])
        nz = vec(nzs[2:2, :])
        if tracer_type == 1
            bias = string("b", bin)
            tracer = NumberCountsTracer(cosmology, zs, nz, Nuisances[bias])
        elseif tracer_type == 2
            tracer = WeakLensingTracer(cosmology, zs, nz)
        else
            print("Not implemented")
            trancer = nothing
        end
        tracers[i] = tracer
        #push!(tracers, tracer)
    end
    Cls = []
    for i in 1:length(cls_meta.pairs)
        pair = cls_meta.pairs[i]
        ids = cls_meta.pairs_ids[i]
        ls = files[string("ls_", pair[1], pair[2], pair[3], pair[4])]
        tracer1 = tracers[ids[1]]
        tracer2 = tracers[ids[2]]
        Cl = Array{Any}(undef, length(ls)) 
        Threads.@threads for i in 1:length(ls)
            Cl[i] = angularCℓ(cosmology, tracer1, tracer2, ls[i]) 
        end
        #Cl = [angularCℓ(cosmology, tracer1, tracer2, l) for l in ls])
        push!(Cls, Cl)
    end
    Cls = vcat(Cls...)
    Theory(tracers, Cls)
end