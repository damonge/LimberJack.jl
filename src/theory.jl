struct Theory
    Tracers
    Cls
end

function Theory(cosmology, cls_meta; path=path)
    ell = cls_meta.ell
    tracers_names = cls_meta.tracers_names
    cls_names = cls_meta.cls_names
    tracers = []
    for tracer_name in tracers_names
        Nzs = Nz(parse(Int, tracer_name[8]), path=path)
        if occursin("gc", tracer_name)
            tracer = NumberCountsTracer(cosmology, Nzs.zs, Nzs.nz, 2.)
        elseif occursin("wl", tracer_name)
            tracer = WeakLensingTracer(cosmology, Nzs.zs, Nzs.nz)
        else
            print("Not implemented")
            trancer = nothing
        end
        push!(tracers, tracer)
    end
    Cls = []
    for cls_name in cls_names
        tracer_name1 = cls_name[1:8]
        tracer_name2 = cls_name[10:17]
        tracer1_id = findall(x->x==tracer_name1, tracers_names)
        tracer2_id = findall(x->x==tracer_name2, tracers_names)
        tracer1 = tracers[tracer1_id][1]
        tracer2 = tracers[tracer2_id][1]
        Cl = [angularCℓ(cosmology, tracer1, tracer2, ℓ) for ℓ in ell]
        push!(Cls, Cl)
    end
    Cls = vcat(Cls...)
    Theory(tracers, Cls)
end