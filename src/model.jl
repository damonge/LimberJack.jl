function get_theory(cosmology, datas; path=path)
    tracers_names = []
    for data in datas
        tracer1 = string(data.tracer1, data.bin1)
        tracer2 = string(data.tracer2, data.bin2)
        push!(tracers_names, tracer1)
        push!(tracers_names, tracer2)
    end
    tracers_names = unique(tracers_names)
    
    tracers = []
    for tracer_name in tracers_names
        Nzs = Nz(parse(Int, tracer_name[6]), path=path)
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
    predictions = []
    for data in datas
        tracer_name1 = string(data.tracer1, data.bin1)
        tracer_name2 = string(data.tracer2, data.bin2)
        tracer1_id = findall(x->x==tracer_name1, tracers_names)
        tracer2_id = findall(x->x==tracer_name2, tracers_names)
        tracer1 = tracers[tracer1_id][1]
        tracer2 = tracers[tracer2_id][1]
        prediction = [angularCℓ(cosmology, tracer1, tracer2, ℓ) for ℓ in data.ell]
        push!(predictions, prediction)
    end
    predictions = vcat(predictions...)
end