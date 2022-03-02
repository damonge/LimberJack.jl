function get_tracers(datas)
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
    return tracers
end