function get_nzs(nz_path, tracer_name)
    nzs = npzread(string(nz_path, "nz_", tracer_name, ".npz"))
    zs = nzs["z"]
    nz = nzs["dndz"]
    cov = get(nzs, "cov", zeros(length(zs)))
    return zs, nz, cov
end

struct meta
    namess
    pairss
    types
    cls
    idx
    cov 
    inv_cov
end

function get_type(sacc_file, tracer_name)
    return sacc_file.tracers[tracer_name].quantity
end

function get_spin(sacc_file, tracer_name)
    tt = string(sacc_file.tracers[tracer_name].quantity)
    if tt == "galaxy_shear"
        spin = "e"
    elseif tt == "galaxy_density"
        spin = "0"
    elseif tt == "cmb_convergence"
        spin = "0"
    end
    return spin
end 

function get_cl_name(s, t1, t2)
    spin1 = get_spin(s, t1)
    spin2 = get_spin(s, t2)
    cl_name = string("cl_", spin1 , spin2)
    if cl_name == "cl_e0"
        cl_name = "cl_0e"
    end
    return cl_name 
end

function apply_scale_cuts(s, config)
    indices = Vector{Int}([])
    for cl in config["order"]
        t1, t2 = cl["tracers"]
        lmin, lmax = cl["ell_cuts"]
        cl_name = get_cl_name(s, t1, t2)
        ind = s.indices(cl_name, (t1, t2),
                        ell__gt=lmin, ell__lt=lmax)
        append!(indices, pyconvert(Vector{Int}, ind))
    end
    s.keep_indices(indices)
    return s
end

function make_data(sacc_path, yaml_path; nzs_path="")
    #load
    config = open(yaml_path, "r") do f
        yaml.safe_load(f)
    end
    s_uncut = sacc.Sacc().load_fits(sacc_path)
    
    #cut
    s = apply_scale_cuts(s_uncut, config)
        
    #build quantities of interest
    cls = Vector{Float64}([])
    ls = []
    indices = Vector{Int}([])
    pairss = []
    for cl in config["order"]
        t1, t2 = cl["tracers"]
        cl_name = get_cl_name(s, t1, t2)
        l, c_ell, ind = s_cut.get_ell_cl(cl_name, string(t1), string(t2),
                                         return_cov=false, return_ind=true)
        append!(indices, pyconvert(Vector{Int}, ind))
        append!(cls, pyconvert(Vector{Float64}, c_ell))
        push!(ls, pyconvert(Vector{Float64}, l))
        push!(pairss, pyconvert(Vector{String}, [t1, t2]))
    end
    namess = unique(vcat(pairs...))
    cov = pyconvert(Vector{Vector{Float64}}, s.covariance.dense)
    cov = permutedims(hcat(cov...))[indices.+1, :][:, indices.+1]
    cov = Hermitian(cov)
    inv_cov = inv(cov)
    lengths = [length(l) for l in ls]
    lengths = vcat([0], lengths)
    idx  = cumsum(lengths)
    types = [get_type(s, name) for name in namess]
    
    # build struct
    metaa = meta(namess, pairss, types, cls, idx,
                cov, inv_cov)
    
    # Initialize
    files = Dict{String}{Vector}()
    
    # Load in l's
    for (pair, l) in zip(pairs, ls)
        t1, t2 = pair
        println(t1, " ", t2, " ", length(l))
        merge!(files, Dict(string("ls_", t1, "_", t2)=> l))
    end
    
    # Load in nz's
    for (name, tracer) in s.tracers.items()
        if string(name) in namess
            if nzs_path == ""
                z=pyconvert(Vector{Float64}, tracer.z)
                nz=pyconvert(Vector{Float64}, tracer.nz)
                merge!(files, Dict(string("nz_", name)=>[z, nz]))
            else
                nzs = np.load(nzs_path+string("nz_", name)+".npz")
                z = nzs["z"]
                dndz = nzs["dndz"]
                merge!(files, Dict(string("nz_", name)=>[z, nz]))
            end
        end
    end
    
    return metaa, files
end 

function Theory(cosmology::Cosmology,
                names, types, pairs,
                idx, files;
                Nuisances=Dict())
    
    nui_type =  eltype(valtype(Nuisances))
    if !(nui_type <: Float64) & (nui_type != Any)
        if nui_type != Real
            cosmology.settings.cosmo_type = nui_type
        end
    end
    
    tracers =  Dict{String}{Tracer}()
    ntracers = length(names)
    @inbounds for i in 1:ntracers
        name = names[i]
        t_type = types[i]
        if t_type == "galaxy_density"
            nzs = files[string("nz_", name)]
            nzs = [nzs[i,:] for i in 1:size(nzs,1)]
            zs_mean, nz_mean = nzs[1], nzs[2]

            b = get(Nuisances, string(name, "_", "b"), 1.0)
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            tracer = NumberCountsTracer(cosmology, zs, nz;
                                        b=b)
        elseif t_type == "galaxy_shear"
            nzs = files[string("nz_", name)]
            nzs = [nzs[i,:] for i in 1:size(nzs,1)]
            zs_mean, nz_mean = nzs[1], nzs[2]

            mb = get(Nuisances, string(name, "_", "mb"), 0.0)
            IA_params = [get(Nuisances, "A_IA", 0.0),
                         get(Nuisances, "alpha_IA", 0.0)]
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            tracer = WeakLensingTracer(cosmology, zs, nz;
                                       mb=mb, IA_params=IA_params)
            
        elseif t_type == "cmb_convergence"
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
