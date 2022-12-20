sacc = pyimport("sacc")
yaml = pyimport("yaml")
np = pyimport("numpy")


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
