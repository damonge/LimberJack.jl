abstract type Nz_typ end
abstract type Data_typ end

struct Data <: Data_typ
    tracer1::String
    tracer2::String
    bin1::Number
    bin2::Number
    cl::Vector
    ell::Vector
    cov::Matrix
    nz1::Nz_typ
    nz2::Nz_typ
    path::String
end

function Data(tracer1, tracer2, bin1, bin2; path="LimberJack.jl/data/")
    cl_fname = string(path, "cl_", tracer1, "__", bin1, "_", 
                     tracer2, "__", bin2, ".npz")
    cov_fname = string(path, "cov_", tracer1, "__", bin1, "_",
                      tracer2, "__", bin2, "_",
                      tracer1, "__", bin1, "_",
                      tracer2, "__", bin2, ".npz")
    cl_file = npzread(cl_fname)
    cov_file = npzread(cov_fname)
    ell = cl_file["ell"]
    ell = [Int(floor(l)) for l in ell]
    cl = cl_file["cl"]
    cl = transpose(cl)[1:length(ell)]
    cov = npzread(cov_fname)["cov"]
    cov = cov[1:length(ell), 1:length(ell)]
    cov = Symmetric(Hermitian(cov))
    nz1 = Nz(bin1; path=path)
    nz2 = Nz(bin2; path=path)
    Data(tracer1, tracer2, bin1, bin2, cl, ell, cov, nz1, nz2, path)
end

struct Nz <: Nz_typ
    nz
    zs
end

function Nz(bin_number; path="LimberJack.jl/data/")
    nzs_fname = string(path, "y1_redshift_distributions_v1.fits")
    bin_name = "BIN$bin_number"
    nzs = FITS(nzs_fname)
    nz = read(nzs["nz_source_mcal"], bin_name)
    zs = read(nzs["nz_source_mcal"], "Z_MID")
    Nz(nz, zs)
end

struct Cls_meta
    cls_names
    cov_names 
    data_vector 
    cov_tot
end

function Cls_meta(datas; path="LimberJack.jl/data/")
    cls_names = [string(data.tracer1, "__", data.bin1, "_", 
                        data.tracer2, "__", data.bin2) 
                 for data in datas]
    cov_names = [string(cl_i, "_", cl_j) for cl_i in cls_names for cl_j in cls_names]
    data_vector = vcat([data.cl for data in datas]...)
    covs_fnames = [string(path, "cov_", cov_fname, ".npz") for cov_fname in cov_names]
    cov_tot = []
    for cov_fname in covs_fnames
        if isfile(cov_fname)
            cov = npzread(cov_fname)["cov"]
            cov = cov[1:39, 1:39]
            push!(cov_tot, cov)
        else 
            push!(cov_tot, zeros(39,39))
        end
    end
    cov_tot = reshape(vcat(cov_tot...), (78,78))
    Cls_meta(cls_names, cov_names, data_vector, cov_tot)
    
end
