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
    cl_path::String
    cov_path::String
end

function Data(tracer1, tracer2, bin1, bin2;
              cl_path="data",
              cov_path="data",
              nz_path="data")
    cl_fname = string("cl_", tracer1, "__", bin1, "_", 
                     tracer2, "__", bin2, ".npz")
    cov_fname = string("cov_", tracer1, "__", bin1, "_",
                      tracer2, "__", bin2, "_",
                      tracer1, "__", bin1, "_",
                      tracer2, "__", bin2, ".npz")
    cl_file = npzread(joinpath(cl_path, cl_fname))
    cov_file = npzread(joinpath(cov_path, cov_fname))
    ell = cl_file["ell"]
    ell = [Int(floor(l)) for l in ell]
    cl = cl_file["cl"]
    cl = transpose(cl)[1:length(ell)]
    cov = cov_file["cov"]
    cov = cov[1:length(ell), 1:length(ell)]
    cov = Symmetric(Hermitian(cov))
    Data(tracer1, tracer2, bin1, bin2,
         cl, ell, cov,
         cl_path, cov_path)
end

struct Nz <: Nz_typ
    nz
    zs
end

function Nz(bin_number; path="../data/DESY1_cls")
    nzs_fname = string("y1_redshift_distributions_v1.fits")
    bin_name = "BIN$bin_number"
    nzs = FITS(joinpath(path, nzs_fname))
    nz = read(nzs["nz_lens"], bin_name)
    zs = read(nzs["nz_lens"], "Z_MID")
    Nz(nz, zs)
end

struct Cls_meta
    cls_names
    cov_names 
    data_vector 
    cov_tot
    tracers_names
    ell 
end

function Cls_meta(datas; covs_path="data")
    # Assume the same ell range for everything
    ell = datas[1].ell
    cls_names = [string(data.tracer1, "__", data.bin1, "_", 
                    data.tracer2, "__", data.bin2) 
                 for data in datas]
    cov_names = [string(cl_i, "_", cl_j) for cl_i in cls_names for cl_j in cls_names]
    data_vector = vcat([data.cl for data in datas]...)
    covs_fnames = [string("cov_", cov_fname, ".npz") for cov_fname in cov_names]
    covs = []
    len = length(datas[1].ell)
    for cov_fname in covs_fnames
        if isfile(joinpath(covs_path, cov_fname))
            cov = npzread(joinpath(covs_path, cov_fname))["cov"]
            cov = cov[1:len, 1:len]
            push!(covs, cov)
        else 
            push!(covs, zeros(len,len))
        end
    end
    dims = length(cls_names)
    
    cov_tot = zeros(len*dims,len*dims)
    k = 0
    for j in 1:dims
        for i in 1:dims
            k = 1+k
            for l in 1:len
                cov_tot[(len*(j-1))+l, (len*(i-1))+1:1:i*len] = covs[k][l, 1:len]
            end
        end
    end
    cov_tot = Symmetric(Hermitian(Matrix(cov_tot)))
    
    tracers_names = Vector{String}()
    for data in datas
        tracer1 = string(data.tracer1, "__", data.bin1)
        tracer2 = string(data.tracer2, "__", data.bin2)
        push!(tracers_names, tracer1)
        push!(tracers_names, tracer2)
    end
    tracers_names = unique(tracers_names)
    
    Cls_meta(cls_names, cov_names, data_vector, cov_tot, tracers_names, ell)
    
end
