abstract type Nz_typ end
abstract type Data_typ end

struct Data <: Data_typ
    cl::Vector
    ell::Vector
    cov::Matrix
    nz1::Nz_type
    nz2::Nz_type
end

function Data(tracer1, tracer2, bin1, bin2; path="LimberJack.jl/data")
    cl_fname = string(path, "cl_", tracer1, "__", bin1, "_", 
                     tracer2, "__", bin2, ".npz")
    cov_fname = string(path, "cov_", tracer1, "__", bin1, "_",
                      tracer2, "__", bin2, "_",
                      tracer1, "__", bin1, "_",
                      tracer2, "__", bin2, ".npz")
    cl_file = npzread(cl_fname)
    cov_file = npzread(cov_fname)
    ell = cl_file["ell"]
    ell = [Int(floor(l)) for l in des_ell]
    cl = cl_file["cl"]
    cl = transpose(cl)[1:length(ell)]
    cov = npzread(cov_fname)["cov"]
    cov = des_cov[1:length(ell), 1:length(ell)]
    cov = Symmetric(Hermitian(cov))
    nz1 = Nz(bin1)
    nz2 = Nz(bin2)
    Bin(cl, ell, cov, nz1, nz2)
end

struct Nz <: Nz_type
    nz::Vector
    zs::Vector
end

function Nz(bin_number)
    nz_fname = "y1_redshift_distributions_v1.fits"
    bin_names = Dict{1 => "BIN1",
                     2 => "BIN2",
                     3 => "BIN3",
                     4 => "BIN4",
                     5 => "BIN5"}
    bin_name = bin_names[bin_number]
    nzs = FITS(nzs_fname)
    nz = read(nzs["nz_source_mcal"], bin_name)
    zs = read(nzs["nz_source_mcal"], "Z_MID")
    Bin(nz, zs)
end
