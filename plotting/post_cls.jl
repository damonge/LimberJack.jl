using LimberJack
using GaussianProcess
using CSV
using DataFrames
using NPZ
using FITSIO
using PythonCall
using Interpolations
np = pyimport("numpy")

function get_cls_SD(chain)

    data_set = "SD"
    meta = np.load(string("../data/", data_set, "/", data_set, "_meta.npz"))
    files = npzread(string("../data/", data_set, "/", data_set, "_files.npz"))

    tracers_names = pyconvert(Vector{String}, meta["tracers"])
    pairs = pyconvert(Vector{Vector{String}}, meta["pairs"])
    idx = pyconvert(Vector{Int}, meta["idx"])
    data_vector = pyconvert(Vector{Float64}, meta["cls"])
    cov_tot = pyconvert(Matrix{Float64}, meta["cov"]);

    #KiDS priors
    Ωm = chain[!, "Ωm"]
    Ωb = chain[!, "Ωb"]
    h = chain[!, "h"]
    s8 = chain[!, "s8"]
    ns = chain[!, "ns"]

    A_IA = chain[!, "A_IA"]
    alpha_IA = chain[!, "alpha_IA"]

    eBOSS__0_0_b = chain[!, "eBOSS__0_0_b"]
    eBOSS__1_0_b = chain[!, "eBOSS__1_0_b"]

    DECALS__0_0_b = chain[!, "DECALS__0_0_b"]
    DECALS__1_0_b = chain[!, "DECALS__1_0_b"]
    DECALS__2_0_b = chain[!, "DECALS__2_0_b"]
    DECALS__3_0_b = chain[!, "DECALS__3_0_b"]
    DECALS__0_0_dz = chain[!, "DECALS__0_0_dz"]
    DECALS__1_0_dz = chain[!, "DECALS__1_0_dz"]
    DECALS__2_0_dz = chain[!, "DECALS__2_0_dz"]
    DECALS__3_0_dz = chain[!, "DECALS__3_0_dz"]

    KiDS1000__0_e_dz = chain[!, "KiDS1000__0_e_dz"]
    KiDS1000__1_e_dz = chain[!, "KiDS1000__1_e_dz"]
    KiDS1000__2_e_dz = chain[!, "KiDS1000__2_e_dz"]
    KiDS1000__3_e_dz = chain[!, "KiDS1000__3_e_dz"]
    KiDS1000__4_e_dz = chain[!, "KiDS1000__4_e_dz"]
    KiDS1000__0_e_m = chain[!, "KiDS1000__0_e_m"]
    KiDS1000__1_e_m = chain[!, "KiDS1000__1_e_m"]
    KiDS1000__2_e_m = chain[!, "KiDS1000__2_e_m"]
    KiDS1000__3_e_m = chain[!, "KiDS1000__3_e_m"]
    KiDS1000__4_e_m = chain[!, "KiDS1000__4_e_m"]

    cls = zeros(Float64, 610, length(h))
    for i in 1:length(h)
        nuisances = Dict("A_IA" => A_IA[i],
                         "alpha_IA" => alpha_IA[i],

                         "eBOSS__0_0_b" => eBOSS__0_0_b[i],
                         "eBOSS__1_0_b" => eBOSS__1_0_b[i],

                         "DECALS__0_0_b" => DECALS__0_0_b[i],
                         "DECALS__1_0_b" => DECALS__1_0_b[i],
                         "DECALS__2_0_b" => DECALS__2_0_b[i],
                         "DECALS__3_0_b" => DECALS__3_0_b[i],
                         "DECALS__0_0_dz" => DECALS__0_0_dz[i],
                         "DECALS__1_0_dz" => DECALS__1_0_dz[i],
                         "DECALS__2_0_dz" => DECALS__2_0_dz[i],
                         "DECALS__3_0_dz" => DECALS__3_0_dz[i],

                         "KiDS1000__0_e_dz" => KiDS1000__0_e_dz[i],
                         "KiDS1000__1_e_dz" => KiDS1000__1_e_dz[i],
                         "KiDS1000__2_e_dz" => KiDS1000__2_e_dz[i],
                         "KiDS1000__3_e_dz" => KiDS1000__3_e_dz[i],
                         "KiDS1000__4_e_dz" => KiDS1000__4_e_dz[i],
                         "KiDS1000__0_e_m" => KiDS1000__0_e_m[i],
                         "KiDS1000__1_e_m" => KiDS1000__1_e_m[i],
                         "KiDS1000__2_e_m" => KiDS1000__2_e_m[i],
                         "KiDS1000__3_e_m" => KiDS1000__3_e_m[i],
                         "KiDS1000__4_e_m" => KiDS1000__4_e_m[i])


        cosmology = LimberJack.Cosmology(Ωm[i], Ωb[i], h[i], ns[i], s8[i],
                                         tk_mode="emulator",
                                         Pk_mode="Halofit")
        cls[:, i] = Theory(cosmology, tracers_names, pairs,
                        idx, files; Nuisances=nuisances)
    end

    return cls
end;

function post_cls_SD(fol_path)
    i = 1
    while isfile(string(fol_path, "chain_", i, ".csv"))
        file_path = string(fol_path, "chain_", i, ".csv")
        chain = CSV.read(file_path, DataFrame)
        cls = get_cls_SD(chain)
        println(i)
        npzwrite(string(fol_path, "cls_", i ,".npz"), cls)
        i += 1
    end
end

