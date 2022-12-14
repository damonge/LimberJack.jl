println(Threads.nthreads())

using LinearAlgebra
using Turing
using LimberJack
using PythonCall
using NPZ
using BenchmarkTools;

np = pyimport("numpy");

meta = np.load("../data/DESY1/gcgc_gcwl_wlwl_meta.npz")
tracers_names = pyconvert(Vector{String}, meta["tracers"])
pairs = pyconvert(Vector{Vector{String}}, meta["pairs"])
pairs_ids = pyconvert(Vector{Vector{Int}}, meta["pairs_ids"])
idx = pyconvert(Vector{Int}, meta["idx"])
data_vector = pyconvert(Vector{Float64}, meta["cls"])
cov_tot = pyconvert(Matrix{Float64}, meta["cov"])
inv_cov_tot = pyconvert(Matrix{Float64}, meta["inv_cov"]);

files = npzread("../data/DESY1/gcgc_gcwl_wlwl_files.npz");

#path = "/home/jaime/PhD/LimberJack.jl/chains/carlos_chains/cl_cross_corr_v3_DES_K1000_all_mag_correctMag/"
#pars = np.loadtxt(string(path, "cl_cross_corr_v3_DES_K1000_all_mag_correctMag.bestfit"))
#pars = pyconvert(Vector{Float64}, pars);

#cls_carlos_ND = np.load(string(path, "cl_cross_corr_bestfit_info.npz"))
#cls_carlos_SD = np.load(string(path, "cl_cross_corr_bestfit_info_copy.npz"))
#cls_carlos_FD = np.append(cls_carlos_SD["cls"], cls_carlos_ND["cls"])
#cls_carlos_FD = pyconvert(Vector{Float64}, cls_carlos_FD)
#carlos_chi = pyconvert(Float64, - (cls_carlos_ND["chi2"] +  cls_carlos_SD["chi2"]));

Ωm = 0.30 #pars[1] + pars[2]
s8 = 0.81 #pars[46]
function make_cls(; Ωm=Ωm, s8=s8)
    #nuisances = Dict("DESgc__0_0_b" => pars[6],
    #                 "DESgc__1_0_b" => pars[7],
    #                 "DESgc__2_0_b" => pars[8],
    #                 "DESgc__3_0_b" => pars[9],
    #                 "DESgc__4_0_b" => pars[10],
    #                 "DESgc__0_0_dz" => pars[11],
    #                 "DESgc__1_0_dz" => pars[12],
    #                 "DESgc__2_0_dz" => pars[13],
    #                 "DESgc__3_0_dz" => pars[14],
    #                 "DESgc__4_0_dz" => pars[15],
    #    
    #                 "A_IA" => pars[16],
    #                 "alpha_IA" => pars[17],
    #    
    #                 "DESwl__0_e_m" => pars[18],
    #                 "DESwl__1_e_m" => pars[19],
    #                 "DESwl__2_e_m" => pars[20],
    #                 "DESwl__3_e_m" => pars[21],
    #                 "DESwl__0_e_dz" => pars[22],
    #                 "DESwl__1_e_dz" => pars[23],
    #                 "DESwl__2_e_dz" => pars[24],
    #                 "DESwl__3_e_dz" => pars[25],
    #    
    #                 "eBOSS__0_0_b" => pars[26],
    #                 "eBOSS__1_0_b" => pars[27],
    #
    #                 "DECALS__0_0_b" => pars[28],
    #                 "DECALS__1_0_b" => pars[29],
    #                 "DECALS__2_0_b" => pars[30],
    #                 "DECALS__3_0_b" => pars[31],
    #                 "DECALS__0_0_dz" => pars[32],
    #                 "DECALS__1_0_dz" => pars[33],
    #                 "DECALS__2_0_dz" => pars[34],
    #                 "DECALS__3_0_dz" => pars[35],
    #                
    #                 "KiDS1000__0_e_m" => pars[36],
    #                 "KiDS1000__1_e_m" => pars[37],
    #                 "KiDS1000__2_e_m" => pars[38],
    #                 "KiDS1000__3_e_m" => pars[39],
    #                 "KiDS1000__4_e_m" => pars[40],
    #                 "KiDS1000__0_e_dz" => pars[41],
    #                 "KiDS1000__1_e_dz" => pars[42],
    #                 "KiDS1000__2_e_dz" => pars[43],
    #                 "KiDS1000__3_e_dz" => pars[44],
    #                 "KiDS1000__4_e_dz" => pars[45])

    Ωb = 0.05 #pars[1]
    h = 0.67 #pars[5]
    ns = 0.96 #pars[4]
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8, 
                                     tk_mode="emulator",
                                     Pk_mode="Halofit")

    return Theory(cosmology,
                  tracers_names, pairs,
                  idx, files)

end

println(@benchmark make_cls())
