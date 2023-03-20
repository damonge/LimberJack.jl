using Pkg
Pkg.activate("../../../MicroCanonicalHMC.jl/")

using Base.Threads
using LinearAlgebra
using Turing
using LimberJack
using DataFrames
using CSV
using YAML
using NPZ
using PythonCall
sacc = pyimport("sacc");

using MicroCanonicalHMC

sacc_path = "../../data/FD/cls_FD_covG.fits"
yaml_path = "../../data/DECALS/DECALS.yml"
sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)
meta, files = make_data(sacc_file, yaml_file)

cov = meta.cov
data = meta.data

@model function model(data; files=files)
    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ TruncatedNormal(72, 5, 0.64, 0.82)
    s8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)

    DECALS__0_b ~ Uniform(0.8, 3.0)
    DECALS__1_b ~ Uniform(0.8, 3.0)
    DECALS__2_b ~ Uniform(0.8, 3.0)
    DECALS__3_b ~ Uniform(0.8, 3.0)
    DECALS__0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DECALS__1_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DECALS__2_dz ~ TruncatedNormal(0.0, 0.006, -0.2, 0.2)
    DECALS__3_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)

    nuisances = Dict("DECALS__0_b" => DECALS__0_b,
                     "DECALS__1_b" => DECALS__1_b,
                     "DECALS__2_b" => DECALS__2_b,
                     "DECALS__3_b" => DECALS__3_b,
                     "DECALS__0_dz" => DECALS__0_dz,
                     "DECALS__1_dz" => DECALS__1_dz,
                     "DECALS__2_dz" => DECALS__2_dz,
                     "DECALS__3_dz" => DECALS__3_dz)

    cosmology = Cosmology(Ωm, Ωb, h, ns, s8,
                          tk_mode="EisHu",
                          Pk_mode="Halofit")

    theory = Theory_st(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(theory, cov)
end

d = 13
eps = 0.07
L = round(sqrt(d), digits=2)
sigma = ones(d)

stats_model = model(data)
target = TuringTarget(stats_model)
spl = MCHMC(eps, L; sigma=sigma)

# Start sampling.
folpath = "../../chains/MCHMC"
folname = string("DECALS_eps_", eps, "_L_", L)
folname = joinpath(folpath, folname)

if isdir(folname)
    println("Found existing file")
else
    mkdir(folname)
    println(string("Created new folder ", folname))
end

nchains = nthreads()

@threads for i in 1:nchains    
    file_name = string("chain_", i)
    samples= Sample(spl, target, 10_000;
                    burn_in=200, fol_name=folname, file_name=file_name, dialog=true)
end        