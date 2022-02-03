using Turing
using Random
using ForwardDiff
using LimberJack
using CSV
using NPZ
using FITSIO
using Plots
using LinearAlgebra

des_data = npzread("/home/jaime/PhD/LimberJack.jl/data/cl_DESgc__2_DESgc__2.npz")["cl"][1:39]
des_ell = npzread("/home/jaime/PhD/LimberJack.jl/data/cl_DESgc__2_DESgc__2.npz")["ell"]
des_ell = [Int(floor(l)) for l in des_ell]
des_cov = npzread("/home/jaime/PhD/LimberJack.jl/data/cov_DESgc__2_DESgc__2_DESgc__2_DESgc__2.npz")["cov"]
des_cov = Symmetric(Hermitian(des_cov))
des_err = (view(des_cov, diagind(des_cov)))[1:39].^0.5
des_nzs = FITS("/home/jaime/PhD/LimberJack.jl/data/y1_redshift_distributions_v1.fits")
des_nz = read(des_nzs["nz_source_mcal"], "BIN2")
des_zs = read(des_nzs["nz_source_mcal"], "Z_MID");

@model function model(data)
    Ωm ~ Uniform(0.2, 0.3)
    Ωb ~ Uniform(0.0, 0.1)
    h ~ Uniform(0.6, 0.8)
    ns ~ Uniform(0.9, 1.0)
    s8 ~ Uniform(0.7, 1.0)
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8, tk_mode="EisHu", Pk_mode="Halofit")
    #cosmology = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.8, tk_mode="EisHu")
    tg = NumberCountsTracer(cosmology, des_zs, des_nz, 2.)
    #ts = WeakLensingTracer(cosmology, des_zs, des_nz)
    predictions = [angularCℓ(cosmology, tg, tg, ℓ) for ℓ in des_ell]
    data ~ MvNormal(predictions, des_cov)
end;

iterations = 1000
ϵ = 0.05
τ = 10

# Start sampling.
chain = sample(model(des_data), HMC(ϵ, τ), iterations, progress=true)
CSV.write("Clgg_des_test_chain.csv", chain)
