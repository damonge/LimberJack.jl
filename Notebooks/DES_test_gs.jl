using Turing
using Random
using ForwardDiff
using LimberJack
using CSV
using NPZ
using FITSIO
using Plots
using LinearAlgebra

des_data = npzread("/mnt/zfsusers/jaimerz/PhD/LimberJack.jl/data/cl_DESgc__2_DESwl__3.npz")["cl"]
des_data = transpose(des_data)[1:39]
des_ell = npzread("/mnt/zfsusers/jaimerz/PhD/LimberJack.jl/data/cl_DESgc__2_DESwl__3.npz")["ell"]
des_ell = [Int(floor(l)) for l in des_ell]
des_cov = npzread("/mnt/zfsusers/jaimerz/PhD/LimberJack.jl/data/cov_DESgc__2_DESwl__3_DESgc__2_DESwl__3.npz")["cov"]
des_cov = des_cov[1:39, 1:39]
des_cov = Symmetric(Hermitian(des_cov))
des_err = (view(des_cov, diagind(des_cov)))[1:39].^0.5
des_nzs = FITS("/mnt/zfsusers/jaimerz/PhD/LimberJack.jl/data/y1_redshift_distributions_v1.fits")
des_nz = read(des_nzs["nz_source_mcal"], "BIN2")
des_zs = read(des_nzs["nz_source_mcal"], "Z_MID")

@model function model(data)
    Ωm ~ Uniform(0.2, 0.3)
    h ~ Uniform(0.6, 0.8)
    s8 ~ Uniform(0.7, 1.0)
    cosmology = LimberJack.Cosmology(Ωm, 0.05, h, 0.96, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    tg = NumberCountsTracer(cosmology, des_zs, des_nz, 2.)
    ts = WeakLensingTracer(cosmology, des_zs, des_nz)
    predictions = [angularCℓ(cosmology, tg, ts, ℓ) for ℓ in des_ell]
    data ~ MvNormal(predictions, des_cov)
end;

iterations = 1000
burn = 1000
TAP = 0.65

# Start sampling.
chain = sample(model(data), NUTS(burn, TAP), iterations, progress=true)
CSV.write("Cl_DES_test_chain.csv", chain)
