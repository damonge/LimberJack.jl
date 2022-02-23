using Turing
using Random
using ForwardDiff
using LimberJack
using CSV
using NPZ
using FITSIO

ell = npzread("/mnt/zfsusers/jaimerz/PhD/LimberJack.jl/data/cl_DESgc__2_DESwl__3.npz")["ell"]
ell = [Int(floor(l)) for l in ell]
nzs = FITS("/mnt/zfsusers/jaimerz/PhD/LimberJack.jl/data/y1_redshift_distributions_v1.fits")
nz = read(des_nzs["nz_source_mcal"], "BIN2")
zs = read(des_nzs["nz_source_mcal"], "Z_MID")

true_cosmology = LimberJack.Cosmology(0.25, 0.05, 0.67, 0.96, 0.81,
                                      tk_mode="EisHu");
tg = NumberCountsTracer(true_cosmology, zs, nz, 2.)
ts = WeakLensingTracer(true_cosmology, zs, nz)
true_data = [angularCℓ(true_cosmology, tg, ts, ℓ) for ℓ in ell]
data = true_data #+ 0.1 * true_data .* rand(length(true_data)) 

@model function model(data)
    Ωm ~ Uniform(0.1, 0.4)
    s8 ~ Uniform(0.5, 1.0)
    h ~ Uniform(0.5, 0.9)
    cosmology = LimberJack.Cosmology(Ωm, 0.05, h, 0.96, s8,
                                    tk_mode="EisHu")
    tg = NumberCountsTracer(cosmology, zs, nz, 2.)
    ts = WeakLensingTracer(cosmology, zs, nz)
    predictions = [angularCℓ(cosmology, tg, ts, ℓ) for ℓ in ell]
    data ~ MvNormal(predictions, 0.01.*data)
end;

iterations = 1000
burn = 1000
TAP = 0.65

# Start sampling.
chain = sample(model(data), NUTS(burn, TAP), iterations, progress=true)
CSV.write("Cl_EisHu_chain.csv", chain)
