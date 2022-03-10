using Turing
using LimberJack
using CSV
using NPZ
using FITSIO
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

iterations = 500
step_size = 0.005
samples_per_step = 10
cores = 4

# Start sampling.
folname = string("DES_gc2wl3_", "stpsz_", step_size, "_smpls_", samples_per_step)
if isdir(folname)
    println("Folder already exists")
    if isfile(joinpath(folname, "chain.jls"))
        println("Restarting from past chain")
        past_chain = read(joinpath(folname, "chain.jls"), Chains)
        new_chain = sample(model(des_data), HMC(step_size, samples_per_step), iterations,
                           progress=true; save_state=true, resume_from=past_chain)
    end
else
    mkdir(folname)
    println("Created new folder")
    new_chain = sample(model(des_data), HMC(step_size, samples_per_step),
                iterations, progress=true; save_state=true)
end

info = describe(new_chain)[1]
fname_info = string("info.csv")
CSV.write(joinpath(folname, fname_info), info)


fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chain)
    
fname_csv = string("chain.csv")
CSV.write(joinpath(folname, fname_csv), new_chain)
