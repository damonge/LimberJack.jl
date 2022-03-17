using Turing
using Random
using ForwardDiff
using LimberJack
using CSV
using NPZ
using FITSIO
using LinearAlgebra

ell = npzread("../data/DESY1_cls/DESgc_DESwl/cl_DESgc__2_DESwl__3.npz")["ell"]
ell = [Int(floor(l)) for l in ell]
nzs = FITS("../data/DESY1_cls/y1_redshift_distributions_v1.fits")
nz1 = read(nzs["nz_lens"], "BIN1")
nz2 = read(nzs["nz_lens"], "BIN2")
nz3 = read(nzs["nz_lens"], "BIN3")
nz4 = read(nzs["nz_lens"], "BIN4")
nz5 = read(nzs["nz_lens"], "BIN5")
zs = read(nzs["nz_lens"], "Z_MID");

true_cosmology = LimberJack.Cosmology(0.25, 0.05, 0.67, 0.96, 0.81,
                                      tk_mode="EisHu", Pk_mode="Halofit");
                                      tk_mode="EisHu", Pk_mode="Halofit");
tg1 = NumberCountsTracer(true_cosmology, zs, nz1, 2.)
tg2 = NumberCountsTracer(true_cosmology, zs, nz2, 2.)
tg3 = NumberCountsTracer(true_cosmology, zs, nz3, 2.)
tg4 = NumberCountsTracer(true_cosmology, zs, nz4, 2.)
tg5 = NumberCountsTracer(true_cosmology, zs, nz5, 2.)
ts1 = WeakLensingTracer(true_cosmology, zs, nz1)
ts2 = WeakLensingTracer(true_cosmology, zs, nz2)
ts3 = WeakLensingTracer(true_cosmology, zs, nz3)
ts4 = WeakLensingTracer(true_cosmology, zs, nz4)
ts5 = WeakLensingTracer(true_cosmology, zs, nz5)
true_gg11 = [angularCℓ(true_cosmology, tg1, tg1, ℓ) for ℓ in ell]
true_gg22 = [angularCℓ(true_cosmology, tg2, tg2, ℓ) for ℓ in ell]
true_gg33 = [angularCℓ(true_cosmology, tg3, tg3, ℓ) for ℓ in ell]
true_gg44 = [angularCℓ(true_cosmology, tg4, tg4, ℓ) for ℓ in ell]
true_gg55 = [angularCℓ(true_cosmology, tg5, tg5, ℓ) for ℓ in ell]
true_gs11 = [angularCℓ(true_cosmology, tg1, ts1, ℓ) for ℓ in ell]
true_gs22 = [angularCℓ(true_cosmology, tg2, ts2, ℓ) for ℓ in ell]
true_gs33 = [angularCℓ(true_cosmology, tg3, ts3, ℓ) for ℓ in ell]
true_gs44 = [angularCℓ(true_cosmology, tg4, ts4, ℓ) for ℓ in ell]
true_gs55 = [angularCℓ(true_cosmology, tg5, ts5, ℓ) for ℓ in ell]
true_ss11 = [angularCℓ(true_cosmology, ts1, ts1, ℓ) for ℓ in ell]
true_ss22 = [angularCℓ(true_cosmology, ts2, ts2, ℓ) for ℓ in ell]
true_ss33 = [angularCℓ(true_cosmology, ts3, ts3, ℓ) for ℓ in ell]
true_ss44 = [angularCℓ(true_cosmology, ts4, ts4, ℓ) for ℓ in ell]
true_ss55 = [angularCℓ(true_cosmology, ts5, ts5, ℓ) for ℓ in ell]
true_data = vcat([true_gg11, true_gg22, true_gg33, true_gg44, true_gg55,
                  true_gs11, true_gs22, true_gs33, true_gs44, true_gs55,
                  true_ss11, true_ss22, true_ss33, true_ss44, true_ss55]...)
data = true_data #+ 0.1 * true_data .* rand(length(true_data)) 
data_mean = mean(data)
covmat = Diagonal(0.0001.*(data_mean.+data))

@model function model(data)
    Ωm ~ Uniform(0.1, 0.5)
    s8=0.81 #~ Uniform(0.5, 1.0)
    h=0.67 #~ Uniform(0.5, 0.9)
    cosmology = LimberJack.Cosmology(Ωm, 0.05, h, 0.96, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    tg1 = NumberCountsTracer(cosmology, zs, nz1, 2.)
    tg2 = NumberCountsTracer(cosmology, zs, nz2, 2.)
    tg3 = NumberCountsTracer(cosmology, zs, nz3, 2.)
    tg4 = NumberCountsTracer(cosmology, zs, nz4, 2.)
    tg5 = NumberCountsTracer(cosmology, zs, nz5, 2.)
    ts1 = WeakLensingTracer(cosmology, zs, nz1)
    ts2 = WeakLensingTracer(cosmology, zs, nz2)
    ts3 = WeakLensingTracer(cosmology, zs, nz3)
    ts4 = WeakLensingTracer(cosmology, zs, nz4)
    ts5 = WeakLensingTracer(cosmology, zs, nz5)
    gg11 = [angularCℓ(cosmology, tg1, tg1, ℓ) for ℓ in ell]
    gg22 = [angularCℓ(cosmology, tg2, tg2, ℓ) for ℓ in ell]
    gg33 = [angularCℓ(cosmology, tg3, tg3, ℓ) for ℓ in ell]
    gg44 = [angularCℓ(cosmology, tg4, tg4, ℓ) for ℓ in ell]
    gg55 = [angularCℓ(cosmology, tg5, tg5, ℓ) for ℓ in ell]
    gs11 = [angularCℓ(cosmology, tg1, ts1, ℓ) for ℓ in ell]
    gs22 = [angularCℓ(cosmology, tg2, ts2, ℓ) for ℓ in ell]
    gs33 = [angularCℓ(cosmology, tg3, ts3, ℓ) for ℓ in ell]
    gs44 = [angularCℓ(cosmology, tg4, ts4, ℓ) for ℓ in ell]
    gs55 = [angularCℓ(cosmology, tg5, ts5, ℓ) for ℓ in ell]
    ss11 = [angularCℓ(cosmology, ts1, ts1, ℓ) for ℓ in ell]
    ss22 = [angularCℓ(cosmology, ts2, ts2, ℓ) for ℓ in ell]
    ss33 = [angularCℓ(cosmology, ts3, ts3, ℓ) for ℓ in ell]
    ss44 = [angularCℓ(cosmology, ts4, ts4, ℓ) for ℓ in ell]
    ss55 = [angularCℓ(cosmology, ts5, ts5, ℓ) for ℓ in ell]
    predictions = vcat([gg11, gg22, gg33, gg44, gg55,
                        gs11, gs22, gs33, gs44, gs55,
                        ss11, ss22, ss33, ss44, ss55]...)
    data ~ MvNormal(predictions, covmat)
end;

iterations = 500
step_size = 0.005
samples_per_step = 10
cores = 4

# Start sampling.
folpath = "../chains"
folname = string("Halofit_gs_test_", "stpsz_", step_size, "_smpls_", samples_per_step)
folname = joinpath(folpath, folname)
if isdir(folname)
    println("Folder already exists")
    if isfile(joinpath(folname, "chain.jls"))
        println("Restarting from past chain")
        past_chain = read(joinpath(folname, "chain.jls"), Chains)
        new_chain = sample(model(data), HMC(step_size, samples_per_step), iterations,
                           progress=true; save_state=true, resume_from=past_chain)
    else
        new_chain = sample(model(data), HMC(step_size, samples_per_step),
                           iterations, progress=true; save_state=true)
    end
else
    mkdir(folname)
    println("Created new folder")
    new_chain = sample(model(data), HMC(step_size, samples_per_step),
                iterations, progress=true; save_state=true)
end

info = describe(new_chain)[1]
fname_info = string("info.csv")
CSV.write(joinpath(folname, fname_info), info)


fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chain)
    
fname_csv = string("chain.csv")
CSV.write(joinpath(folname, fname_csv), new_chain)
