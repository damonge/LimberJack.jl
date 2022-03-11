using Turing
using Random
using ForwardDiff
using LimberJack
using CSV
using NPZ
using FITSIO

# Add four processes to use for sampling.
using ClusterManager
#add 20 processes on the long queue
cores = 4
ClusterManagers.addprocs_lsf(cores; bsub_flags = `-q berg`)
#loading Turing on the processes
@everywhere using Turing, LimberJack

ell = npzread("data/cl_DESgc__2_DESwl__3.npz")["ell"]
ell = [Int(floor(l)) for l in ell]
nzs = FITS("data/y1_redshift_distributions_v1.fits")
nz = read(nzs["nz_source_mcal"], "BIN2")
zs = read(nzs["nz_source_mcal"], "Z_MID")

true_cosmology = LimberJack.Cosmology(0.25, 0.05, 0.67, 0.96, 0.81,
                                      tk_mode="EisHu", Pk_mode="Halofit");
tg = NumberCountsTracer(true_cosmology, zs, nz, 2.)
ts = WeakLensingTracer(true_cosmology, zs, nz)
true_data = [angularCℓ(true_cosmology, tg, ts, ℓ) for ℓ in ell]
data = true_data #+ 0.1 * true_data .* rand(length(true_data)) 

@model function model(data)
    Ωm ~ Uniform(0.1, 0.4)
    s8 ~ Uniform(0.5, 1.0)
    h ~ Uniform(0.5, 0.9)
    cosmology = LimberJack.Cosmology(Ωm, 0.05, h, 0.96, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    tg = NumberCountsTracer(cosmology, zs, nz, 2.)
    ts = WeakLensingTracer(cosmology, zs, nz)
    predictions = [angularCℓ(cosmology, tg, ts, ℓ) for ℓ in ell]
    data ~ MvNormal(predictions, 0.01.*data)
end;

iterations = 10
step_size = 0.05
samples_per_step = 5
# Start sampling.

folname = string("Parallelization_test_", "stpsz_", step_size, "_smpls_", samples_per_step)
if isdir(folname)
    println("Folder already exists")
    if isfile("chain.jls")
        println("Restarting from past chain")
        past_chain = read("chain.jls", Chains)
        new_chain = sample(model(data), HMC(step_size, samples_per_step),
                           MCMCDistributed(), iterations, cores, progress=true;
                           save_state=true, resume_from=past_chain)
    end
else
    mkdir(folname)
    println("Created new folder")
    new_chain = sample(model(data), HMC(step_size, samples_per_step),
                       MCMCDistributed(), iterations, cores, progress=true; save_state=true)
end

info = describe(new_chain)[1]
fname_info = string("info.csv")
CSV.write(joinpath(folname, fname_info), info)


fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chain)
    
fname_csv = string("chain.csv")
CSV.write(joinpath(folname, fname_csv), new_chain)