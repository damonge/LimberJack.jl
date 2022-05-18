using Turing
using Random
using ForwardDiff
using LimberJack
using CSV
using NPZ
using FITSIO

println(Threads.nthreads())

files = npzread("../data/DESY1_cls/Cls_meta.npz")
Cls_meta = cls_meta(files)
cov_tot = files["cov"]
data_vector = files["cls"]

@model function model(data_vector; cov_tot=cov_tot)
    Ωm ~ Uniform(0.1, 0.6)
    Ωb = 0.05 
    h = 0.67 
    s8 ~ Uniform(0.6, 1.0)
    ns = 0.96 

    nuisances = Dict()
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    theory = Theory(cosmology, nuisances, Cls_meta, files)
    data_vector ~ MvNormal(theory, cov_tot)
end;

iterations = 500
TAP = 0.60
adaptation = 100

# Start sampling.
folpath = "../chains"
folname = string("test_linear_", "TAP", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    println("Removed folder")
    rm(folname)
end

mkdir(folname)
println("Created new folder")

new_chains = sample(model(data_vector), NUTS(adaptation, TAP),
                   iterations, progress=true; save_state=true)

summary = describe(new_chains)[1]
fname_summary = string("summary", now(), ".csv")
CSV.write(joinpath(folname, fname_summary), summary)

fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chains)
    
fname_csv = string("chain_", now(), ".csv")
CSV.write(joinpath(folname, fname_csv), new_chains)

