using Turing
using LimberJack
using CSV
using NPZ
using FITSIO
using Dates

files = npzread("../data/DESY1_cls/Cls_meta.npz")
Cls_meta = cls_meta(files)
cov_tot = files["cov"]
data_vector = files["cls"]

@model function model(data_vector; cov_tot=cov_tot)
    Ωm ~ Uniform(0.1, 0.6)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.6, 0.9)
    s8 ~ Uniform(0.6, 1.0)
    ns ~ Uniform(0.87, 1.07)
    
    b0 ~ Uniform(0.8, 3.0)
    b1 ~ Uniform(0.8, 3.0)
    b2 ~ Uniform(0.8, 3.0)
    b3 ~ Uniform(0.8, 3.0)
    b4 ~ Uniform(0.8, 3.0)
    
    dz_g0 ~ Normal(0.008, 0.007)
    dz_g1 ~ Normal(-0.005, 0.007)
    dz_g2 ~ Normal(0.006, 0.006)
    dz_g3 ~ Normal(0.0, 0.010)
    dz_g4 ~ Normal(0.0, 0.010)
    
    dz_k0 ~ Normal(-0.001, 0.016)
    dz_k1 ~ Normal(-0.019, 0.013)
    dz_k2 ~ Normal(-0.009, 0.011)
    dz_k3 ~ Normal(-0.018, 0.022)
    
    m0 ~ Normal(0.0, 0.035)
    m1 ~ Normal(0.0, 0.035)
    m2 ~ Normal(0.0, 0.035)
    m3 ~ Normal(0.0, 0.035)
    
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)

    nuisances = Dict("b0" => b0,
                     "b1" => b1,
                     "b2" => b2,
                     "b3" => b3,
                     "b4" => b4,
                     "dz_g0" => dz_g0,
                     "dz_g1" => dz_g1,
                     "dz_g2" => dz_g2,
                     "dz_g3" => dz_g3,
                     "dz_g4" => dz_g4,
                     "dz_k0" => dz_k0,
                     "dz_k1" => dz_k1,
                     "dz_k2" => dz_k2,
                     "dz_k3" => dz_k3,
                     "m0" => m0,
                     "m1" => m1,
                     "m2" => m2,
                     "m3" => m3,
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA)
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    theory = Theory_parallel(cosmology, nuisances, Cls_meta, files)
    data_vector ~ MvNormal(theory, cov_tot)
end;

iterations = 5000
TAP = 0.60
adaptation = 1000

# Start sampling.
folpath = "../chains"
folname = string("DES_full_parallel_", "TAP", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    println("Removed folder")
    rm(folname)
end

mkdir(folname)
println("Created new folder")

new_chain = sample(model(data_vector), NUTS(adaptation, TAP),
                   iterations, progress=true; save_state=true)

#new_chain = sample(model(data_vector), NUTS(adaptation, TAP), iterations,
#                   progress=true; save_state=true, resume_from=past_chain)


summary = describe(new_chain)[1]
fname_summary = string("summary", now(), ".csv")
CSV.write(joinpath(folname, fname_summary), summary)

fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chain)
    
fname_csv = string("chain_", now(), ".csv")
CSV.write(joinpath(folname, fname_csv), new_chain)
