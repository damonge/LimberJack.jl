using Turing
using LimberJack
using CSV
using NPZ
using FITSIO
using Dates

println(Threads.nthreads())

files = npzread("../data/DESY1_cls/Cls_meta.npz")
Cls_meta = cls_meta(files)
cov_tot = files["cov"]
data_vector = files["cls"]

@model function model(data_vector; cov_tot=cov_tot)
    Ωm ~ Uniform(0.1, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    s8 ~ Uniform(0.6, 1.0)
    ns ~ Uniform(0.87, 1.07)
    
    b0 ~ Uniform(0.8, 3.0)
    b1 ~ Uniform(0.8, 3.0)
    b2 ~ Uniform(0.8, 3.0)
    b3 ~ Uniform(0.8, 3.0)
    b4 ~ Uniform(0.8, 3.0)
    dz_g0 ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    dz_g1 ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    dz_g2 ~ TruncatedNormal(0.0, 0.006, -0.2, 0.2)
    dz_g3 ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    dz_g4 ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    dz_k0 ~ TruncatedNormal(-0.001, 0.016, -0.2, 0.2)
    dz_k1 ~ TruncatedNormal(-0.019, 0.013, -0.2, 0.2)
    dz_k2 ~ TruncatedNormal(-0.009, 0.011, -0.2, 0.2)
    dz_k3 ~ TruncatedNormal(-0.018, 0.022, -0.2, 0.2)
    m0 ~ Normal(0.012, 0.023)
    m1 ~ Normal(0.012, 0.023)
    m2 ~ Normal(0.012, 0.023)
    m3 ~ Normal(0.012, 0.023)
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
    
    theory = Theory(cosmology, Cls_meta, files;
                    Nuisances=nuisances).cls
    data_vector ~ MvNormal(theory, cov_tot)
end;

cycles = 10
iterations = 500
TAP = 0.60
adaptation = 1000
#nchains = Threads.nthreads()


new_chain = sample(model(data_vector), NUTS(adaptation, TAP), 
                   iterations, progress=true; save_state=true)

# Start sampling.
folpath = "../chains"
folname = string("DES_full_", "TAP", TAP)
folname = joinpath(folpath, folname)

mkdir(folname)
println("Created new folder")

for i in 1:cycles
    if i == 1
        chain = sample(model(data_vector), NUTS(adaptation, TAP), 
                       iterations, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1, ".jls")), Chains)
        chain = sample(model(data_vector), NUTS(adaptation, TAP), 
                       iterations, progress=true; save_state=true,
                       resume_from=old_chain)
    end 
    write(joinpath(folname, string("chain_", i,".jls")), chain)
end