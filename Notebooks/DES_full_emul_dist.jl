using Distributed

@everywhere using Turing
@everywhere using LimberJack
@everywhere using CSV
@everywhere using NPZ
@everywhere using FITSIO

@everywhere println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

@everywhere files = npzread("../data/DESY1_cls/Cls_meta.npz")
@everywhere Cls_meta = cls_meta(files)
@everywhere cov_tot = files["cov"]
@everywhere data_vector = files["cls"]


@everywhere @model function model(data_vector; cov_tot=cov_tot)
    #KiDS priors
    Ωm ~ Uniform(0.1, 0.9)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Uniform(0.64, 0.82)
    s8 ~ Uniform(0.6, 1.0)
    ns ~ Uniform(0.70, 1.30)
    
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
                                     tk_mode="emulator",
                                     Pk_mode="Halofit")
    
    theory = Theory(cosmology, Cls_meta, files;
                    Nuisances=nuisances).cls
    data_vector ~ MvNormal(theory, cov_tot)
end;

cycles = 3
iterations = 500
TAP = 0.60
adaptation = 1000
init_ϵ = 0.03
nchains = nprocs()
println("sampling settings: ")
println("cycles ", cycles)
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)
println("init_ϵ ", init_ϵ)
println("nchains ", nchains)

# Start sampling.
folpath = "../chains"
folname = string("DES_full_emul_dist_", "TAP", TAP)
folname = joinpath(folpath, folname)

mkdir(folname)
println(string("Created new folder ", folname))

for i in 1:cycles
    if i == 1
        chain = sample(model(data_vector), NUTS(adaptation, TAP; init_ϵ=init_ϵ), 
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(model(data_vector), NUTS(adaptation, TAP; init_ϵ=init_ϵ), 
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true,
                       resume_from=old_chain)
    end 
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
