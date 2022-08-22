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

@everywhere  zs_g0, nz_g0 = get_nzs(files, 1, 0)
@everywhere  zs_g1, nz_g1 = get_nzs(files, 1, 1)
@everywhere  zs_g2, nz_g2 = get_nzs(files, 1, 2)
@everywhere  zs_g3, nz_g3 = get_nzs(files, 1, 3)
@everywhere  zs_g4, nz_g4 = get_nzs(files, 1, 4)
@everywhere  zs_k0, nz_k0 = get_nzs(files, 2, 0)
@everywhere  zs_k1, nz_k1 = get_nzs(files, 2, 1)
@everywhere  zs_k2, nz_k2 = get_nzs(files, 2, 2)
@everywhere  zs_k3, nz_k3 = get_nzs(files, 2, 3)


@everywhere @model function model(data_vector; cov_tot=cov_tot)
    #KiDS priors
    Ωm ~ Uniform(0.1, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    ns ~ Uniform(0.87, 1.07)
    s8 ~ Uniform(0.6, 0.9)
    
    b0 ~ Uniform(0.8, 3.0)
    b1 ~ Uniform(0.8, 3.0)
    b2 ~ Uniform(0.8, 3.0)
    b3 ~ Uniform(0.8, 3.0)
    b4 ~ Uniform(0.8, 3.0)
    nz_g0 ~ MvNormal(nz_g0, cov_g0)
    nz_g1 ~ MvNormal(nz_g1, cov_g1)
    nz_g2 ~ MvNormal(nz_g2, cov_g2)
    nz_g3 ~ MvNormal(nz_g3, cov_g3)
    nz_g4 ~ MvNormal(nz_g4, cov_g4)
    nz_k0 ~ MvNormal(nz_k0, cov_k0)
    nz_k1 ~ MvNormal(nz_k1, cov_k1)
    nz_k2 ~ MvNormal(nz_k2, cov_k2)
    nz_k3 ~ MvNormal(nz_k3, cov_k3)
    mb0 ~ Normal(0.012, 0.023)
    mb1 ~ Normal(0.012, 0.023)
    mb2 ~ Normal(0.012, 0.023)
    mb3 ~ Normal(0.012, 0.023)
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)

    nuisances = Dict("b0" => b0,
                     "b1" => b1,
                     "b2" => b2,
                     "b3" => b3,
                     "b4" => b4,
                     "nz_g0" => nz_g0,
                     "nz_g1" => nz_g1,
                     "nz_g2" => nz_g2,
                     "nz_g3" => nz_g3,
                     "nz_g4" => nz_g4,
                     "nz_k0" => nz_k0,
                     "nz_k1" => nz_k1,
                     "nz_k2" => nz_k2,
                     "nz_k3" => nz_k3,
                     "mb0" => mb0,
                     "mb1" => mb1,
                     "mb2" => mb2,
                     "mb3" => mb3,
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA)
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    theory = Theory(cosmology, Cls_meta, files;
                    Nuisances=nuisances).cls
    data_vector ~ MvNormal(theory, cov_tot)
end;

cycles = 6
steps = 50
iterations = 250
TAP = 0.60
adaptation = 1000
init_ϵ = 0.05
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
folname = string("DES_full_", "TAP_", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    fol_files = readdir(folname)
    last_chain = last([file for file in fol_files if occursin("chain", file)])
    last_n = parse(Int, last_chain[7])
    println("Restarting chain")
else
    mkdir(folname)
    println(string("Created new folder ", folname))
    last_n = 0
end

for i in (1+last_n):(last_n+cycles)
    if i == 1
        chain = sample(model(data_vector), HMC(init_ϵ, steps),
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(model(data_vector), HMC(init_ϵ, steps),
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true,
                       resume_from=old_chain)
    end  
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
