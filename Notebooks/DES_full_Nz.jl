using Distributed

@everywhere using Turing
@everywhere using ReverseDiff
@everywhere Turing.setadbackend(:reversediff)
@everywhere using LimberJack
@everywhere using CSV
@everywhere using NPZ
@everywhere using FITSIO
@everywhere using LinearAlgebra

@everywhere println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

@everywhere files = npzread("../data/DESY1_cls/wlwl.npz")
@everywhere cov_tot = files["cov"]
@everywhere data_vector = files["cls"]

@everywhere nz_path = "../data/DESY1_cls/fullmarg_nzs/"
@everywhere zs_k0, nz_k0, cov_k0 = get_nzs(nz_path, 2, 0)
@everywhere zs_k1, nz_k1, cov_k1 = get_nzs(nz_path, 2, 1)
@everywhere zs_k2, nz_k2, cov_k2 = get_nzs(nz_path, 2, 2)
@everywhere zs_k3, nz_k3, cov_k3 = get_nzs(nz_path, 2, 3)

@everywhere cov_k0 = Diagonal(cov_k0)
@everywhere cov_k1 = Diagonal(cov_k1)
@everywhere cov_k2 = Diagonal(cov_k2)
@everywhere cov_k3 = Diagonal(cov_k3)

@everywhere @model function model(data_vector; cov_tot=cov_tot, nz_path=nz_path)
    #KiDS priors
    Ωm ~ Uniform(0.1, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    ns ~ Uniform(0.87, 1.07)
    s8 ~ Uniform(0.6, 0.9)
    
    Nz_k0 ~ MvNormal(nz_k0, cov_k0)
    Nz_k1 ~ MvNormal(nz_k1, cov_k1)
    Nz_k2 ~ MvNormal(nz_k2, cov_k2)
    Nz_k3 ~ MvNormal(nz_k3, cov_k3)
    mb0 ~ Normal(0.012, 0.023)
    mb1 ~ Normal(0.012, 0.023)
    mb2 ~ Normal(0.012, 0.023)
    mb3 ~ Normal(0.012, 0.023)
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)

    nuisances = Dict("nz_k0" => Nz_k0,
                     "nz_k1" => Nz_k1,
                     "nz_k2" => Nz_k2,
                     "nz_k3" => Nz_k3,
                     "mb0" => mb0,
                     "mb1" => mb1,
                     "mb2" => mb2,
                     "mb3" => mb3,
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA)
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    theory = Theory(cosmology, files;
                    nz_path=nz_path,
                    Nuisances=nuisances)
    
    data_vector ~ MvNormal(theory, cov_tot)
end;

cycles = 6
steps = 50
iterations = 250
TAP = 0.60
adaptation = 250
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
folname = string("DES_wlwl_Nzs2_", "TAP_", TAP)
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