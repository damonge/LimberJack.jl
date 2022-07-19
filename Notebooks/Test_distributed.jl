using Distributed
using Turing

@everywhere using Turing
@everywhere using LimberJack
@everywhere using CSV
@everywhere using NPZ
@everywhere using FITSIO
@everywhere using Dates

@everywhere println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

@everywhere files = npzread("../data/DESY1_cls/Cls_meta.npz")
@everywhere Cls_meta = cls_meta(files)
@everywhere cov_tot = files["cov"]
@everywhere data_vector = files["cls"]

@everywhere @model function model(data_vector; cov_tot=cov_tot)
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
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)

    nuisances = Dict("b0" => b0,
                     "b1" => b1,
                     "b2" => b2,
                     "b3" => b3,
                     "b4" => b4,
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA)
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    theory = Theory(cosmology, nuisances, Cls_meta, files)
    data_vector ~ MvNormal(theory, cov_tot)
end;

iterations = 2
TAP = 0.1
adaptation = 2
nchains = nprocs()

# Start sampling.
folpath = "../chains"
folname = string("Test_distributed_", "TAP", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    println("Removed folder")
    rm(folname)
end

mkdir(folname)
println("Created new folder")

new_chain = sample(model(data_vector), NUTS(adaptation, TAP), MCMCDistributed(),
                   iterations, nchains, progress=true; save_state=true)

#= 
new_chain = sample(model(data_vector), NUTS(adaptation, TAP), MCMCThreads(),
                   iterations, nchains, progress=true; save_state=true,
                   resume_from=past_chain)
=#

summary = describe(new_chain)[1]
fname_summary = string("summary", now(), ".csv")
CSV.write(joinpath(folname, fname_summary), summary)

fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chain)

fname_csv = string("chain_", now(), ".csv")
CSV.write(joinpath(folname, fname_csv), new_chain)
