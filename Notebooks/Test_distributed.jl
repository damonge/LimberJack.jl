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
    Ωb = 0.05
    h = 0.67
    s8 = 0.811
    ns = 0.96
    nuisances = Dict()
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    theory = Theory(cosmology, Cls_meta, files;
                    Nuisances=nuisances).cls
    data_vector ~ MvNormal(theory, cov_tot)
end;

nchains = nprocs()
cycles = 10
iterations = 500
TAP = 0.60
adaptation = 1000

# Start sampling.
folpath = "../chains"
folname = string("Test_distributed_", "TAP", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    rm(folname, recursive=true)
end
    
mkdir(folname)
println("Created new folder")

@everywhere for i in 1:cycles
    if i == 1
        chain = sample(model(data_vector), NUTS(adaptation, TAP; init_ϵ=0.03), #MCMCThrea(),
                       iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("worker_", myid(), "_chain_", i,".jls")), Chains)
        chain = sample(model(data_vector), NUTS(adaptation, TAP; init_ϵ=0.03), #MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true,
                       resume_from=old_chain)
    end 
    write(joinpath(folname, string("worker_", myid(), "_chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("worker_", myid(), "_chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("worker_", myid(), "_summary_", i,".csv")), describe(chain)[1])
end
