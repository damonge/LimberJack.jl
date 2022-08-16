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
    
    pars = [4.426868e-02,     2.093138e-01,     8.963611e-01,     8.495440e-01,
            1.343888e+00,    1.639047e+00,      1.597174e+00,     1.944583e+00,     2.007245e+00,
           -4.679383e-03,   -2.839996e-03,      1.771571e-03,     1.197051e-03,    -5.199799e-03,
            2.389208e-01,   -6.435288e-01, 
            1.802722e-03,   -5.508994e-03,     1.952514e-02,    -1.117726e-03,
           -1.744083e-02,    6.777779e-03,    -1.097939e-03,    -4.912315e-03,
            8.536883e-01,    2.535825e-01];

    nuisances = Dict("b0" => pars[5],
                     "b1" => pars[6],
                     "b2" => pars[7],
                     "b3" => pars[8],
                     "b4" => pars[9],
                     "dz_g0" => pars[10],
                     "dz_g1" => pars[11],
                     "dz_g2" => pars[12],
                     "dz_g3" => pars[13],
                     "dz_g4" => pars[14],
                     "dz_k0" => pars[21],
                     "dz_k1" => pars[22],
                     "dz_k2" => pars[23],
                     "dz_k3" => pars[24],
                     "m0" => pars[17],
                     "m1" => pars[18],
                     "m2" => pars[19],
                     "m3" => pars[20],
                     "A_IA" => pars[15],
                     "alpha_IA" => pars[16])
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="emulator",
                                     Pk_mode="Halofit")
    
    theory = Theory(cosmology, Cls_meta, files;
                    Nuisances=nuisances).cls
    data_vector ~ MvNormal(theory, cov_tot)
end;

cycles = 6
steps = 10
iterations = 250
TAP = 0.60
adaptation = 1000
init_ϵ = 0.005
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
folname = string("DES_cosmo_emul_", "ϵ", init_ϵ)
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

for i in 1:cycles
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

