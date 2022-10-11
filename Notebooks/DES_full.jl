using Distributed

@everywhere using Turing
@everywhere using LimberJack
@everywhere using CSV
@everywhere using NPZ
@everywhere using FITSIO
@everywhere using PythonCall
@everywhere np = pyimport("numpy")

@everywhere println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

@everywhere data_set = "DESY1"
@everywhere meta = np.load(string("../data/", data_set, "/", "gcgc_gcwl_wlwl", "_meta.npz"))
@everywhere files = npzread(string("../data/", data_set, "/", "gcgc_gcwl_wlwl", "_files.npz"))

@everywhere tracers_names = pyconvert(Vector{String}, meta["tracers"])
@everywhere pairs = pyconvert(Vector{Vector{String}}, meta["pairs"])
@everywhere idx = pyconvert(Vector{Int}, meta["idx"])
@everywhere data_vector = pyconvert(Vector{Float64}, meta["cls"])
@everywhere cov_tot = pyconvert(Matrix{Float64}, meta["cov"]);

@everywhere @model function model(data_vector;
                                  tracers_names=tracers_names,
                                  pairs=pairs,
                                  idx=idx,
                                  cov_tot=cov_tot, 
                                  files=files)
    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Uniform(0.64, 0.82)
    s8 ~ Uniform(0.6, 0.9)
    ns ~ Uniform(0.84, 1.1)
    
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
    
    theory = Theory(cosmology, tracers_names, pairs,
                    idx, files; Nuisances=nuisances)
    data_vector ~ MvNormal(theory, cov_tot)
end;

cycles = 6
steps = 50
iterations = 100
TAP = 0.60
adaptation = 100
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
folname = string(data_set, "_TAP_", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    fol_files = readdir(folname)
    println("Found existing file")
    if length(fol_files) != 0
        last_chain = last([file for file in fol_files if occursin("chain", file)])
        last_n = parse(Int, last_chain[7])
        println("Restarting chain")
    else
        last_n = 0
    end
else
    mkdir(folname)
    println(string("Created new folder ", folname))
    last_n = 0
end

for i in (1+last_n):(cycles+last_n)
    if i == 1
        chain = sample(model(data_vector), NUTS(adaptation, TAP; init_ϵ=init_ϵ), #HMC(init_ϵ, steps),
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(model(data_vector), NUTS(adaptation, TAP; init_ϵ=init_ϵ), #HMC(init_ϵ, steps),
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true,
                       resume_from=old_chain)
    end  
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
