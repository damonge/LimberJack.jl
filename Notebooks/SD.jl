using Distributed

@everywhere using Turing
@everywhere using LimberJack
@everywhere using CSV
@everywhere using NPZ
@everywhere using FITSIO
@everywhere using PythonCall
@everywhere np = pyimport("numpy")

@everywhere println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

@everywhere data_set = "SD"
@everywhere meta = np.load(string("../data/", data_set, "/", data_set, "_meta.npz"))
@everywhere files = npzread(string("../data/", data_set, "/", data_set, "_files.npz"))

@everywhere tracers_names = pyconvert(Vector{String}, meta["tracers"])
@everywhere pairs = pyconvert(Vector{Vector{String}}, meta["pairs"]);
@everywhere pairs_ids = pyconvert(Vector{Vector{Int}}, meta["pairs_ids"])
@everywhere idx = pyconvert(Vector{Int}, meta["idx"])
@everywhere data_vector = pyconvert(Vector{Float64}, meta["cls"])
@everywhere cov_tot = pyconvert(Matrix{Float64}, meta["cov"]);


@everywhere @model function model(data_vector;
                                  tracers_names=tracers_names,
                                  pairs=pairs,
                                  pairs_id=pairs_ids,
                                  idx=idx,
                                  cov_tot=cov_tot, 
                                  files=files)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Uniform(0.64, 0.82)
    s8 ~ Uniform(0.6, 0.9)
    ns ~ Uniform(0.84, 1.1)
    
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)

    eBOSS__0_0_b ~ Uniform(0.8, 5.0)
    eBOSS__1_0_b ~ Uniform(0.8, 5.0)
    
    DECALS__0_0_b ~ Uniform(0.8, 3.0)
    DECALS__1_0_b ~ Uniform(0.8, 3.0)
    DECALS__2_0_b ~ Uniform(0.8, 3.0)
    DECALS__3_0_b ~ Uniform(0.8, 3.0)
    DECALS__0_0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DECALS__1_0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DECALS__2_0_dz ~ TruncatedNormal(0.0, 0.006, -0.2, 0.2)
    DECALS__3_0_dz ~ TruncatedNormal(0.0, 0.010, -0.2, 0.2)
    
    KiDS1000__0_e_dz ~ TruncatedNormal(0.0, 0.0106, -0.2, 0.2)
    KiDS1000__1_e_dz ~ TruncatedNormal(0.0, 0.0113, -0.2, 0.2)
    KiDS1000__2_e_dz ~ TruncatedNormal(0.0, 0.0118, -0.2, 0.2)
    KiDS1000__3_e_dz ~ TruncatedNormal(0.0, 0.0087, -0.2, 0.2)
    KiDS1000__4_e_dz ~ TruncatedNormal(0.0, 0.0097, -0.2, 0.2)
    KiDS1000__0_e_m ~ Normal(0.0, 0.019)
    KiDS1000__1_e_m ~ Normal(0.0, 0.020)
    KiDS1000__2_e_m ~ Normal(0.0, 0.017)
    KiDS1000__3_e_m ~ Normal(0.0, 0.012)
    KiDS1000__4_e_m ~ Normal(0.0, 0.010)


    nuisances = Dict("A_IA" => A_IA,
                     "alpha_IA" => alpha_IA,
        
                     "eBOSS__0_0_b" => eBOSS__0_0_b,
                     "eBOSS__1_0_b" => eBOSS__1_0_b,
        
                     "DECALS__0_0_b" => DECALS__0_0_b,
                     "DECALS__1_0_b" => DECALS__1_0_b,
                     "DECALS__2_0_b" => DECALS__2_0_b,
                     "DECALS__3_0_b" => DECALS__3_0_b,
                     "DECALS__0_0_dz" => DECALS__0_0_dz,
                     "DECALS__1_0_dz" => DECALS__1_0_dz,
                     "DECALS__2_0_dz" => DECALS__2_0_dz,
                     "DECALS__3_0_dz" => DECALS__3_0_dz,
                    
                     "KiDS1000__0_e_dz" => KiDS1000__0_e_dz,
                     "KiDS1000__1_e_dz" => KiDS1000__1_e_dz,
                     "KiDS1000__2_e_dz" => KiDS1000__2_e_dz,
                     "KiDS1000__3_e_dz" => KiDS1000__3_e_dz,
                     "KiDS1000__4_e_dz" => KiDS1000__4_e_dz,
                     "KiDS1000__0_e_m" => KiDS1000__0_e_m,
                     "KiDS1000__1_e_m" => KiDS1000__1_e_m,
                     "KiDS1000__2_e_m" => KiDS1000__2_e_m,
                     "KiDS1000__3_e_m" => KiDS1000__3_e_m,
                     "KiDS1000__4_e_m" => KiDS1000__4_e_m)

    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="emulator",
                                     Pk_mode="Halofit")
    
    theory = Theory(cosmology, tracers_names, pairs,
                    pairs_ids, idx, files;
                    Nuisances=nuisances)
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
    if length(fol_files)
        last_chain = last([file for file in fol_files if occursin("chain", file)])
        last_n = parse(Int, last_chain[7])
        println("Restarting chain")
    else
        last_n = 0
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
