using Distributed

@everywhere using LinearAlgebra
@everywhere using Turing
@everywhere using LimberJack
@everywhere using CSV
@everywhere using NPZ
@everywhere using FITSIO
@everywhere using PythonCall
@everywhere np = pyimport("numpy")

@everywhere println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

@everywhere data_set = "DECALS_KiDS"
@everywhere meta = np.load(string("../data/", data_set, "/", data_set, "_meta.npz"))
@everywhere files = npzread(string("../data/", data_set, "/", data_set, "_files.npz"))

@everywhere tracers_names = pyconvert(Vector{String}, meta["tracers"])
@everywhere pairs = pyconvert(Vector{Vector{String}}, meta["pairs"])
@everywhere idx = pyconvert(Vector{Int}, meta["idx"])
@everywhere data_vector = pyconvert(Vector{Float64}, meta["cls"])
@everywhere cov_tot = pyconvert(Matrix{Float64}, meta["cov"]);
@everywhere errs = sqrt.(diag(cov_tot))
@everywhere fake_cov = Hermitian(cov_tot ./ (errs * errs'))
@everywhere fid_cosmo = Cosmology(0.3, 0.05, 0.67, 0.96, 0.81,
                      tk_mode="emulator", Pk_mode="Halofit")
@everywhere fid_nui =  nuisances = Dict("DECALS__0_0_b" => 1.166,
                                        "DECALS__1_0_b" => 1.399,
                                        "DECALS__2_0_b" => 1.349,
                                        "DECALS__3_0_b" => 1.823)
@everywhere fid_data = Theory(fid_cosmo, tracers_names, pairs, idx, files;
                              Nuisances=fid_nui)
@everywhere fake_data = fid_data ./ errs;

@everywhere @model function model(data;
                                  tracers_names=tracers_names,
                                  pairs=pairs,
                                  idx=idx,
                                  cov=fake_cov, 
                                  files=files)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb = 0.05 #~ Uniform(0.028, 0.065)
    h = 0.67 #~ Uniform(0.64, 0.82)
    s8 ~ Uniform(0.6, 0.9)
    ns = 0.96 #~ Uniform(0.84, 1.1)
    
    DECALS__0_0_b = 1.166 #~ Uniform(0.8, 3.0)
    DECALS__1_0_b = 1.399 #~ Uniform(0.8, 3.0)
    DECALS__2_0_b = 1.349 #~ Uniform(0.8, 3.0)
    DECALS__3_0_b = 1.823 #~ Uniform(0.8, 3.0)

    nuisances = Dict(
                     "DECALS__0_0_b" => DECALS__0_0_b,
                     "DECALS__1_0_b" => DECALS__1_0_b,
                     "DECALS__2_0_b" => DECALS__2_0_b,
                     "DECALS__3_0_b" => DECALS__3_0_b)

    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="emulator",
                                     Pk_mode="Halofit")
    
    theory = Theory(cosmology, tracers_names, pairs,
                    idx, files; Nuisances=nuisances)
    data ~ MvNormal(theory ./ errs, cov)
end;

cycles = 6
steps = 50
iterations = 100
TAP = 0.65
adaptation = 100
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
folname = string(data_set, "_fake_data_cosmo_TAP_", TAP)
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
        chain = sample(model(fake_data), NUTS(adaptation, TAP), #HMC(init_ϵ, steps),
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(model(fake_data), NUTS(adaptation, TAP), #HMC(init_ϵ, steps),
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true,
                       resume_from=old_chain)
    end  
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
