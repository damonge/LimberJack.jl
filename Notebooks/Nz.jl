using Distributed

@everywhere using Turing
@everywhere using LimberJack
@everywhere using CSV
@everywhere using NPZ
@everywhere using FITSIO
@everywhere using LinearAlgebra
@everywhere using PythonCall
@everywhere np = pyimport("numpy")

@everywhere println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

@everywhere fol = "DESY1"
@everywhere data_set = "wlwl_lite10"
@everywhere meta = np.load(string("../data/", fol, "/", data_set, "_meta.npz"))
@everywhere files = npzread(string("../data/", fol, "/", data_set, "_files.npz"))

@everywhere tracers_names = pyconvert(Vector{String}, meta["tracers"])
@everywhere pairs = pyconvert(Vector{Vector{String}}, meta["pairs"]);
@everywhere pairs_ids = pyconvert(Vector{Vector{Int}}, meta["pairs_ids"])
@everywhere idx = pyconvert(Vector{Int}, meta["idx"])
@everywhere data_vector = pyconvert(Vector{Float64}, meta["cls"])
@everywhere cov_tot = pyconvert(Matrix{Float64}, meta["cov"]);

@everywhere nz_path = "../data/DESY1/lite10_nzs/"
@everywhere zs_k0, nz_k0, cov_k0 = get_nzs(nz_path, "DESwl__0_e")
@everywhere zs_k1, nz_k1, cov_k1 = get_nzs(nz_path, "DESwl__1_e")
@everywhere zs_k2, nz_k2, cov_k2 = get_nzs(nz_path, "DESwl__2_e")
@everywhere zs_k3, nz_k3, cov_k3 = get_nzs(nz_path, "DESwl__3_e")

@everywhere @model function model(data_vector;
                                  tracers_names=tracers_names,
                                  pairs=pairs,
                                  pairs_id=pairs_ids,
                                  idx=idx,
                                  cov_tot=cov_tot, 
                                  files=files)
    Ωm ~ Uniform(0.1, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    ns ~ Uniform(0.87, 1.07)
    s8 ~ Uniform(0.6, 0.9)
    
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)
    DESwl__0_e_nz ~ MvNormal(nz_k0, cov_k0)
    DESwl__1_e_nz ~ MvNormal(nz_k1, cov_k1)
    DESwl__2_e_nz ~ MvNormal(nz_k2, cov_k2)
    DESwl__3_e_nz ~ MvNormal(nz_k3, cov_k3)
    DESwl__0_e_m ~ Normal(0.012, 0.023)
    DESwl__1_e_m ~ Normal(0.012, 0.023)
    DESwl__2_e_m ~ Normal(0.012, 0.023)
    DESwl__3_e_m ~ Normal(0.012, 0.023)


    nuisances = Dict("A_IA" => A_IA,
                     "alpha_IA" => alpha_IA,

                     "DESwl__0_e_nz" => DESwl__0_e_nz,
                     "DESwl__1_e_nz" => DESwl__1_e_nz,
                     "DESwl__2_e_nz" => DESwl__2_e_nz,
                     "DESwl__3_e_nz" => DESwl__3_e_nz,
                     "DESwl__0_e_m" => DESwl__0_e_m,
                     "DESwl__1_e_m" => DESwl__1_e_m,
                     "DESwl__2_e_m" => DESwl__2_e_m,
                     "DESwl__3_e_m" => DESwl__3_e_m)
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
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
folname = string("DES_wlwl_Nzs_", "TAP_", TAP)
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

for i in (1+last_n):(last_n+cycles)
    if i == 1
        chain = sample(model(data_vector), NUTS(adaptation, TAP),
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(model(data_vector), NUTS(adaptation, TAP),
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true,
                       resume_from=old_chain)
    end  
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
