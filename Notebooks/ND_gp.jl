using Distributed

@everywhere using Turing
@everywhere using LimberJack
@everywhere using GaussianProcess
@everywhere using CSV
@everywhere using NPZ
@everywhere using FITSIO
@everywhere using Random
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

@everywhere fid_cosmo = Cosmology()
@everywhere N = 100
@everywhere latent_x = Vector(0:0.3:3)
@everywhere x = Vector(range(0., stop=3., length=N))

@everywhere function generated_quantities(model::DynamicPPL.Model, chain::MCMCChains.Chains)
   varinfo = DynamicPPL.VarInfo(model)
   iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
   return map(iters) do (sample_idx, chain_idx)
       DynamicPPL.setval!(varinfo, chain, sample_idx, chain_idx)
       model(varinfo)
   end
end

@everywhere @model function model(data_vector;
                                  tracers_names=tracers_names,
                                  pairs=pairs,
                                  pairs_id=pairs_ids,
                                  idx=idx,
                                  cov_tot=cov_tot, 
                                  files=files,
                                  fid_cosmo=fid_cosmo,
                                  latent_x=latent_x,
                                  x=x)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Uniform(0.64, 0.82)
    ns ~ Uniform(0.84, 1.1)
    s8 = 0.811
    
    DESgc__0_0_b ~ Uniform(0.8, 3.0)
    DESgc__1_0_b ~ Uniform(0.8, 3.0)
    DESgc__2_0_b ~ Uniform(0.8, 3.0)
    DESgc__3_0_b ~ Uniform(0.8, 3.0)
    DESgc__4_0_b ~ Uniform(0.8, 3.0)
    DESgc__0_0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__1_0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__2_0_dz ~ TruncatedNormal(0.0, 0.006, -0.2, 0.2)
    DESgc__3_0_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    DESgc__4_0_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)

    DESwl__0_e_dz ~ TruncatedNormal(-0.001, 0.016, -0.2, 0.2)
    DESwl__1_e_dz ~ TruncatedNormal(-0.019, 0.013, -0.2, 0.2)
    DESwl__2_e_dz ~ TruncatedNormal(-0.009, 0.011, -0.2, 0.2)
    DESwl__3_e_dz ~ TruncatedNormal(-0.018, 0.022, -0.2, 0.2)
    DESwl__0_e_m ~ Normal(0.012, 0.023)
    DESwl__1_e_m ~ Normal(0.012, 0.023)
    DESwl__2_e_m ~ Normal(0.012, 0.023)
    DESwl__3_e_m ~ Normal(0.012, 0.023)

    eBOSS__0_0_b ~ Uniform(0.8, 5.0)
    eBOSS__1_0_b ~ Uniform(0.8, 5.0)

    nuisances = Dict("DESgc__0_0_b" => DESgc__0_0_b,
                     "DESgc__1_0_b" => DESgc__1_0_b,
                     "DESgc__2_0_b" => DESgc__2_0_b,
                     "DESgc__3_0_b" => DESgc__3_0_b,
                     "DESgc__4_0_b" => DESgc__4_0_b,
                     "DESgc__0_0_dz" => DESgc__0_0_dz,
                     "DESgc__1_0_dz" => DESgc__1_0_dz,
                     "DESgc__2_0_dz" => DESgc__2_0_dz,
                     "DESgc__3_0_dz" => DESgc__3_0_dz,
                     "DESgc__4_0_dz" => DESgc__4_0_dz,
        
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA,

                     "DESwl__0_e_dz" => DESwl__0_e_dz,
                     "DESwl__1_e_dz" => DESwl__1_e_dz,
                     "DESwl__2_e_dz" => DESwl__2_e_dz,
                     "DESwl__3_e_dz" => DESwl__3_e_dz,
                     "DESwl__0_e_m" => DESwl__0_e_m,
                     "DESwl__1_e_m" => DESwl__1_e_m,
                     "DESwl__2_e_m" => DESwl__2_e_m,
                     "DESwl__3_e_m" => DESwl__3_e_m,
        
                     "eBOSS__0_0_b" => eBOSS__0_0_b,
                     "eBOSS__1_0_b" => eBOSS__1_0_b)

    eta ~ Uniform(0.01, 0.1)
    l ~ Uniform(0.1, 4)
    latent_N = length(latent_x)
    v ~ filldist(truncated(Normal(0, 1), -3, 3), latent_N)
    
    mu = fid_cosmo.Dz(vec(latent_x))
    K = sqexp_cov_fn(latent_x; eta=eta, l=l)
    latent_gp = latent_GP(mu, v, K)
    gp = conditional(latent_x, x, latent_gp, sqexp_cov_fn;
                      eta=eta, l=l)
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="emulator",
                                     Pk_mode="Halofit";
                                     custom_Dz=gp)
    
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
folname = string(data_set, "_gp_hp_TAP_", TAP)
folname = joinpath(folpath, folname)

mkdir(folname)
println(string("Created new folder ", folname))

for i in 1:cycles
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
    derived = generated_quantities(model(data_vector), chain)
    gps = vec([row.gp for row in derived])
    cls = vec([row.theory for row in derived])
    CSV.write(joinpath(folname, string("gps_", i,".csv")), 
              DataFrame(gps, :auto), header = false)
    CSV.write(joinpath(folname, string("cls_", i,".csv")), 
              DataFrame(cls, :auto), header = false)
    
end
