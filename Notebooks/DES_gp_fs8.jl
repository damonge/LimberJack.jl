using Distributed

@everywhere using Turing
@everywhere using LimberJack
@everywhere using GaussianProcess
@everywhere using CSV
@everywhere using NPZ
@everywhere using FITSIO
@everywhere using Random
@everywhere using DataFrames

@everywhere println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

@everywhere files = npzread("../data/DESY1_cls/gcgc_gcwl_wlwl.npz")
@everywhere cov_tot = files["cov"]
@everywhere data_vector = files["cls"]

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

@everywhere @model function model(data_vector; cov_tot=cov_tot, fid_cosmo=fid_cosmo,
                                  latent_x=latent_x, x=x,
                                  nz_path="../data/DESY1_cls/fiducial_nzs/")
    
    Ωm ~ Uniform(0.1, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    ns ~ Uniform(0.87, 1.07)
    s8 = 0.811
    
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
                     "dz_g0" => dz_g0,
                     "dz_g1" => dz_g1,
                     "dz_g2" => dz_g2,
                     "dz_g3" => dz_g3,
                     "dz_g4" => dz_g4,
                     "dz_k0" => dz_k0,
                     "dz_k1" => dz_k1,
                     "dz_k2" => dz_k2,
                     "dz_k3" => dz_k3,
                     "mb0" => mb0,
                     "mb1" => mb1,
                     "mb2" => mb2,
                     "mb3" => mb3,
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA)

    eta = 0.05
    l = 1
    latent_N = length(latent_x)
    v ~ filldist(truncated(Normal(0, 1), -3, 3), latent_N)
    
    mu = fid_cosmo.Dz(vec(latent_x))
    K = sqexp_cov_fn(latent_x; eta=eta, l=l)
    dmu = fid_cosmo.fs8z(vec(latent_x))
    dK = sqexp_cov_grad(latent_x; eta=eta, l=l)
    latent_gp = latent_GP(mu, v, K)
    latent_dgp = latent_GP(dmu, v, dK)
    gp = conditional(latent_x, x, latent_gp, sqexp_cov_fn;
                      eta=eta, l=l)
    dgp = conditional(latent_x, x, latent_dgp, sqexp_cov_grad;
                      eta=eta, l=l)
    
    cosmology = Cosmology(Ωm, Ωb, h, ns, s8,
                          tk_mode="EisHu",
                          Pk_mode="Halofit", 
                          custom_Dz=[x, gp, dgp])
    
    cls = Theory(cosmology, files;
                 Nuisances=nuisances,
                 nz_path=nz_path)
    
    fs8s = fs8(z_fs8)
    theory = [cls; fs8s]
    
    data_vector ~ MvNormal(theory, cov_tot)
    return(gp=gp, theory=theory)
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
folname = string("DES_full_gp_nos8_", "TAP_", TAP)
folname = joinpath(folpath, folname)

mkdir(folname)
println(string("Created new folder ", folname))

for i in 1:cycles
    if i == 1
        chain = sample(model(data_vector), NUTS(adaptation, TAP), #HMC(init_ϵ, steps), 
                       MCMCDistributed(), iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(model(data_vector), NUTS(adaptation, TAP), #HMC(init_ϵ, steps), 
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
