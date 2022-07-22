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

@everywhere fid_cosmo = Cosmology()
@everywhere N = 100
@everywhere latent_x = Vector(0:0.3:3)
@everywhere x = Vector(range(0., stop=3., length=N))

@everywhere struct Determin{T<:Any} <: ContinuousUnivariateDistribution
  val::T
end

@everywhere Distributions.rand(rng::AbstractRNG, d::Determin) = d.val
@everywhere Distributions.logpdf(d::Determin, x::T) where T<:Real = zero(x)
@everywhere Bijectors.bijector(d::Determin) = Identity{0}()

@everywhere @model function model(data_vector; cov_tot=cov_tot, fid_cosmo=fid_cosmo,
                      latent_x=latent_x, x=x)
    eta = 0.03
    l = 1
    latent_N = length(latent_x)
    v ~ MvNormal(zeros(latent_N), ones(latent_N))
    
    Ωm ~ Uniform(0.1, 0.6)
    Ωb = 0.05
    h = 0.67
    s8 = 0.811
    ns = 0.96 
    
    nuisances = Dict()
    
    mu = fid_cosmo.Dz(vec(latent_x))
    K = sqexp_cov_fn(latent_x; eta=eta, l=l)
    latent_gp = latent_GP(mu, v, K)
    gp = conditional(latent_x, x, latent_gp, sqexp_cov_fn;
                      eta=eta, l=l)
    gp_val ~ Determin(gp)
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit", 
                                     custom_Dz=gp)
    
    theory = Theory(cosmology, Cls_meta, files;
                    Nuisances=nuisances).cls
    data_vector ~ MvNormal(theory, cov_tot)
end;

@everywhere cycles = 3
@everywhere iterations = 500
@everywhere TAP = 0.60
@everywhere adaptation = 1000
@everywhere init_ϵ=0.03

# Start sampling.
@everywhere folpath = "../chains"
@everywhere folname = string("DES_simplest_gp_", "TAP", TAP)
@everywhere folname = joinpath(folpath, folname)

mkdir(folname)
println("Created new folder")

@everywhere for i in 1:cycles
    if i == 1
        chain = sample(model(data_vector), NUTS(adaptation, TAP), 
                       iterations, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("worker_", myid(), "_chain_", i-1,".jls")), Chains)
        chain = sample(model(data_vector), NUTS(adaptation, TAP), 
                       iterations, progress=true; save_state=true,
                       resume_from=old_chain)
    end 
    write(joinpath(folname, string("worker_", myid(), "_chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("worker_", myid(), "_chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("worker_", myid(), "_summary_", i,".csv")), describe(chain)[1])
end