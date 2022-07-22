using Turing
using LimberJack
using CSV
using NPZ
using FITSIO
using Dates
using GaussianProcess
using Random

println(Threads.nthreads())

files = npzread("../data/DESY1_cls/Cls_meta.npz")
Cls_meta = cls_meta(files)
cov_tot = files["cov"]
data_vector = files["cls"]

fid_cosmo = Cosmology()
N = 100
latent_x = Vector(0:0.3:3)
x = Vector(range(0., stop=3., length=N))

struct Determin{T<:Any} <: ContinuousUnivariateDistribution
  val::T
end

Distributions.rand(rng::AbstractRNG, d::Determin) = d.val
Distributions.logpdf(d::Determin, x::T) where T<:Real = zero(x)
Bijectors.bijector(d::Determin) = Identity{0}()

@model function model(data_vector; cov_tot=cov_tot, fid_cosmo=fid_cosmo,
                      latent_x=latent_x, x=x)
    eta = 0.3 #~ Uniform(0.0, 0.5)
    l = 1 #~ Uniform(0.1, 3)
    latent_N = length(latent_x)
    v ~ MvNormal(zeros(latent_N), ones(latent_N))
    
    Ωm ~ Uniform(0.1, 0.5)
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

cycles = 10
iterations = 500
TAP = 0.60
adaptation = 1000

# Start sampling.
folpath = "../chains"
folname = string("DES_simple_test_gp_", "TAP", TAP)
folname = joinpath(folpath, folname)

mkdir(folname)
println("Created new folder")

for i in 1:cycles
    if i == 1
        chain = sample(model(data_vector), NUTS(adaptation, TAP; init_ϵ=0.035), 
                       iterations, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1, ".jls")), Chains)
        chain = sample(model(data_vector), NUTS(0, TAP; init_ϵ=0.035), 
                       iterations, progress=true; save_state=true,
                       resume_from=old_chain)
    end 
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
