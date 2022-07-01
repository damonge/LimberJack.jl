using Turing
using LimberJack
using CSV
using NPZ
using FITSIO
using Dates
using GaussianProcess

println(Threads.nthreads())

files = npzread("../data/DESY1_cls/Cls_meta.npz")
Cls_meta = cls_meta(files)
cov_tot = files["cov"]
data_vector = files["cls"]

fid_cosmo = Cosmology()
N = 100
latent_x = Vector(0:0.3:3)
x = Vector(range(0., stop=3., length=N))

@model function model(data_vector; cov_tot=cov_tot, fid_cosmo=fid_cosmo,
                      latent_x=latent_x, x=x)
    eta ~ Uniform(0.0, 0.5)
    l ~ Uniform(0.1, 3)
    latent_N = length(latent_x)
    v ~ MvNormal(zeros(latent_N), ones(latent_N))
    
    Ωm ~ Uniform(0.1, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    s8 ~ Uniform(0.6, 1.0)
    ns ~ Uniform(0.87, 1.07)
    
    nuisances = Dict()
    
    mu = fid_cosmo.Dz(vec(latent_x))
    K = sqexp_cov_fn(latent_x; eta=eta, l=l)
    latent_gp = latent_GP(mu, v, K)
    gp =  conditional(latent_x, x, latent_gp, sqexp_cov_fn;
                      eta=eta, l=l)
    
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
#nchains = Threads.nthreads()


new_chain = sample(model(data_vector), NUTS(adaptation, TAP), 
                   iterations, progress=true; save_state=true)

# Start sampling.
folpath = "../chains"
folname = string("DES_cosmo_gp_", "TAP", TAP)
folname = joinpath(folpath, folname)

mkdir(folname)
println("Created new folder")

for i in 1:cycles
    if i == 1
        chain = sample(model(data_vector), NUTS(adaptation, TAP), 
                       iterations, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1, ".jls")), Chains)
        chain = sample(model(data_vector), NUTS(adaptation, TAP), 
                       iterations, progress=true; save_state=true,
                       resume_from=old_chain)
    end 
    write(joinpath(folname, string("chain_", i,".jls")), chain)
end