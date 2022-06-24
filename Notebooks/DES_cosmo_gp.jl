using Turing
using LimberJack
using CSV
using NPZ
using FITSIO
using Dates
using Distances
using LinearAlgebra

println(Threads.nthreads())

files = npzread("../data/DESY1_cls/Cls_meta.npz")
Cls_meta = cls_meta(files)
cov_tot = files["cov"]
data_vector = files["cls"]

sqexp_cov_fn(D, mu, phi) = @.(mu * exp(-D^2 / (2*phi))) + 0.005 * I
fid_cosmo = Cosmology()

@model function model(data_vector; cov_tot=cov_tot, fid_cosmo=fid_cosmo)
    eta ~ Uniform(0.0, 0.5)
    l ~ Uniform(0.1, 3)
    
    Ωm ~ Uniform(0.1, 0.6)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    s8 = 0.81 #~ Uniform(0.6, 1.0)
    ns ~ Uniform(0.87, 1.07)

    nuisances = Dict()
    
    N = 100
    latent_x = Vector(0:0.3:3)
    x = zeros(100, 1)
    x[:, 1] = Vector(range(0., stop=3., length=N))
    latent_N = length(latent_x)
    total_N = N + latent_N
    total_x = [x; latent_x]
    # Distance matrix.
    D = pairwise(Distances.Euclidean(), total_x, dims=1)
    
    # Set up GP
    K = sqexp_cov_fn(D, eta, l)
    Koo = K[(N+1):end, (N+1):end] # GP-GP cov
    Knn = K[1:N, 1:N]             # Data-Data cov
    Kno = K[1:N, (N+1):end]       # Data-GP cov
    
    latent_Dz = fid_cosmo.Dz(vec(latent_x))
    #rand(MvNormal(zeros(gp_N), 0.1*ones(gp_N)))
    latent_gp = latent_Dz .+ cholesky(Koo).U' * v
    # Conditional 
    Koo_inv = inv(Koo)            
    C = Kno * Koo_inv
    gp = C * latent_gp
    
    cosmology = Cosmology(Ωm, Ωb, h, ns, s8,
                          tk_mode="EisHu",
                          Pk_mode="Halofit", 
                          custom_Dz=gp)
    
    theory = Theory(cosmology, Cls_meta, files;
                    Nuisances=nuisances).cls
    data_vector ~ MvNormal(theory, cov_tot)
end;

iterations = 5000
TAP = 0.60
adaptation = 1000

# Start sampling.
folpath = "../chains"
folname = string("DES_cosmo_gp_", "TAP", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    println("Removed folder")
    rm(folname)
end

mkdir(folname)
println("Created new folder")

new_chain = sample(model(data_vector), NUTS(adaptation, TAP),
                   iterations, progress=true; save_state=true)

summary = describe(new_chain)[1]
fname_summary = string("summary", now(), ".csv")
CSV.write(joinpath(folname, fname_summary), summary)

fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chain)
    
fname_csv = string("chain_", now(), ".csv")
CSV.write(joinpath(folname, fname_csv), new_chain)
