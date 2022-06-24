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

sqexp_cov_fn(D, eta, l) = @.(eta * exp(-D^2 / (2*l))) + 0.005 * LinearAlgebra.I

@model function model(data_vector; cov_tot=cov_tot)
    eta ~ Uniform(0.0, 0.5)
    l ~ Uniform(0.1, 3)
    
    Ωm ~ Uniform(0.1, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    s8 = 0.81 #~ Uniform(0.6, 1.0)
    ns ~ Uniform(0.87, 1.07)
    
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
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
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
#nchains = Threads.nthreads()

# Start sampling.
folpath = "../chains"
folname = string("DES_full_gp_", "TAP", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    println("Removed folder")
    rm(folname)
end

mkdir(folname)
println("Created new folder")

new_chain = sample(model(data_vector), NUTS(adaptation, TAP), 
                   iterations, progress=true; save_state=true)

#new_chain = sample(model(data_vector), NUTS(adaptation, TAP), MCMCThreads(),
#                   iterations, nchains, progress=true; save_state=true,
#                   resume_from=past_chain)

summary = describe(new_chain)[1]
fname_summary = string("summary", now(), ".csv")
CSV.write(joinpath(folname, fname_summary), summary)

fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chain)
    
fname_csv = string("chain_", now(), ".csv")
CSV.write(joinpath(folname, fname_csv), new_chain)
