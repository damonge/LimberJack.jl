using Distributed

@everywhere using Turing
@everywhere using LimberJack
@everywhere using GaussianProcess
@everywhere using CSV
@everywhere using NPZ

@everywhere println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

@everywhere sacc_path = "../../data/FD/cls_FD_covG.fits"
@everywhere yaml_path = "../../data/ND/ND.yml"
@everywhere meta, files = make_data(sacc_path, yaml_path)
@everywhere cls_data = meta.cls
    
@everywhere fs8_meta = npzread("../data/fs8s/fs8s.npz")
@everywhere fs8_zs = fs8_meta["z"]
@everywhere fs8_data = fs8_meta["data"]
@everywhere fs8_cov = fs8_meta["cov"]

@everywhere cov_tot = zeros(Float64, length(fs8_data)+length(cls_data), length(fs8_data)+length(cls_data))
@everywhere cov_tot[1:length(fs8_data), 1:length(fs8_data)] = fs8_cov
@everywhere cov_tot[length(fs8_data)+1:(length(fs8_data)+length(cls_data)),
            length(fs8_data)+1:(length(fs8_data)+length(cls_data))] = cls_cov
@everywhere data_tot = [fs8_data ; cls_data];
    
@everywhere errs = sqrt.(diag(cov_tot))
@everywhere data = data_tot ./ errs
@everywhere fake_cov = Hermitian(cov_tot ./ (errs * errs'));

@everywhere fid_cosmo = Cosmology()
@everywhere n = 101
@everywhere N = 201
@everywhere latent_x = Vector(LinRange(0, 3, n))
@everywhere x = Vector(LinRange(0, 3, N))

@everywhere @model function model(;
                          names=meta.names,
                          types=meta.types,
                          pairs=meta.pairs,
                          idx=meta.idx,
                          cov=fake_cov, 
                          files=files,
                          fid_cosmo=fid_cosmo,
                          latent_x=latent_x,
                          x=x)

    #DESY1 priors
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

    eta ~ Uniform(0.01, 0.1) # = 0.2
    l ~ Uniform(0.1, 4) # = 0.3
    v ~ filldist(truncated(Normal(0, 1), -3, 3), n)

    mu = fid_cosmo.Dz(vec(latent_x))
    K = sqexp_cov_fn(latent_x; eta=eta, l=l)
    latent_gp = latent_GP(mu, v, K)
    gp = conditional(latent_x, x, latent_gp, sqexp_cov_fn;
                      eta=eta, l=l)

    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="emulator",
                                     Pk_mode="Halofit";
                                     custom_Dz=[x, gp])

    cls = Theory(cosmology, names, types, pairs,
                    idx, files; Nuisances=nuisances)
    fs8s = fs8(cosmology, fs8_zs)
    theory = [fs8s; cls]

    data ~ MvNormal(theory ./ errs, cov)
end

@everywhere stat_model = model()|(;data)

@everywhere cycles = 6
@everywhere iterations = 100

@everywhere TAP = 0.65
@everywhere adaptation = 100
@everywhere sampler = NUTS(adaptation, TAP)
@everywhere nchains = nprocs()

# Start sampling.
folpath = "../chains"
folname = string("ND_RSD_super_gp_TAP_", TAP)
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
        chain = sample(stat_model, sampler, MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(stat_model, sampler, MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true, resume_from=old_chain)
    end 
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
