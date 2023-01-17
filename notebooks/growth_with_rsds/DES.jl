using Distributed

@everywhere begin
    using LinearAlgebra
    using Turing
    using AdvancedHMC
    using LimberJack
    using CSV
    using YAML
    using PythonCall
    sacc = pyimport("sacc");

    println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

    sacc_path = "../../data/FD/cls_FD_covG.fits"
    yaml_path = "../../data/DESY1/gcgc_gcwl_wlwl.yml"
    sacc_file = sacc.Sacc().load_fits(sacc_path)
    yaml_file = YAML.load_file(yaml_path)
    meta, files = make_data(sacc_file, yaml_file)

    data_vector = meta.data
    cov_tot = meta.cov
    errs = sqrt.(diag(cov_tot))
    fake_data = data_vector ./ errs
    fake_cov = Hermitian(cov_tot ./ (errs * errs'));
end 

@everywhere @model function model(data;
                                  cov=fake_cov,
                                  meta=meta, 
                                  files=files)
    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ TruncatedNormal(72, 5, 0.64, 0.82)
    s8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)

    DESgc__0_b ~ Uniform(0.8, 3.0)
    DESgc__1_b ~ Uniform(0.8, 3.0)
    DESgc__2_b ~ Uniform(0.8, 3.0)
    DESgc__3_b ~ Uniform(0.8, 3.0)
    DESgc__4_b ~ Uniform(0.8, 3.0)
    DESgc__0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__1_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__2_dz ~ TruncatedNormal(0.0, 0.006, -0.2, 0.2)
    DESgc__3_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    DESgc__4_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    DESwl__0_dz ~ TruncatedNormal(-0.001, 0.016, -0.2, 0.2)
    DESwl__1_dz ~ TruncatedNormal(-0.019, 0.013, -0.2, 0.2)
    DESwl__2_dz ~ TruncatedNormal(0.009, 0.011, -0.2, 0.2)
    DESwl__3_dz ~ TruncatedNormal(-0.018, 0.022, -0.2, 0.2)
    DESwl__0_m ~ Normal(0.012, 0.023)
    DESwl__1_m ~ Normal(0.012, 0.023)
    DESwl__2_m ~ Normal(0.012, 0.023)
    DESwl__3_m ~ Normal(0.012, 0.023)
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)

    nuisances = Dict("DESgc__0_b" => DESgc__0_b,
                     "DESgc__1_b" => DESgc__1_b,
                     "DESgc__2_b" => DESgc__2_b,
                     "DESgc__3_b" => DESgc__3_b,
                     "DESgc__4_b" => DESgc__4_b,
                     "DESgc__0_dz" => DESgc__0_dz,
                     "DESgc__1_dz" => DESgc__1_dz,
                     "DESgc__2_dz" => DESgc__2_dz,
                     "DESgc__3_dz" => DESgc__3_dz,
                     "DESgc__4_dz" => DESgc__4_dz,
                     "DESwl__0_dz" => DESwl__0_dz,
                     "DESwl__1_dz" => DESwl__1_dz,
                     "DESwl__2_dz" => DESwl__2_dz,
                     "DESwl__3_dz" => DESwl__3_dz,
                     "DESwl__0_m" => DESwl__0_m,
                     "DESwl__1_m" => DESwl__1_m,
                     "DESwl__2_m" => DESwl__2_m,
                     "DESwl__3_m" => DESwl__3_m,
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA,)

    cosmology = Cosmology(Ωm, Ωb, h, ns, s8,
                          tk_mode="EisHu",
                          Pk_mode="Halofit")

    theory = Theory(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(theory ./ errs, cov)
end

stats_model = model(fake_data)
MAP_vals = [ 2.89781002e-01,  4.77686400e-02,  7.38011259e-01,  7.98425957e-01,
        9.69728709e-01,  1.49721568e+00,  1.81865817e+00,  1.79001308e+00,
        2.18263403e+00,  2.24598151e+00, -3.46696435e-03, -3.77762873e-03,
       -1.60856663e-03, -2.09706269e-03,  5.92820210e-04, -1.05679892e-02,
        5.08483360e-03,  4.75152724e-03, -8.16263963e-04,  1.90524810e-02,
        3.83250992e-03,  2.35920245e-02, -1.54272168e-03,  3.16431800e-01,
       -2.64120253e-01]
MAP_names = ["Ωm", "Ωb", "h", "s8",  "ns",
              "DESgc__0_b", "DESgc__1_b","DESgc__2_b","DESgc__3_b","DESgc__4_b",
              "DESgc__0_dz", "DESgc__1_dz", "DESgc__2_dz", "DESgc__3_dz", "DESgc__4_dz",
              "DESwl__0_dz", "DESwl__1_dz", "DESwl__2_dz", "DESwl__3_dz",
              "DESwl__0_m", "DESwl__1_m", "DESwl__2_m", "DESwl__3_m",
              "A_IA", "alpha_IA",];
loglike = Loglike(stats_model, MAP_names)
mass_matrix = get_mass_matrix(loglike, MAP_vals)
metric = DenseEuclideanMetric(hess_cov)
sampler = Sampler(NUTS(adaptation, TAP; metric=metric),
                  stats_model)

cycles = 6
iterations = 1000
nchains = nprocs()

adaptation = 1000
TAP = 0.65
init_ϵ = 0.01

sampler = Turing.NUTS(adaptation, TAP;
                   init_ϵ = init_ϵ,
                   metricT=AdvancedHMC.DenseEuclideanMetric)

println("sampling settings: ")
println("cycles ", cycles)
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)
println("nchains ", nchains)

# Start sampling.
folpath = "../../chains"
folname = string("DESY1_k1k_priors_EisHu_dense")
folname = joinpath(folpath, folname)

if isdir(folname)
    fol_files = readdir(folname)
    println("Found existing file ", folname)
    if length(fol_files) != 0
        last_chain = last([file for file in fol_files if occursin("chain", file)])
        last_n = parse(Int, last_chain[7])
        println("Restarting chain")
    else
        println("Starting new chain")
        last_n = 0
    end
else
    mkdir(folname)
    println(string("Created new folder ", folname))
    last_n = 0
end

for i in (1+last_n):(cycles+last_n)
    if i == 1
        chain = sample(stats_model, sampler, MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(stats_model, sampler, MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true, resume_from=old_chain)
    end  
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
