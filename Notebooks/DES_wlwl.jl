using Turing
using LimberJack
using CSV
using NPZ
using FITSIO

cl_path = "/mnt/extraspace/gravityls_3/S8z/Cls_new_pipeline/512_DES_eBOSS_CMB/DESwl_DESwl"
cov_path = "/mnt/extraspace/gravityls_3/S8z/Cls_new_pipeline/512_DES_eBOSS_CMB/cov"
nz_path = "data"
datas = [Data("DESwl", "DESwl", 1, 1, cl_path=cl_path, cov_path=cov_path),
         Data("DESwl", "DESwl", 1, 2, cl_path=cl_path, cov_path=cov_path),
         Data("DESwl", "DESwl", 1, 3, cl_path=cl_path, cov_path=cov_path),
         #Data("DESwl", "DESwl", 1, 4, cl_path=cl_path, cov_path=cov_path),
         Data("DESwl", "DESwl", 2, 2, cl_path=cl_path, cov_path=cov_path),
         Data("DESwl", "DESwl", 2, 3, cl_path=cl_path, cov_path=cov_path),
         #Data("DESwl", "DESwl", 2, 4, cl_path=cl_path, cov_path=cov_path),
         Data("DESwl", "DESwl", 3, 3, cl_path=cl_path, cov_path=cov_path)]; #,
         #Data("DESwl", "DESwl", 3, 4, cl_path=cl_path, cov_path=cov_path),
         #Data("DESwl", "DESwl", 4, 4, cl_path=cl_path, cov_path=cov_path)];
Nzs = [Nz(1), Nz(2), Nz(3), Nz(4)]
Cls_metas = Cls_meta(datas, covs_path=cov_path);
cov_tot = Cls_metas.cov_tot;
data_vector = Cls_metas.data_vector;

@model function model(data_vector)
    Ωm ~ Uniform(0.2, 0.3)
    h ~ Uniform(0.6, 0.8)
    s8 ~ Uniform(0.7, 1.0)
    cosmology = LimberJack.Cosmology(Ωm, 0.05, h, 0.96, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    theory = Theory(cosmology, datas, Nzs)
    data_vector ~ MvNormal(theory, cov_tot)
end;

iterations = 500
step_size = 0.005
samples_per_step = 10
cores = 4

# Start sampling.
folname = string("DES_wlwl_", "stpsz_", step_size, "_smpls_", samples_per_step)
if isdir(folname)
    println("Folder already exists")
    if isfile(joinpath(folname, "chain.jls"))
        println("Restarting from past chain")
        past_chain = read(joinpath(folname, "chain.jls"), Chains)
        new_chain = sample(model(data_vector), HMC(step_size, samples_per_step), iterations,
                           progress=true; save_state=true, resume_from=past_chain)
    end
else
    mkdir(folname)
    println("Created new folder")
    new_chain = sample(model(data_vector), HMC(step_size, samples_per_step),
                iterations, progress=true; save_state=true)
end

info = describe(new_chain)[1]
fname_info = string("info.csv")
CSV.write(joinpath(folname, fname_info), info)


fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chain)
    
fname_csv = string("chain.csv")
CSV.write(joinpath(folname, fname_csv), new_chain)
