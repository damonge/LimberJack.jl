using Turing
using LimberJack
using CSV
using NPZ
using FITSIO

cl_path = "../data/DESY1_cls/DESgc_DESgc"
cov_path = "../data/DESY1_cls/covs"
datas = [Data("DESgc", "DESgc", 0, 0, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 0, 1, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 0, 2, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 0, 3, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 0, 4, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 1, 1, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 1, 2, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 1, 3, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 1, 4, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 2, 2, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 2, 3, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 2, 4, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 3, 3, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 3, 4, cl_path=cl_path, cov_path=cov_path),
         Data("DESgc", "DESgc", 4, 4, cl_path=cl_path, cov_path=cov_path)];
Nzs = [Nz(1), Nz(2), Nz(3), Nz(4), Nz(5)]
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
    theory = Theory(cosmology, Cls_metas, Nzs).Cls
    data_vector ~ MvNormal(theory, cov_tot)
end;

iterations = 500
step_size = 0.005
samples_per_step = 10
cores = 4

# Start sampling.
folpath = "../chains"
folname = string("DES_gcgc_", "stpsz_", step_size, "_smpls_", samples_per_step)
folname = joinpath(folpath, folname)
if isdir(folname)
    println("Folder already exists")
    if isfile(joinpath(folname, "chain.jls"))
        println("Restarting from past chain")
        past_chain = read(joinpath(folname, "chain.jls"), Chains)
        new_chain = sample(model(data_vector), HMC(step_size, samples_per_step), iterations,
                           progress=true; save_state=true, resume_from=past_chain)
    else
        new_chain = sample(model(data_vector), HMC(step_size, samples_per_step),
                    iterations, progress=true; save_state=true)
    end
else
    mkdir(folname)
    println("Created new folder")
    new_chain = sample(model(data_vector), HMC(step_size, samples_per_step),
                iterations, progress=true; save_state=true)
end

cov_names_mat = Tables.table(Cls_metas.cov_names_mat)
fname_info = string("cov_names_mat.csv")
CSV.write(joinpath(folname, fname_info), cov_names_mat,
                   writeheader=Cls_metas.cls_names)

summary = describe(new_chain)[1]
fname_summary = string("summary.csv")
CSV.write(joinpath(folname, fname_summary), summary)

fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chain)
    
fname_csv = string("chain.csv")
CSV.write(joinpath(folname, fname_csv), new_chain))
