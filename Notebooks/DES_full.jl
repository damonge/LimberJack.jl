using Turing
using LimberJack
using CSV
using NPZ
using FITSIO

cl_path1 = "../data/DESY1_cls/DESgc_DESgc/"
cl_path2 = "../data/DESY1_cls/DESgc_DESwl/"
cl_path3 = "../data/DESY1_cls/DESwl_DESwl/"
cov_path = "../data/DESY1_cls/covs/"
datas = [Data("DESgc", "DESgc", 0, 0, cl_path=cl_path1, cov_path=cov_path),
          Data("DESgc", "DESgc", 1, 1, cl_path=cl_path1, cov_path=cov_path),
          Data("DESgc", "DESgc", 2, 2, cl_path=cl_path1, cov_path=cov_path),
          Data("DESgc", "DESgc", 3, 3, cl_path=cl_path1, cov_path=cov_path),
          Data("DESgc", "DESgc", 4, 4, cl_path=cl_path1, cov_path=cov_path),   
          Data("DESgc", "DESwl", 0, 0, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 1, 0, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 2, 0, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 3, 0, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 4, 0, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 1, 1, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 2, 1, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 3, 1, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 4, 1, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 2, 2, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 3, 2, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 4, 2, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 3, 3, cl_path=cl_path2, cov_path=cov_path),
          Data("DESgc", "DESwl", 4, 3, cl_path=cl_path2, cov_path=cov_path)]; 
Nzs = [Nz(1), Nz(2), Nz(3), Nz(4), Nz(5)]
Cls_metas = Cls_meta(datas, covs_path=cov_path);
cov_tot = Cls_metas.cov_tot;
data_vector = Cls_metas.data_vector;

@model function model(data_vector)
    Ωm ~ Uniform(0.2, 0.3)
    h ~ Uniform(0.6, 0.8)
    s8 ~ Uniform(0.7, 1.0)
    
    b1 ~ Uniform(1.0, 3.0)
    b2 ~ Uniform(1.0, 3.0)
    b3 ~ Uniform(1.0, 3.0)
    b4 ~ Uniform(1.0, 3.0)
    b5 ~ Uniform(1.0, 3.0)
    
    nuisances = Dict("b1" => b1,
                     "b2" => b2,
                     "b3" => b3,
                     "b4" => b4,
                     "b5" => b5)
    
    cosmology = LimberJack.Cosmology(Ωm, 0.05, h, 0.96, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    theory = Theory(cosmology, Cls_metas, Nzs, nuisances).Cls
    data_vector ~ MvNormal(theory, cov_tot)
end;

iterations = 500
TAP = 0.7
adaptation = 100
cores = 4

# Start sampling.
folpath = "../chains"
folname = string("DES_full_test_", "NUTS_TAP_", TAP)
folname = joinpath(folpath, folname)
if isdir(folname)
    println("Folder already exists")
    if isfile(joinpath(folname, "chain.jls"))
        println("Restarting from past chain")
        past_chain = read(joinpath(folname, "chain.jls"), Chains)
        new_chain = sample(model(data_vector), NUTS(adaptation, TAP), iterations,
                           progress=true; save_state=true, resume_from=past_chain)
    else
        new_chain = sample(model(data_vector), NUTS(adaptation, TAP),
                    iterations, progress=true; save_state=true)
    end
else
    mkdir(folname)
    println("Created new folder")
    new_chain = sample(model(data_vector), NUTS(adaptation, TAP),
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
CSV.write(joinpath(folname, fname_csv), new_chain)
