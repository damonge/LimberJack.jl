using Turing
using LimberJack
using CSV
using NPZ
using FITSIO

path = "data"
datas = [Data("DESgc", "DESgc", 2 , 2, path=path),
         Data("DESgc", "DESwl", 2 , 3, path=path)];
Cls_metas = Cls_meta(datas, path=path);
cov_tot = Cls_metas.cov_tot;
data_vector = Cls_metas.data_vector;

@model function model(data_vector)
    Ωm ~ Uniform(0.2, 0.3)
    h ~ Uniform(0.6, 0.8)
    s8 ~ Uniform(0.7, 1.0)
    cosmology = LimberJack.Cosmology(Ωm, 0.05, h, 0.96, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    theory = get_theory(cosmology, datas, path=path)
    data_vector ~ MvNormal(theory, cov_tot)
end;

iterations = 500
step_size = 0.005
samples_per_step = 10
cores = 4

# Start sampling.
folname = string("DES_gc2gc2_gc2wl3_", "stpsz_", step_size, "_smpls_", samples_per_step)
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