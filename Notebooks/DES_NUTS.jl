using Turing
using LimberJack
using CSV
using NPZ
using FITSIO

files = npzread("../data/DESY1_cls/Cls_meta.npz")
Cls_meta = cls_meta(files)
cov_tot = files["cov"]
data_vector = files["cls"]

@model function model(data_vector; cov_tot=cov_tot)
    Ωm ~ Uniform(0.2, 0.3)
    h ~ Uniform(0.6, 0.8)
    s8 ~ Uniform(0.7, 1.0)
    
    b0 ~ Uniform(1.0, 3.0)
    b1 ~ Uniform(1.0, 3.0)
    b2 ~ Uniform(1.0, 3.0)
    b3 ~ Uniform(1.0, 3.0)
    b4 ~ Uniform(1.0, 3.0)
    
    nuisances = Dict("b0" => b0,
                     "b1" => b1,
                     "b2" => b2,
                     "b3" => b3,
                     "b4" => b4)
    
    cosmology = LimberJack.Cosmology(Ωm, 0.05, h, 0.96, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    theory = Theory(cosmology, nuisances, Cls_meta, files)
    data_vector ~ MvNormal(theory, cov_tot)
end;

iterations = 500
TAP = 0.70
adaptation = 100

# Start sampling.
folpath = "../chains"
folname = string("DES_NUTS_", "TAP", TAP)
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

info = describe(new_chain)[1]
fname_info = string("info.csv")
CSV.write(joinpath(folname, fname_info), info)


fname_jls = string("chain.jls")
write(joinpath(folname, fname_jls), new_chain)
    
fname_csv = string("chain.csv")
CSV.write(joinpath(folname, fname_csv), new_chain)
