using Turing
using Random
using ForwardDiff
using LimberJack
using CSV

zs = 0.02:0.02:1.0
ks = [0.001, 0.01, 0.1, 1.0, 10.0]
ℓs = [10, 30, 100, 300]

true_cosmology = LimberJack.Cosmology(0.25, 0.05, 0.67, 0.96, 0.81, tk_mode="EisHu");
nz = @. exp(-0.5*((true_cosmology.zs-0.5)/0.05)^2)
tg = NumberCountsTracer(true_cosmology, true_cosmology.zs, nz, 2.)
ts = WeakLensingTracer(true_cosmology, true_cosmology.zs, nz)
true_data = [angularCℓ(true_cosmology, tg, ts, ℓ) for ℓ in ℓs]
data = true_data + 0.1 * true_data .* rand(length(true_data)) 

@model function model(data)
    Ωm ~ Uniform(0.2, 0.3)
    cosmology = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.8, tk_mode="EisHu")
    tg = NumberCountsTracer(cosmology, cosmology.zs, nz, 2.)
    ts = WeakLensingTracer(cosmology, cosmology.zs, nz)
    predictions = [angularCℓ(cosmology, tg, ts, ℓ) for ℓ in ℓs]
    data ~ MvNormal(predictions, 0.01.*data)
end;

iterations = 100
ϵ = 0.05
τ = 10

# Start sampling.
chain = sample(model(data), HMC(ϵ, τ), iterations, progress=true)
CSV.write("Cl_EisHu_chain.csv", chain)
