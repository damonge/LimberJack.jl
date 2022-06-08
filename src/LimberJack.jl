module LimberJack

export Settings, CosmoPar, Cosmology
export Ez, Hmpc, comoving_radial_distance, growth_factor
export NumberCountsTracer, WeakLensingTracer, CMBLensingTracer, get_Fℓ
export angularCℓs, angularCℓ, lin_Pk, nonlin_Pk
export Theory, Theory_parallel, cls_meta, fill_NuisancePars

using Interpolations, QuadGK, OrdinaryDiffEq, ForwardDiff, LinearAlgebra, PyCall

include("core.jl")
include("tracers.jl")
include("spectra.jl")
include("halofit.jl")
include("theory.jl")


end
