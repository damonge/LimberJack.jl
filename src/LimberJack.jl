module LimberJack

export CosmoPar, Cosmology
export Ez, Hmpc, comoving_radial_distance, growth_factor
export NumberCountsTracer, WeakLensingTracer, CMBLensingTracer, get_Fℓ
export angularCℓs, angularCℓ, lin_Pk, nonlin_Pk
export Theory, Theory_parallel, cls_meta

using Interpolations, OrdinaryDiffEq, ForwardDiff, LinearAlgebra, QuadGK, Trapz

include("core.jl")
include("tracers.jl")
include("spectra.jl")
include("halofit.jl")
include("theory.jl")


end
