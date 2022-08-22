module LimberJack

export Settings, CosmoPar, Cosmology
export Ez, Hmpc, comoving_radial_distance, growth_factor
export NumberCountsTracer, WeakLensingTracer, CMBLensingTracer, get_Fℓ
export angularCℓs, angularCℓ, lin_Pk, nonlin_Pk
export Theory, cls_meta, get_nzs
export get_emulated_log_pk0, Emulator

using Interpolations, OrdinaryDiffEq, ForwardDiff 
using LinearAlgebra, Statistics, PyCall, Trapz, QuadGK, NPZ

include("core.jl")
include("tracers.jl")
include("spectra.jl")
include("halofit.jl")
include("theory.jl")
include("emulator.jl")


end
