
function Cℓintegrand(cosmo::Cosmology,
                     t1::Tracer,
                     t2::Tracer,
                     logk::Float64,
                     ℓ::Float64)
    k = exp(logk)
    chi = (ℓ+0.5)/k
    if chi > cosmo.chi_max
        return 0
    end
    z = cosmo.z_of_chi(chi)
    w1 = t1.wint(chi)*t1.bias
    w2 = t2.wint(chi)*t2.bias
    pk = nonlin_Pk(cosmo, k, z)
    k*w1*w2*pk
end

function angularCℓ(cosmo::Cosmology, t1::Tracer, t2::Tracer, ℓ::Float64; res=200)
    # OPT: we are not optimizing the limits of integration
    logks = LinRange(log(10^-4),log(10^2), res)
    dlogk = logks[2]-logks[1]
    integrand = [Cℓintegrand(cosmo, t1, t2, logk, ℓ)/(ℓ+0.5) for logk in logks]
    Cℓ = sum(0.5 .* (integrand[1:res-1] .+ integrand[2:res]) .* dlogk)
    fℓ1 = get_Fℓ(t1, ℓ)
    fℓ2 = get_Fℓ(t2, ℓ)
    return Cℓ * fℓ1 * fℓ2
end

function angularCℓs(cosmo::Cosmology, t1::Tracer, t2::Tracer, ℓs::Vector{Float64}; res=200)
    # OPT: we are not optimizing the limits of integration
    logks = LinRange(log(10^-4),log(10^2), res)
    dlogk = logks[2]-logks[1]
    Cℓs = Vector{Union{Real, ForwardDiff.Dual{Nothing, Float64, 2}}}(undef, length(ℓs))
    for i in 1:length(ℓs)
        ℓ = ℓs[i]
        integrand = [Cℓintegrand(cosmo, t1, t2, logk, ℓ)/(ℓ+0.5) for logk in logks]
        Cℓ = sum(0.5 .* (integrand[1:res-1] .+ integrand[2:res]) .* dlogk)
        fℓ1 = get_Fℓ(t1, ℓ)
        fℓ2 = get_Fℓ(t2, ℓ)
        Cℓs[i] = Cℓ * fℓ1 * fℓ2
    end
    return Cℓs
end
