
function Cℓintegrand(cosmo::Cosmology,
                     t1::Tracer,
                     t2::Tracer,
                     logk::Float64, ℓ::Int64)
    k = exp(logk)
    chi = (ℓ+0.5)/k
    if chi > cosmo.chi_max
        return 0
    end
    z = cosmo.z_of_chi(chi)
    w1 = t1.wint(chi)*t1.bias
    w2 = t2.wint(chi)*t2.bias
    pk = power_spectrum(cosmo, k, z)
    k*w1*w2*pk
end

function angularCℓ(cosmo::Cosmology, t1::Tracer, t2::Tracer, ℓ)
    # OPT: we are not optimizing the limits of integration
    Cℓ = quadgk(lk -> Cℓintegrand(cosmo, t1, t2, lk, ℓ),
                log(10^-4), log(10^2), rtol=1E-5)[1]/(ℓ+0.5)
    fℓ1 = get_Fℓ(t1, ℓ)
    fℓ2 = get_Fℓ(t2, ℓ)
    return Cℓ * fℓ1 * fℓ2
end
