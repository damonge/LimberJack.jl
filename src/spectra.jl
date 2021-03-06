function Cℓintegrand(cosmo::Cosmology,
                     t1::Tracer,
                     t2::Tracer,
                     logk,
                     ℓ)
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

function angularCℓs(cosmo::Cosmology, t1::Tracer, t2::Tracer, ℓs)
    # OPT: we are not optimizing the limits of integration
    cosmo_type = cosmo.settings.cosmo_type
    logks = cosmo.logk
    dlogk = cosmo.dlogk
    res = length(logks)
    Cℓs = zeros(cosmo_type, length(ℓs))
    for i in 1:length(ℓs)
        ℓ = ℓs[i]
        integrand = zeros(cosmo_type, length(logks))
        for j in 1:length(logks)
            logk = logks[j]
            integrand[j] = Cℓintegrand(cosmo, t1, t2, logk, ℓ)/(ℓ+0.5)
        end
        #integrand = [Cℓintegrand(cosmo, t1, t2, logk, ℓ)/(ℓ+0.5) for logk in logks]
        #Cℓ = sum(0.5 .* (integrand[1:res-1] .+ integrand[2:res]) .* dlogk)
        Cℓ = trapz(logks, integrand)
        fℓ1 = get_Fℓ(t1, ℓ)
        fℓ2 = get_Fℓ(t2, ℓ)
        Cℓs[i] = Cℓ * fℓ1 * fℓ2
    end
    return Cℓs
end
