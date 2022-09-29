"""
    Cℓintegrand(cosmo::Cosmology, t1::Tracer, t2::Tracer, logk, ℓ)
Returns the integrand of the angular power spectrum. 
Arguments:
- `cosmo::Cosmology` : cosmology structure.
- `t1::Tracer` : tracer structure.
- `t2::Tracer` : tracer structure.
- `logk::Vector{Float}` : log scale array.
- `ℓ::Float` : multipole.
Returns:
- `integrand::Vector{Real}` : integrand of the angular power spectrum.
"""
function Cℓintegrand(cosmo::Cosmology,
                     t1::Tracer,
                     t2::Tracer,
                     logk,
                     ℓ)

    chis = zeros(cosmo.settings.cosmo_type, cosmo.settings.nk)
    chis[1:cosmo.settings.nk] = (ℓ+0.5) ./ cosmo.ks
    chis .*= (chis .< cosmo.chi_max)

    z = cosmo.z_of_chi(chi)
    w1 = t1.wint(chis) # *t1.b
    w2 = t2.wint(chis) # *t2.b

    if typeof(t1) in [NumberCountsTracer, WeakLensingTracer]
        w1 .*= t1.b
    end

    if typeof(t2) in [NumberCountsTracer, WeakLensingTracer]
        w2 .*= t2.b
    end

    pk = nonlin_Pk(cosmo, k, z)
    return @. (k*w1*w2*pk)
end

"""
    angularCℓs(cosmo::Cosmology, t1::Tracer, t2::Tracer, ℓs)

Returns the angular power spectrum. 

Arguments:

- `cosmo::Cosmology` : cosmology structure.
- `t1::Tracer` : tracer structure.
- `t2::Tracer` : tracer structure.
- `ℓs::Vector{Float}` : multipole array.

Returns:
- `Cℓs::Vector{Real}` : angular power spectrum.

"""
function angularCℓs(cosmo::Cosmology, t1::Tracer, t2::Tracer, ℓs)
    # OPT: we are not optimizing the limits of integration
    Cℓs = zeros(cosmo.settings.cosmo_type, length(ℓs))
    for i in 1:length(ℓs)
        ℓ = ℓs[i]
        integrand = Cℓintegrand(cosmo, t1, t2, ℓ)/(ℓ+0.5)
        #for j in 1:length(logks)
        #    logk = logks[j]
        #    integrand[j] = Cℓintegrand(cosmo, t1, t2, logk, ℓ)/(ℓ+0.5)
        #end
        Cℓ = trapz(cosmo.logks, integrand)
        fℓ1 = _get_Fℓ(t1, ℓ)
        fℓ2 = _get_Fℓ(t2, ℓ)
        Cℓs[i] = Cℓ * fℓ1 * fℓ2
    end
    return Cℓs
end

function _get_Fℓ(t::Tracer, ℓ::Real)
    if typeof(t) == WeakLensingTracer
        return @. sqrt((ℓ+2)*(ℓ+1)*ℓ*(ℓ-1))/(ℓ+0.5)^2
    elseif typeof(t) == CMBLensingTracer
        return @. (ℓ+1)*ℓ/(ℓ+0.5)^2
    else
        return 1
    end
end
