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
    k = exp.(logk)
    chi = (ℓ+0.5) ./ k
    chi .*= (chi .< cosmo.chi_max)

    z = cosmo.z_of_chi(chi)
    w1 = t1.wint(chi) # *t1.b
    w2 = t2.wint(chi) # *t2.b

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
    cosmo_type = cosmo.settings.cosmo_type
    logks = cosmo.logk
    dlogk = cosmo.dlogk
    res = length(logks)
    Cℓs = zeros(cosmo_type, length(ℓs))
    for i in 1:length(ℓs)
        ℓ = ℓs[i]
        integrand = Cℓintegrand(cosmo, t1, t2, logks, ℓ)/(ℓ+0.5)
        #for j in 1:length(logks)
        #    logk = logks[j]
        #    integrand[j] = Cℓintegrand(cosmo, t1, t2, logk, ℓ)/(ℓ+0.5)
        #end
        #integrand = [Cℓintegrand(cosmo, t1, t2, logk, ℓ)/(ℓ+0.5) for logk in logks]
        #Cℓ = sum(0.5 .* (integrand[1:res-1] .+ integrand[2:res]) .* dlogk)
        Cℓ = trapz(logks, integrand)
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
