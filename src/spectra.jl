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
                     t1::AbstractInterpolation,
                     t2::AbstractInterpolation,
                     ℓ::Number)

    chis = zeros(cosmo.settings.cosmo_type, cosmo.settings.nk)
    chis[1:cosmo.settings.nk] = (ℓ+0.5) ./ cosmo.ks
    chis .*= (chis .< cosmo.chi_max)
    z = cosmo.z_of_chi(chis)

    w1 = t1(chis)
    w2 = t2(chis)
    pk = nonlin_Pk(cosmo, cosmo.ks, z)

    return @. (cosmo.ks*w1*w2*pk)
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
function angularCℓs(cosmo::Cosmology, t1::Tracer, t2::Tracer, ℓs::Vector)
    # OPT: we are not optimizing the limits of integration
    Cℓs = [integrate(cosmo.logk, Cℓintegrand(cosmo, t1.wint, t2.wint, ℓ)/(ℓ+0.5), SimpsonEven()) for ℓ in ℓs]
    return t1.F(ℓs) .* t2.F(ℓs) .* Cℓs
end

function _get_Fℓ(t::Tracer, ℓ::Vector{Float64})
    if typeof(t) == WeakLensingTracer
        return @. sqrt((ℓ+2)*(ℓ+1)*ℓ*(ℓ-1))/(ℓ+0.5)^2
    elseif typeof(t) == CMBLensingTracer
        return @. (ℓ+1)*ℓ/(ℓ+0.5)^2
    else
        return 1
    end
end