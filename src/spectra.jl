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
    sett = cosmo.settings
    chis = zeros(sett.cosmo_type, sett.nk)
    chis[1:sett.nk] = (ℓ+0.5) ./ sett.ks
    chis .*= (chis .< cosmo.chi_max)
    z = cosmo.z_of_chi(chis)

    w1 = t1(chis)
    w2 = t2(chis)
    pk = nonlin_Pk(cosmo, sett.ks, z)

    return @. (sett.ks*w1*w2*pk)
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
    sett = cosmo.settings
    Cℓs = [integrate(sett.logk, Cℓintegrand(cosmo, t1.wint, t2.wint, ℓ)/(ℓ+0.5), SimpsonEven()) for ℓ in ℓs]
    return t1.F(ℓs) .* t2.F(ℓs) .* Cℓs
end

function angularCℓsFast(cosmo::Cosmology, W::Matrix{Any}, F::Matrix{Any})
    sett = cosmo.settings

    P = zeros(Float64, sett.nz_t, sett.nℓ)
    for z ∈ axes(sett.zs_t, 1)
        for ℓ ∈  axes(sett.ℓs, 1)
            P[z, ℓ] = nonlin_Pk(cosmo, (sett.ℓs[ℓ]+0.5)/chis[z], sett.zs_t[z])
        end
    end
    Ezs = Ez(cosmo, sett.zs_t)
    chis = cosmo.chi(sett.zs_t)
    dz = (sett.zs_t[2] - sett.zs_t[1])
    SimpsonWeights = SimpsonWeightArray(sett.nz_t)
    C_ℓij = zeros(sett.cosmo_type, sett.nℓ, ntracers, ntracers)
    @turbo for ℓ ∈ axes(C_ℓij, 1)
        for i ∈ axes(C_ℓij, 2)
            for j ∈ axes(C_ℓij, 3)
                for z ∈ axes(sett.zs_t, 1)
                    integrand = (W[i, z] * W[j, z] * P[z, ℓ]) / (Ezs[z] * chis[z]^2)
                    C_ℓij[ℓ,i,j] += integrand * SimpsonWeights[z] * dz
                end
                C_ℓij[ℓ,i,j] *= (CLIGHT_HMPC/cosmo.cpar.h) * F[i, ℓ] * F[j, ℓ]
            end
        end
    end
    return C_ℓij
end