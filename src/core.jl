# c/(100 km/s/Mpc) in Mpc
const CLIGHT_HMPC = 2997.92458

function w_tophat(x::Float64)
    x2 = x^2

    if x < 0.1
        w = 1. + x2*(-1.0/10.0 + x2*(1.0/280.0 +
            x2*(-1.0/15120.0 + x2*(1.0/1330560.0 +
            x2* (-1.0/172972800.0)))));
    else
        w = 3 * (sin(x)-x*cos(x))/(x2*x)
    end
    return w
end    

function _σR2(ks, pk, dlogk, R)
    x = ks .* R
    wk = w_tophat.(x)
    integ = @. pk * wk^2 * ks^3
    # OPT: proper integration instead?
    return sum(integ)*dlogk/(2*pi^2)
end

struct CosmoPar{T<:Real}
    Ωm::T
    Ωb::T
    h::T
    n_s::T
    σ8::T
end

struct Cosmology{T<:Real}
    cosmo::CosmoPar{T}
    # Power spectrum
    ks::Array{T, 1}
    pk0::Array{T, 1}
    dlogk::T
    lplk::AbstractInterpolation{T, 1}
    # Redshift and background
    zs::Array{T, 1}
    chis::Array{T, 1}
    chi::AbstractInterpolation{T, 1}
    z_of_chi::AbstractInterpolation{T, 1}
    chi_max::T
    Dzs::Array{T, 1}
    Dz::AbstractInterpolation{T, 1}
end

Cosmology(Ωc, Ωb, h, n_s, σ8; nk=256, nz=256) = begin
    Ωm = Ωc + Ωb
    cpar = CosmoPar(Ωm, Ωb, h, n_s, σ8)

    # Compute linear power spectrum at z=0.
    ks = 10 .^ range(-4., stop=2., length=nk)
    dlogk = log(ks[2]/ks[1])
    tk = TkBBKS(cpar, ks)
    pk0 = @. ks^n_s * tk
    σ8_2_here = _σR2(ks, pk0, dlogk, 8.0/h)
    norm = σ8^2 / σ8_2_here
    pk0[:] = pk0 .* norm
    # OPT: interpolation method
    pki = LinearInterpolation(log.(ks), log.(pk0))

    # Compute redshift-distance relation
    zs = range(0., stop=3., length=nz)
    norm = CLIGHT_HMPC / h
    chis = [quadgk(z -> 1.0/_Ez(cpar, z), 0.0, zz, rtol=1E-5)[1] * norm
            for zz in zs]
    # OPT: tolerances, interpolation method
    chii = LinearInterpolation(zs, chis)
    zi = LinearInterpolation(chis, zs)

    # ODE solution for growth factor
    z_ini = 1000.0
    a_ini = 1.0/(1.0+z_ini)
    ez_ini = _Ez(cpar, z_ini)
    d0 = [a_ini^3*ez_ini, a_ini]
    a_s = reverse(@. 1.0 / (1.0 + zs))
    prob = ODEProblem(_dgrowth!, d0, (a_ini, 1.0), cpar)
    sol = solve(prob, Tsit5(), reltol=1E-6,
                abstol=1E-8, saveat=a_s)
    # OPT: interpolation (see below), ODE method, tolerances
    # Note that sol already includes some kind of interpolation,
    # so it may be possible to optimize this by just using
    # sol directly.
    s = vcat(sol.u'...)
    Dzs = reverse(s[:, 2] / s[end, 2])
    # OPT: interpolation method
    Dzi = LinearInterpolation(zs, Dzs)

    Cosmology(cpar, ks, pk0, dlogk, pki,
              collect(zs), chis, chii, zi, chis[end],
              Dzs, Dzi)
end

Cosmology() = Cosmology(0.25, 0.05, 0.67, 0.96, 0.81)

function σR2(cosmo::Cosmology, R)
    return _σR2(cosmo.ks, cosmo.pk0, cosmo.dlogk, R)
end

function TkBBKS(cosmo::CosmoPar, k)
    tfac = 2.725 / 2.7
    q = @. (tfac^2 * k/(cosmo.Ωm * cosmo.h^2 * exp(-cosmo.Ωb*(1+sqrt(2*cosmo.h)/cosmo.Ωm))))
    return (@. (log(1+2.34q)/(2.34q))^2/sqrt(1+3.89q+(16.1q)^2+(5.46q)^3+(6.71q)^4))
end

function _Ez(cosmo::CosmoPar, z)
    @. sqrt(cosmo.Ωm*(1+z)^3+1-cosmo.Ωm)
end

function _dgrowth!(dd, d, cosmo::CosmoPar, a)
    ez = _Ez(cosmo, 1.0/a-1.0)
    dd[1] = d[2] * 1.5 * cosmo.Ωm / (a^2*ez)
    dd[2] = d[1] / (a^3*ez)
end

# Functions we will actually export
Ez(cosmo::Cosmology, z) = _Ez(cosmo.cosmo, z)
Hmpc(cosmo::Cosmology, z) = cosmo.cosmo.h*Ez(cosmo, z)/CLIGHT_HMPC
radial_comoving_distance(cosmo::Cosmology, z) = cosmo.chi(z)
growth_factor(cosmo::Cosmology, z) = cosmo.Dz(z)
function power_spectrum(cosmo::Cosmology, k, z)
    Dz2 = growth_factor(cosmo, z)^2
    @. exp(cosmo.lplk(log(k)))*Dz2
end
