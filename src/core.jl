# c/(100 km/s/Mpc) in Mpc
const CLIGHT_HMPC = 2997.92458
# c/(km/s/Mpc) in Mpc
const CLIGHT_MPC = 299792.458

function w_tophat(x::Real)
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

struct CosmoPar{T}
    Ωm::T
    Ωb::T
    h::T
    n_s::T
    σ8::T
end

struct Cosmology
    cosmo::CosmoPar
    # Power spectrum
    ks::Array
    pk0::Array
    dlogk
    lplk::AbstractInterpolation
    # Redshift and background
    zs::Array
    chis::Array
    chi::AbstractInterpolation
    z_of_chi::AbstractInterpolation
    chi_max
    Dzs::Array
    Dz::AbstractInterpolation
end

Cosmology(cpar::CosmoPar; nk=256, nz=256) = begin
    # Compute linear power spectrum at z=0.
    ks = 10 .^ range(-4., stop=2., length=nk)
    dlogk = log(ks[2]/ks[1])
    tk = TkBBKS(cpar, ks)
    pk0 = @. ks^cpar.n_s * tk
    σ8_2_here = _σR2(ks, pk0, dlogk, 8.0/cpar.h)
    norm = cpar.σ8^2 / σ8_2_here
    pk0[:] = pk0 .* norm
    # OPT: interpolation method
    pki = LinearInterpolation(log.(ks), log.(pk0), extrapolation_bc=Linear())

    # Compute redshift-distance relation
    zs = range(0., stop=3., length=nz)
    norm = CLIGHT_HMPC / cpar.h
    chis = [quadgk(z -> 1.0/_Ez(cpar, z), 0.0, zz, rtol=1E-5)[1] * norm
            for zz in zs]
    # OPT: tolerances, interpolation method
    chii = LinearInterpolation(zs, chis, extrapolation_bc=Linear())
    zi = LinearInterpolation(chis, zs, extrapolation_bc=Linear())

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
    Dzi = LinearInterpolation(zs, Dzs, extrapolation_bc=Linear())

    Cosmology(cpar, ks, pk0, dlogk, pki,
              collect(zs), chis, chii, zi, chis[end],
              Dzs, Dzi)
end

Cosmology(Ωc, Ωb, h, n_s, σ8; nk=256, nz=256) = begin
    Ωm = Ωc + Ωb
    cpar = CosmoPar(Ωm, Ωb, h, n_s, σ8)

    Cosmology(cpar, nk=nk, nz=nz)
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
comoving_radial_distance(cosmo::Cosmology, z) = cosmo.chi(z)
growth_factor(cosmo::Cosmology, z) = cosmo.Dz(z)

function power_spectrum(cosmo::Cosmology, k, z)
    Dz2 = growth_factor(cosmo, z)^2
    @. exp(cosmo.lplk(log(k)))*Dz2
end

function delta(x, x0 = 0)
    if x == x0
        delta = 1
    else
        delta = 0
    end
    return delta
end

function get_pz(dpdz)
    z = dpdz[1]
    p = dpdz[2]
    pz = LinearInterpolation(z, p, extrapolation_bc=0)
    norm = quadgk(pz, minimum(z), maximum(z), rtol=1E-5)[1] 
    ppz(zz) = pz(zz)./norm
    return ppz
end

function lensing_kernel(cosmo::Cosmology, z, dpdz)
    X = cosmo.chi(z)
    a = 1/(1+z)
    XX(zz) = cosmo.chi(zz)
    H = 100*cosmo.cosmo.h/CLIGHT_MPC
    Wm = cosmo.cosmo.Ωm
    pz = get_pz(dpdz)
    qL(zz) = pz(zz)*(XX(zz)-X)/(XX(zz)) 
    QL = quadgk(qL, z, Inf)[1]
    QL *= (3/2)*H^2*Wm*(X/a)
    return QL
end

function shear_kernel(cosmo::Cosmology, dpdz)
    Gl(l) = @.(sqrt((l+2.)*(l+1.)*l*(l-1.))/(l+0.5)^2)
    qL(z) = lensing_kernel(cosmo, z, dpdz)
    qgamma(z, l) = Gl(l).*qL(z)
    return qgamma
end 

function convergence_kernel(cosmology::Cosmology, dpdz)
    Kl(l) = l.*(l.+1)./(l.+1/2).^2
    qL(z) = lensing_kernel(cosmo, z, dpdz)
    qk(z, l) = Kl(l).*qL(z)
    return qk
end 

function CMBlensing_kernel(cosmo::Cosmology, z, zs)
    X = cosmo.chi(z)
    Xs = cosmo.chi(zs)
    a = 1/(1+z)
    XX(zz) = cosmo.chi(zz)
    H = 100*cosmo.cosmo.h/CLIGHT_MPC
    Wm = cosmo.cosmo.Ωm
    if z < zs
        qCMBL = (3/2)*H^2*Wm*(X/a)*(Xs-X)/(Xs) 
    else
        qCMBL = 0 
    end
    return qCMBL
end

function CMBk_kernel(cosmology::Cosmology, zs)
    Kl(l) = l.*(l.+1)./(l.+1/2).^2
    qCMBL(z) = CMBlensing_kernel(cosmo, z, zs)
    qCMBk(z, l) = Kl(l).*qCMBL(z)
    return qCMBk
end 

function clustering_kernel(cosmology::Cosmology, bg, dpdz) 
    pz = get_pz(dpdz)
    qg(z, l) = ones(length(l)).*bg.*pz(z).*Hmpc(cosmology, z)
    return qg
end
    
function Cl(cosmology::Cosmology, l, tracer_u, tracer_v)
    P_uv(k, z) = power_spectrum(cosmo, k, z)
    XX(z) = cosmo.chi(z)
    cl(z) = (1/Hmpc(cosmology, z)).*(tracer_u(z, l).*tracer_v(z, l)).*P_uv((l.+1/2)./XX(z), z)./XX(z)^2
    Cl = quadgk(cl, 0, Inf)[1]
    return Cl
end

# Make the Cl integral in K
# Compare to David's tracers.jl Number_counts

struct NumberCountsTracer{T<:Real}
    wint::AbstractInterpolation{T, 1}
    bias::T
    wnorm::T
end

NumberCountsTracer(z, wz, bias) = begin
    wint = LinearInterpolation(z, wz, extrapolation_bc=0)
    area = quadgk(wint, z[1], z[end], rtol=1E-5)[1]
    NumberCountsTracer(wint, bias, 1.0/area)
end

function Cℓintegrand(cosmo::Cosmology,
                     t1::NumberCountsTracer,
                     t2::NumberCountsTracer,
                     logk::Float64, ℓ::Int64)
    k = exp(logk)
    chi = (ℓ+0.5)/k
    if chi > cosmo.chi_max
        return 0
    end
    z = cosmo.z_of_chi(chi)
    hz = Hmpc(cosmo, z)
    w1 = t1.wint(z)*t1.wnorm*hz*t1.bias
    w2 = t2.wint(z)*t2.wnorm*hz*t1.bias
    pk = power_spectrum(cosmo, k, z)
    k*w1*w2*pk
end

function angularCℓ(cosmo, t1, t2, ℓ)
    quadgk(lk -> Cℓintegrand(cosmo, t1, t2, lk, ℓ),
           log(10^-4), log(10^2), rtol=1E-5)[1]/(ℓ+0.5)
end

        