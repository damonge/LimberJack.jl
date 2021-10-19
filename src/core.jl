# c/(100 km/s/Mpc) in Mpc
const CLIGHT_HMPC = 2997.92458

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
    θCMB::T
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

Cosmology(cpar::CosmoPar; nk=256, nz=256, tk_mode="BBKS") = begin
    # Compute linear power spectrum at z=0.
    ks = 10 .^ range(-4., stop=2., length=nk)
    dlogk = log(ks[2]/ks[1])
    if tk_mode== "EisHu"
        tk = TkEisHu(cpar, ks)
    elseif tk_mode== "BBKS"
        tk = TkBBKS(cpar, ks)
    else
        print("Transfer function not implemented")
    end
    pk0 = @. ks^cpar.n_s * tk
    σ8_2_here = _σR2(ks, pk0, dlogk, 8.0/cpar.h)
    norm = cpar.σ8^2 / σ8_2_here
    pk0[:] = pk0 .* norm
    # OPT: interpolation method
    pki = LinearInterpolation(log.(ks), log.(pk0))

    # Compute redshift-distance relation
    zs = range(0., stop=3., length=nz)
    norm = CLIGHT_HMPC / cpar.h
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

Cosmology(Ωc, Ωb, h, n_s, σ8; θCMB=2.728/2.7, nk=256, nz=256) = begin
    Ωm = Ωc + Ωb
    cpar = CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB)

    Cosmology(cpar, nk=nk, nz=nz)
end

Cosmology() = Cosmology(0.25, 0.05, 0.67, 0.96, 0.81)

function σR2(cosmo::Cosmology, R)
    return _σR2(cosmo.ks, cosmo.pk0, cosmo.dlogk, R)
end

function TkBBKS(cosmo::CosmoPar, k)
    q = @. (cosmo.θCMB^2 * k/(cosmo.Ωm * cosmo.h^2 * exp(-cosmo.Ωb*(1+sqrt(2*cosmo.h)/cosmo.Ωm))))
    return (@. (log(1+2.34q)/(2.34q))^2/sqrt(1+3.89q+(16.1q)^2+(5.46q)^3+(6.71q)^4))
end

function get_zeq(cosmo::CosmoPar)
    wm=cosmo.Ωm*cosmo.h^2
    return (2.5*10^4)*wm*(cosmo.θCMB^-4)
end

function get_keq(cosmo::CosmoPar)
    wm=cosmo.Ωm*cosmo.h^2
    return (7.46*10^-2)*wm/(cosmo.h*cosmo.θCMB^2) #
end

function get_zdrag(cosmo::CosmoPar)
    wb=cosmo.Ωb*cosmo.h^2
    wm=cosmo.Ωm*cosmo.h^2
    b1 = 0.313*(wm^-0.419)*(1+0.607*wm*0.674)
    b2 = 0.238*wm^0.223
    return 1291*((wm^0.251)/(1+0.659*wm^0.828))*(1+b1*wb^b2)
end

function R(cosmo::CosmoPar, z)
    wb=cosmo.Ωb*cosmo.h^2
    R = 31.5*wb*(cosmo.θCMB^-4)*(z/10^3)^-1
    return R
end

function get_rs(cosmo::CosmoPar)
    keq = get_keq(cosmo)
    zeq = get_zeq(cosmo)
    zd = get_zdrag(cosmo)
    rs = (sqrt(1+R(cosmo, zd+1))+sqrt(R(cosmo, zd+1)+R(cosmo, zeq)))
    rs /= (1+sqrt(R(cosmo, zeq)))
    rs =  (log(rs))
    rs *= (2/(3*keq))*sqrt(6/R(cosmo, zeq))
    return rs
end
    
function G(y)
    return @. (y*(-6*sqrt(1+y)+(2+3y)*log((sqrt(1+y)+1)/(sqrt(1+y)-1))))
end

function T0(cosmo::CosmoPar, k, ac, bc)
    keq = get_keq(cosmo)
    q = @. (k/(13.41*keq))
    C = @. ((14.2/ac) + (386/(1+69.9*q^1.08)))
    T0 = @.(log(ℯ+1.8*bc*q)/(log(ℯ+1.8*bc*q)+C*q^2))
    return T0
end 

function Tb(cosmo::CosmoPar, k)
   wm=cosmo.Ωm*cosmo.h^2
   wb=cosmo.Ωb*cosmo.h^2
   s = get_rs(cosmo)
   zd = get_zdrag(cosmo)
   zeq = get_zeq(cosmo)
   keq = get_keq(cosmo)
   ksilk = 1.6*wb^0.52*wm^0.73*(1+(10.4*wm)^-0.95)
   ab = 2.07*keq*s*(1+R(cosmo, zd))^(-3/4)*G((1+zeq)/(1+zd))
   bb =  0.5+(cosmo.Ωb/cosmo.Ωm)+(3-2*cosmo.Ωb/cosmo.Ωm)*sqrt((17.2*wm)^2+1)
   bnode = 8.41*(wm)^0.435
   ss = s./(1 .+(bnode./(k.*s)).^3).^(1/3)
   Tb1 = T0(cosmo, k, 1, 1)./(1 .+(k.*s/5.2).^2)
   Tb2 = (ab./(1 .+(bb./(k.*s)).^3)).*exp.(-(k/ksilk).^ 1.4)
   Tb = (Tb1.+Tb2).*sin.(k.*ss)./(k.*ss)
   return Tb
end 

function Tc(cosmo::CosmoPar, k)
   Wc = cosmo.Ωm-cosmo.Ωb
   s = get_rs(cosmo)
   keq = get_keq(cosmo)
   wm=cosmo.Ωm*cosmo.h^2
   q = @.(k/(13.41*keq))
   a1 = (46.9*wm)^0.670*(1+(32.1*wm)^-0.532)
   a2 = (12.0*wm)^0.424*(1+(45.0*wm)^-0.582)
   ac = (a1^(-cosmo.Ωb/cosmo.Ωm))*(a2^(-(cosmo.Ωb/cosmo.Ωm)^3))
   b1 = 0.944*(1+(458*wm)^-0.708)^-1
   b2 = (0.395*wm)^(-0.0266)
   bc = (1+b1*((Wc/cosmo.Ωm)^b2-1))^-1
   f = @.(1/(1+(k*s/5.4)^4))
   Tc1 = f.*T0(cosmo, k, 1, bc)
   Tc2 = (1 .-f).*T0(cosmo, k, ac, bc)
   Tc = Tc1 .+ Tc2
   return Tc
end 

function TkEisHu(cosmo::CosmoPar, k)
    Wc = cosmo.Ωm-cosmo.Ωb
    Tk = (cosmo.Ωb/cosmo.Ωm).*Tb(cosmo, k).+(Wc/cosmo.Ωm).*Tc(cosmo, k)
    return Tk.^2 #apparently we have to put here a square
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
