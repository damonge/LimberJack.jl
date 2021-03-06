# c/(100 km/s/Mpc) in Mpc
const CLIGHT_HMPC = 2997.92458

macro Name(arg)
   string(arg)
end

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

mutable struct Settings
    cosmo_type::DataType
    nz::Int
    nz_pk::Int
    nk::Int
    tk_mode::String
    Pk_mode::String
end

struct CosmoPar{T}
    Ωm::T
    Ωb::T
    h::T
    n_s::T
    σ8::T
    θCMB::T
    Ωr::T
    ΩΛ::T
end

CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB) = begin
    # This is 4*sigma_SB*(2.7 K)^4/rho_crit(h=1)
    prefac = 2.38163816E-5
    Neff = 3.046
    f_rel = 1.0 + Neff * (7.0/8.0) * (4.0/11.0)^(4.0/3.0)
    Ωr = prefac*f_rel*θCMB^4/h^2
    ΩΛ = 1-Ωm-Ωr
    CosmoPar{Real}(Ωm, Ωb, h, n_s, σ8, θCMB, Ωr, ΩΛ)
end

struct Cosmology
    settings::Settings
    cosmo::CosmoPar
    # Power spectrum
    ks::Array
    pk0::Array
    logk
    dlogk
    # Redshift and background
    zs::Array
    chi::AbstractInterpolation
    z_of_chi::AbstractInterpolation
    chi_max
    chi_LSS
    Dz::AbstractInterpolation
    PkLz0::AbstractInterpolation
    Pk::AbstractInterpolation
end

Cosmology(cpar::CosmoPar, settings::Settings) = begin
    # Load settings
    cosmo_type = settings.cosmo_type
    nk = settings.nk
    nz_pk = settings.nz_pk
    nz = settings.nz
    # Compute linear power spectrum at z=0.
    logk = range(log(0.0001), stop=log(100.0), length=nk)
    ks = exp.(logk)
    dlogk = log(ks[2]/ks[1])
    if settings.tk_mode == "emulator"
        ks_emul, pk0_emul = get_emulated_log_pk0(cpar)
        pki_emul = LinearInterpolation(log.(ks_emul), log.(pk0_emul),
                                       extrapolation_bc=Line())
        pk0 = exp.(pki_emul(logk))
    elseif settings.tk_mode == "EisHu"
        tk = TkEisHu(cpar, ks./ cpar.h)
        pk0 = @. ks^cpar.n_s * tk
    elseif settings.tk_mode == "BBKS"
        tk = TkBBKS(cpar, ks)
        pk0 = @. ks^cpar.n_s * tk
     else
        print("Transfer function not implemented")
    end
    #Renormalize Pk
    σ8_2_here = _σR2(ks, pk0, dlogk, 8.0/cpar.h)
    norm = cpar.σ8^2 / σ8_2_here
    pk0 *= norm
    # OPT: interpolation method
    pki = LinearInterpolation(logk, log.(pk0), extrapolation_bc=Line())
    # Compute redshift-distance relation
    zs = range(0., stop=3., length=nz)
    norm = CLIGHT_HMPC / cpar.h
    chis = zeros(cosmo_type, nz)
    for i in 1:nz
        zz = zs[i]
        chis[i] = quadgk(z -> 1.0/_Ez(cpar, z), 0.0, zz, rtol=1E-5)[1] * norm
    end
    # OPT: tolerances, interpolation method
    chii = LinearInterpolation(zs, chis, extrapolation_bc=Line())
    zi = LinearInterpolation(chis, zs, extrapolation_bc=Line())
    # Distance to LSS
    chi_LSS = quadgk(z -> 1.0/_Ez(cpar, z), 0.0, 1100., rtol=1E-5)[1] * norm


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
    Dzi = LinearInterpolation(zs, Dzs, extrapolation_bc=Line())

    # OPT: separate zs for Pk and background
    zs_pk = range(0., stop=3., length=nz_pk)
    Dzs = Dzi(zs_pk)

    if settings.Pk_mode == "linear"
        Pks = [@. pk*Dzs^2 for pk in pk0]
        Pks = reduce(vcat, transpose.(Pks))
        Pk = LinearInterpolation((logk, zs_pk), log.(Pks))
    elseif settings.Pk_mode == "Halofit"
        Pk = get_PKnonlin(cpar, zs_pk, ks, pk0, Dzs, cosmo_type)
    else 
        Pks = [@. pk*Dzs^2 for pk in pk0]
        Pks = reduce(vcat, transpose.(Pks))
        Pk = LinearInterpolation((logk, zs_pk), log.(Pks))
        print("Pk mode not implemented. Using linear Pk.")
    end
    Cosmology(settings, cpar, ks, pk0, logk, dlogk,
              collect(zs), chii, zi, chis[end],
              chi_LSS, Dzi, pki, Pk)
end

Cosmology(Ωm, Ωb, h, n_s, σ8; θCMB=2.725/2.7, nk=500,
          nz=500, nz_pk=100, tk_mode="BBKS", Pk_mode="linear") = begin
    cosmo_type = eltype([Ωm, Ωb, h, n_s, σ8, θCMB])
    cpar = CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB)
    settings = Settings(cosmo_type, nz, nz_pk, nk, tk_mode, Pk_mode)
    Cosmology(cpar, settings)
end

Cosmology() = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81)

function σR2(cosmo::Cosmology, R)
    return _σR2(cosmo.ks, cosmo.pk0, cosmo.dlogk, R)
end

function TkBBKS(cosmo::CosmoPar, k)
    q = @. (cosmo.θCMB^2 * k/(cosmo.Ωm * cosmo.h^2 * exp(-cosmo.Ωb*(1+sqrt(2*cosmo.h)/cosmo.Ωm))))
    return (@. (log(1+2.34q)/(2.34q))^2/sqrt(1+3.89q+(16.1q)^2+(5.46q)^3+(6.71q)^4))
end

function _T0(keq, k, ac, bc)
    q = @. (k/(13.41*keq))
    C = @. ((14.2/ac) + (386/(1+69.9*q^1.08)))
    T0 = @.(log(ℯ+1.8*bc*q)/(log(ℯ+1.8*bc*q)+C*q^2))
    return T0
end 

function TkEisHu(cosmo::CosmoPar, k)
    Ωc = cosmo.Ωm-cosmo.Ωb
    wm=cosmo.Ωm*cosmo.h^2
    wb=cosmo.Ωb*cosmo.h^2

    # Scales
    # k_eq
    keq = (7.46*10^-2)*wm/(cosmo.h * cosmo.θCMB^2)
    # z_eq
    zeq = (2.5*10^4)*wm*(cosmo.θCMB^-4)
    # z_drag
    b1 = 0.313*(wm^-0.419)*(1+0.607*wm^0.674)
    b2 = 0.238*wm^0.223
    zd = 1291*((wm^0.251)/(1+0.659*wm^0.828))*(1+b1*wb^b2)
    # k_Silk
    ksilk = 1.6*wb^0.52*wm^0.73*(1+(10.4*wm)^-0.95)/cosmo.h
    # r_s
    R_prefac = 31.5*wb*(cosmo.θCMB^-4)
    Rd = R_prefac*((zd+1)/10^3)^-1
    Rdm1 = R_prefac*(zd/10^3)^-1
    Req = R_prefac*(zeq/10^3)^-1
    rs = sqrt(1+Rd)+sqrt(Rd+Req)
    rs /= (1+sqrt(Req))
    rs =  (log(rs))
    rs *= (2/(3*keq))*sqrt(6/Req)

    # Tc
    q = @.(k/(13.41*keq))
    a1 = (46.9*wm)^0.670*(1+(32.1*wm)^-0.532)
    a2 = (12.0*wm)^0.424*(1+(45.0*wm)^-0.582)
    ac = (a1^(-cosmo.Ωb/cosmo.Ωm))*(a2^(-(cosmo.Ωb/cosmo.Ωm)^3))
    b1 = 0.944*(1+(458*wm)^-0.708)^-1
    b2 = (0.395*wm)^(-0.0266)
    bc = (1+b1*((Ωc/cosmo.Ωm)^b2-1))^-1
    f = @.(1/(1+(k*rs/5.4)^4))
    Tc1 = f.*_T0(keq, k, 1, bc)
    Tc2 = (1 .-f).*_T0(keq, k, ac, bc)
    Tc = Tc1 .+ Tc2

    # Tb
    y = (1+zeq)/(1+zd)
    Gy = y*(-6*sqrt(1+y)+(2+3y)*log((sqrt(1+y)+1)/(sqrt(1+y)-1)))
    ab = 2.07*keq*rs*(1+Rdm1)^(-3/4)*Gy
    bb =  0.5+(cosmo.Ωb/cosmo.Ωm)+(3-2*cosmo.Ωb/cosmo.Ωm)*sqrt((17.2*wm)^2+1)
    bnode = 8.41*(wm)^0.435
    ss = rs./(1 .+(bnode./(k.*rs)).^3).^(1/3)
    Tb1 = _T0(keq, k, 1, 1)./(1 .+(k.*rs/5.2).^2)
    Tb2 = (ab./(1 .+(bb./(k.*rs)).^3)).*exp.(-(k/ksilk).^ 1.4)
    Tb = (Tb1.+Tb2).*sin.(k.*ss)./(k.*ss)

    Tk = (cosmo.Ωb/cosmo.Ωm).*Tb.+(Ωc/cosmo.Ωm).*Tc
    return Tk.^2
end

function _Ez(cosmo::CosmoPar, z)
    @. sqrt(cosmo.Ωm*(1+z)^3+cosmo.Ωr*(1+z)^4+cosmo.ΩΛ)
end

function _dgrowth!(dd, d, cosmo::CosmoPar, a)
    ez = _Ez(cosmo, 1.0/a-1.0)
    dd[1] = d[2] * 1.5 * cosmo.Ωm / (a^2*ez)
    dd[2] = d[1] / (a^3*ez)
end

# Functions we will actually export
function chi_to_z(cosmo::Cosmology, chi)
    closest_chi, idx = findmin(abs(chi-cosmo.chis))
    if closest_chi >= chi
        idx += -1
    end

    return z
end
Ez(cosmo::Cosmology, z) = _Ez(cosmo.cosmo, z)
Hmpc(cosmo::Cosmology, z) = cosmo.cosmo.h*Ez(cosmo, z)/CLIGHT_HMPC
comoving_radial_distance(cosmo::Cosmology, z) = cosmo.chi(z)
growth_factor(cosmo::Cosmology, z) = cosmo.Dz(z)

function nonlin_Pk(cosmo::Cosmology, k, z)
    return @. exp(cosmo.Pk(log(k), z))
end

function lin_Pk(cosmo::Cosmology, k, z)
    pk0 = @. exp(cosmo.PkLz0(log(k)))
    Dz2 = cosmo.Dz(z)^2
    return pk0 .* Dz2
end
