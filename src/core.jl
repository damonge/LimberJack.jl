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
    CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB, Ωr, ΩΛ)
end

struct Cosmology
    cosmo::CosmoPar
    # Power spectrum
    ks::Array
    pk0::Array
    dlogk
    # Redshift and background
    zs::Array
    chis::Array
    chi::AbstractInterpolation
    z_of_chi::AbstractInterpolation
    chi_max
    chi_LSS
    Dzs::Array
    Dz::AbstractInterpolation
    primordial_lPk::AbstractInterpolation
    lin_Pk::AbstractInterpolation
    Pk::AbstractInterpolation
end

Cosmology(cpar::CosmoPar; nk=256, nz=256, kmin=-4, kmax=2, zmax=3, tk_mode="BBKS") = begin
    # Compute linear power spectrum at z=0.
    ks = 10 .^ range(-4., stop=2., length=nk)
    dlogk = log(ks[2]/ks[1])
    # Compute redshift-distance relation
    zs = range(0., stop=zmax, length=nz)
    norm = CLIGHT_HMPC / cpar.h
    chis = [quadgk(z -> 1.0/_Ez(cpar, z), 0.0, zz, rtol=1E-5)[1] * norm
            for zz in zs]
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

    pk0, primordial_lPk = _primordial_lPk(cpar, ks, dlogk)
    lin_Pk = _lin_Pk(zs, ks, primordial_lPk, Dzi)
    Pk = _Pk(cpar, zs, ks, lin_Pk)
    
    Cosmology(cpar, ks, pk0, dlogk,
              collect(zs), chis, chii, zi, chis[end],
              chi_LSS, Dzs, Dzi, primordial_lPk,
              lin_Pk, Pk)
end

Cosmology(Ωc, Ωb, h, n_s, σ8; θCMB=2.725/2.7, nk=256, nz=256, tk_mode="BBKS") = begin
    Ωm = Ωc + Ωb
    cpar = CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB)

    Cosmology(cpar, nk=nk, nz=nz, tk_mode=tk_mode)
end

Cosmology() = Cosmology(0.25, 0.05, 0.67, 0.96, 0.81)

function _primordial_lPk(cpar::CosmoPar, ks, dlogk; tk_mode="EisHu")
    if tk_mode== "EisHu"
        tk = TkEisHu(cpar, ks./ cpar.h)
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
    pki = LinearInterpolation(log.(ks), log.(pk0), extrapolation_bc=Line())
    return pk0, pki
end

function _lin_Pk(zs, ks, primordial_lPk::AbstractInterpolation, Dz::AbstractInterpolation)
    # output: 2D matrix [z1 = [k1, k2, ...]
    #                   z2 = [k1, k2, ...]
    #                   ...]
    nz = length(zs)
    nk = length(ks)
    PkLs = zeros(nz, nk)
    for i in range(1, stop=nz)
        z_i = zs[i]
        Dz2 = Dz(z_i).^2
        PkLs[i, :] .= @. exp(primordial_lPk(log(ks)))*Dz2
    end
    PkL = LinearInterpolation((zs, ks), PkLs)
    return PkL
end

function _Pk(cpar::CosmoPar, zs, ks, PkL::AbstractInterpolation)
    # output: 2D matrix [z1 = [k1, k2, ...]
    #                   z2 = [k1, k2, ...]
    #                   ...]
    halofit = PKnonlin(cpar, zs, ks, PkL)
    Pk = halofit.pk_NL
    return Pk
end

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

function _omega_x(cosmo::CosmoPar, z, species_x_label)
    Ez = _Ez(cosmo, z)
    a = @. 1.0/(1.0 + z)

    if species_x_label == "crit"
        return 1.0
    elseif species_x_label == "m"
        return @. cosmo.Ωm / (a^3) / Ez^2
    elseif species_x_label == "l"
        return 1.0 - cosmo.Ωm
    else
       println("Only species_x_label = crit, m, l supported so far.")
    end
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
Ez(cosmo::Cosmology, z) = _Ez(cosmo.cosmo, z)
Hmpc(cosmo::Cosmology, z) = cosmo.cosmo.h*Ez(cosmo, z)/CLIGHT_HMPC
comoving_radial_distance(cosmo::Cosmology, z) = cosmo.chi(z)
growth_factor(cosmo::Cosmology, z) = cosmo.Dz(z)
omega_x(cosmo::Cosmology, z, species_x_label) = _omega_x(cosmo.cosmo, z, species_x_label)
primordial_Pk(cosmo::Cosmology, z, k) = exp.(cosmo.primordial_lPk(log.(k)))
lin_Pk(cosmo::Cosmology, z, k) = cosmo.lin_Pk(z, k)
Pk(cosmo::Cosmology, z, k) = cosmo.Pk(z, k)
