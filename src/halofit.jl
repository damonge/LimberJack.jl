#=
halofit:
- Julia version: 
- Author: andrina
- Date: 2021-09-17
- Modified by: damonge and JaimeRZP
- Date: 2022-02-19
=#

function get_σ2(lks, ks, pks, R, kind)
    k3 = ks .^ 3
    x2 = @. (ks*R)^2
    if kind == 2
        pre = x2
    elseif kind == 4
        pre = @. x2*(1-x2)
    else
        pre = 1
    end
    integrand = @. k3 * pre * exp(-x2) * pks
    return trapz(lks, integrand)/(2*pi^2)
end

function get_PKnonlin(cosmo::CosmoPar, z, k, PkLz0, Dzs)
    nk = length(k)
    nz = length(z)
    logk = log.(k)

    Dz2s = Dzs .^ 2
    # OPT: hard-coded range and number of points
    lR0 = log(0.01)
    lR1 = log(10.0)
    lRs = range(lR0, stop=lR1, length=100)
    lσ2s = log.([get_σ2(logk, k, PkLz0, exp(lR), 0) for lR in lRs])
    # When s8<0.6 lσ2i interpolation fails --> Extrapolation needed
    lσ2i = CubicSplineInterpolation(lRs, lσ2s) #, extrapolation_bc=Line())
    rsigs = [get_rsigma(lσ2i, Dz2s[i], lR0, lR1) for i in range(1, stop=nz)]

    onederiv_ints = [get_σ2(logk, k, PkLz0, rsigs[i], 2) * Dz2s[i] for i in range(1, stop=nz)]
    twoderiv_ints = [get_σ2(logk, k, PkLz0, rsigs[i], 4) * Dz2s[i] for i in range(1, stop=nz)]
    neffs = @. 2*onederiv_ints - 3.0
    Cs = @. 4*(twoderiv_ints + onederiv_ints^2)

    # Interpolate linearily over a
    pk_NLs = [power_spectrum_nonlin(cosmo, PkLz0 .* Dz2s[i], k, z[i], rsigs[i], neffs[i], Cs[i])
              for i in range(1, stop=nz)]
    pk_NLs = transpose(reduce(vcat,transpose.(pk_NLs)))
    # TODO: I don't know why this is 2x slower than the above
    #pk_NLs = zeros(Real, nk, nz)
    #for i in 1:nz
    #    pk_NLs[:, i] = power_spectrum_nonlin(cosmo, PkLz0 .* Dz2s[i],
    #                                         k, z[i], rsigs[i], neffs[i], Cs[i])
    #end
    pk_NL = LinearInterpolation((logk, z), log.(pk_NLs))

    return pk_NL
end 

function secant(f, x0, x1, tol)
    # TODO: fail modes
    x_nm1 = x0
    x_nm2 = x1
    f_nm1 = f(x_nm1)
    f_nm2 = f(x_nm2)
    while abs(x_nm1 - x_nm2) > tol
        x_n = (x_nm2*f_nm1-x_nm1*f_nm2)/(f_nm1-f_nm2)
        x_nm2 = x_nm1
        x_nm1 = x_n
        f_nm2 = f_nm1
        f_nm1 = f(x_nm1)
    end
    return 0.5*(x_nm2+x_nm1)
end

function get_rsigma(lσ2i, Dz2, lRmin, lRmax)
    lDz2 = log(Dz2)
    lRsigma = secant(lR -> lσ2i(lR)+lDz2, lRmin, lRmax, 1E-4)
    return exp(lRsigma)
end

function power_spectrum_nonlin(cpar::CosmoPar, PkL, k, z, rsig, neff, C)
    # DAM: note that below I've commented out anything to do with
    # neutrinos or non-Lambda dark energy.
    opz = 1.0+z
    #weffa = -1.0
    Ez2 = cpar.Ωm*opz^3+cpar.Ωr*opz^4+cpar.ΩΛ
    omegaMz = cpar.Ωm*opz^3 / Ez2
    omegaDEwz = cpar.ΩΛ/Ez2

    ksigma = @. 1.0 / rsig
    neff2 = @. neff * neff
    neff3 = @. neff2 * neff
    neff4 = @. neff3 * neff

    delta2_norm = @. k*k*k/2.0/pi/pi

    # compute the present day neutrino massive neutrino fraction
    # uses all neutrinos even if they are moving fast
    # fnu = 0.0

    # eqns A6 - A13 of Takahashi et al.
    an = @. 10.0^(1.5222 + 2.8553*neff + 2.3706*neff2 + 0.9903*neff3 +
        0.2250*neff4 - 0.6038*C)# + 0.1749*omegaDEwz*(1.0 + weffa))
    bn = @. 10.0^(-0.5642 + 0.5864*neff + 0.5716*neff2 - 1.5474*C)# + 0.2279*omegaDEwz*(1.0 + weffa))
    cn = @. 10.0^(0.3698 + 2.0404*neff + 0.8161*neff2 + 0.5869*C)
    gamman = @. 0.1971 - 0.0843*neff + 0.8460*C
    alphan = @. abs(6.0835 + 1.3373*neff - 0.1959*neff2 - 5.5274*C)
    betan = @. 2.0379 - 0.7354*neff + 0.3157*neff2 + 1.2490*neff3 + 0.3980*neff4 - 0.1682*C
    #mun = 0.0
    nun = @. 10.0^(5.2105 + 3.6902*neff)

    # eqns C17 and C18 for Smith et al.
    if abs(1.0 - omegaMz) > 0.01
        f1a = omegaMz ^ -0.0732
        f2a = omegaMz ^ -0.1423
        f3a = omegaMz ^ 0.0725
        f1b = omegaMz ^ -0.0307
        f2b = omegaMz ^ -0.0585
        f3b = omegaMz ^ 0.0743
        fb_frac = omegaDEwz / (1.0 - omegaMz)
        f1 = fb_frac * f1b + (1.0 - fb_frac) * f1a
        f2 = fb_frac * f2b + (1.0 - fb_frac) * f2a
        f3 = fb_frac * f3b + (1.0 - fb_frac) * f3a
    else
        f1 = 1.0
        f2 = 1.0
        f3 = 1.0
    end

    # correction to betan from Bird et al., eqn A10
    #betan += (fnu * (1.081 + 0.395*neff2))

    # eqns A1 - A3
    y = k ./ ksigma
    y2 = @. y * y
    fy = @. y/4.0 + y2/8.0
    DeltakL =  PkL .* delta2_norm

    # correction to DeltakL from Bird et al., eqn A9
    #kh = @. k / cpar.h
    #kh2 = @. kh * kh
    #DeltakL_tilde_fac = @. fnu * (47.48 * kh2) / (1.0 + 1.5 * kh2)
    #DeltakL_tilde = @. DeltakL * (1.0 + DeltakL_tilde_fac)
    #DeltakQ = @. DeltakL * (1.0 + DeltakL_tilde)^betan / (1.0 + DeltakL_tilde*alphan) * exp(-fy)
    DeltakQ = @. DeltakL * (1.0 + DeltakL)^betan / (1.0 + DeltakL*alphan) * exp(-fy)

    DeltakHprime = @. an * y^(3.0*f1) / (1.0 + bn*y^f2 + (cn*f3*y)^(3.0 - gamman))
    #DeltakH = @. DeltakHprime / (1.0 + mun/y + nun/y2)
    #
    # correction to DeltakH from Bird et al., eqn A6-A7
    #Qnu = @. fnu * (0.977 - 18.015 * (cpar.Ωm - 0.3))
    #DeltakH *= @. (1.0 + Qnu)
    DeltakH = @. DeltakHprime / (1.0 + nun/y2)

    DeltakNL = @. DeltakQ + DeltakH
    PkNL = @. DeltakNL / delta2_norm

    return PkNL
end
