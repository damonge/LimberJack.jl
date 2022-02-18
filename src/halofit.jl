#=
halofit:
- Julia version: 
- Author: andrina
- Date: 2021-09-17
=#

function get_σ2(ks, pks, R, kind)
    lks = log.(ks)
    k2 = ks .^ 2
    k3 = k2 .* ks
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

function get_PKnonlin(cosmo::CosmoPar, z, k, lPkLz0, Dz)
    nk = length(k)
    nz = length(z)
    logk = log.(k)
    a = reverse(@. 1.0 / (1.0 + z))
    zs = reverse(z)
    nk = length(k)
    nz = length(z)

    PkLz0 = exp.(lPkLz0(log.(k)))
    Dz2s = Dz(zs) .^ 2
    lRs = range(log(0.1), stop=log(10.0), length=64)
    lσ2s = log.([get_σ2(k, PkLz0, exp(lR), 0) for lR in lRs])
    lσ2i = CubicSplineInterpolation(lRs, lσ2s)
    rsigs = [get_rsigma(lσ2i, Dz2s[i], log(0.1), log(10.0)) for i in range(1, stop=nz)]

    onederiv_ints = [get_σ2(k, PkLz0, rsigs[i], 2) * Dz2s[i] for i in range(1, stop=nz)]
    twoderiv_ints = [get_σ2(k, PkLz0, rsigs[i], 4) * Dz2s[i] for i in range(1, stop=nz)]
    neffs = @. 2*onederiv_ints - 3.0
    Cs = @. 4*(twoderiv_ints + onederiv_ints^2)

    # Interpolate linearily over a
    pk_NLs = [power_spectrum_nonlin(cosmo, PkLz0 .* Dz2s[i], k, a[i], rsigs[i], neffs[i], Cs[i])
              for i in range(1, stop=nz)]
    pk_NLs = reduce(vcat,transpose.(pk_NLs))
    pk_NLs = transpose(reverse(pk_NLs, dims=1))
    pk_NL = LinearInterpolation((log.(k), z), log.(pk_NLs))

    return pk_NL
end 

function get_rsigma(lσ2i, Dz2, lRmin, lRmax)
    lDz2 = log(Dz2)
    lRsigma = find_zero(lR -> lσ2i(lR)+lDz2, (lRmin, lRmax))
    return exp(lRsigma)
end

function power_spectrum_nonlin(cpar::CosmoPar, PkL, k, a, rsig, neff, C)
    zz = 1.0/a - 1.0
    Ez2 = cpar.Ωm*(1+zz)^3+cpar.Ωr*(1+zz)^4+cpar.ΩΛ
    omegaMz = cpar.Ωm*(1+zz)^3 / Ez2
    omegaDEwz = cpar.ΩΛ/Ez2

    # not using these to match CLASS better - might be a bug in CLASS
    # omegaMz = gsl_spline_eval(hf->omeff, a, NULL);
    # omegaDEwz = gsl_spline_eval(hf->deeff, a, NULL);

    ksigma = @. 1.0 / rsig
    neff2 = @. neff * neff
    neff3 = @. neff2 * neff
    neff4 = @. neff3 * neff

    delta2_norm = @. k*k*k/2.0/pi/pi

    # eqns A6 - A13 of Takahashi et al.
    an = @. 10.0^(1.5222 + 2.8553*neff + 2.3706*neff2 + 0.9903*neff3 +
        0.2250*neff4 - 0.6038*C)
    bn = @. 10.0^(-0.5642 + 0.5864*neff + 0.5716*neff2 - 1.5474*C)
    cn = @. 10.0^(0.3698 + 2.0404*neff + 0.8161*neff2 + 0.5869*C)
    gamman = @. 0.1971 - 0.0843*neff + 0.8460*C
    alphan = @. abs(6.0835 + 1.3373*neff - 0.1959*neff2 - 5.5274*C)
    betan = @. 2.0379 - 0.7354*neff + 0.3157*neff2 + 1.2490*neff3 + 0.3980*neff4 - 0.1682*C
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

    # eqns A1 - A3
    y = k ./ ksigma
    y2 = @. y * y
    fy = @. y/4.0 + y2/8.0
    DeltakL =  PkL .* delta2_norm

    DeltakQ = @. DeltakL * (1.0 + DeltakL)^betan / (1.0 + DeltakL*alphan) * exp(-fy)

    DeltakHprime = @. an * y^(3.0*f1) / (1.0 + bn*y^f2 + (cn*f3*y)^(3.0 - gamman))
    DeltakH = @. DeltakHprime / (1.0 + nun/y2)

    DeltakNL = @. DeltakQ + DeltakH
    PkNL = @. DeltakNL / delta2_norm

    return PkNL
end
