#=
Rkmats:
- Julia version: 
- Author: andrina
- Date: 2021-09-22
=#

function Rkmats(cosmo::Cosmology; nk=256, nz=256)
    lkmin = -4
    lkmax = 2
    logk = range(lkmin, stop=lkmax, length=nk)
    k = 10 .^ logk
    logk = log.(k)
    zmin = 0.0
    if nz != 1
        zmax = 3.0
        zs = range(zmin, stop=zmax, length=nz)
    else
        zs = [zmin]
    end
    a = reverse(@. 1.0 / (1.0 + zs))
    zs = reverse(zs)

    PkL = zeros(nz, nk)
    Rkmat = zeros(nz, nk, nk)
    for i in range(1, stop=nz)
        PkL[i, :] .= power_spectrum(cosmo, k, zs[i])
        Rkmat[i, :, :] .= ForwardDiff.jacobian(pklin -> _power_spectrum_nonlin_diff(cosmo, pklin, k,
                        logk, a[i]), PkL[i, :])
    end

    return Rkmat
end

function _power_spectrum_nonlin_diff(cosmo::Cosmology, PkL, k, logk, a)

    rsig = get_rsigma(PkL, logk)
    sigma2 = rsigma_func(rsig, PkL, logk) + 1
    onederiv_int = trapz(logk, onederiv_gauss_norm_int_func(logk, PkL, rsig))
    neff = -rsig/sigma2*onederiv_int - 3.0
    twoderiv_int = trapz(logk, twoderiv_gauss_norm_int_func(logk, PkL, rsig))
    C = -(rsig^2/sigma2*twoderiv_int - rsig^2/sigma2^2*onederiv_int^2
                    + rsig/sigma2*onederiv_int)

    pkNL = power_spectrum_nonlin(cosmo, PkL, k, a, rsig, sigma2, neff, C)

    return pkNL
end

function Rkkmats(cosmo::Cosmology; nk=256, nz=256)
    lkmin = -4
    lkmax = 2
    logk = range(lkmin, stop=lkmax, length=nk)
    k = 10 .^ logk
    logk = log.(k)
    if nz != 1
        zmax = 3.
        zs = range(zmin, stop=zmax, length=nz)
    else
        zs = [zmin]
    end
    a = reverse(@. 1.0 / (1.0 + zs))
    zs = reverse(zs)

    PkL = zeros(nz, nk)
    Rkkmat = zeros(nz, nk^2, nk)
    for i in range(1, stop=nz)
        PkL[i, :] .= power_spectrum(cosmo, k, zs[i])
        Rkkmat[i, :, :] .= ForwardDiff.jacobian(pklin -> _Rkmat(cosmo, pklin, k,
                        logk, a[i]), PkL[i, :])
    end

    return Rkkmat
end

function _Rkmat(cosmo::Cosmology, PkL, k, logk, a)
    Rkmat = ForwardDiff.jacobian(pklin -> _power_spectrum_nonlin_diff(cosmo, pklin, k,
                        logk, a), PkL)
    return vec(Rkmat)
end

# function power_spectrum_nonlin_diff(cosmo::Cosmology, PkL::AbstractArray{T, 1}, k, a,
#     rsig, sigma2, neff, C)::AbstractArray{T, 1} where T<:Real
#
#     zs = @. 1.0/a - 1.0
#     weffa = -1.0
#     omegaMz = omega_x(cosmo, zs, "m")
#     omegaDEwz = omega_x(cosmo, zs,"l")
#
#     # not using these to match CLASS better - might be a bug in CLASS
#     # weffa = gsl_spline_eval(hf->weff, a, NULL);
#     # omegaMz = gsl_spline_eval(hf->omeff, a, NULL);
#     # omegaDEwz = gsl_spline_eval(hf->deeff, a, NULL);
#
#     ksigma = @. 1.0 / rsig
#     neff2 = @. neff * neff
#     neff3 = @. neff2 * neff
#     neff4 = @. neff3 * neff
#
#     delta2_norm = @. k*k*k/2.0/pi/pi
#
#     # compute the present day neutrino massive neutrino fraction
#     # uses all neutrinos even if they are moving fast
# #     om_nu = cosmo.sum_nu_masses / 93.14 / cosmo.h / cosmo.h
# #     fnu = om_nu / (cosmo.Ωm)
#     fnu = 0.0
#
#     # eqns A6 - A13 of Takahashi et al.
#     an = @. 10.0^(1.5222 + 2.8553*neff + 2.3706*neff2 + 0.9903*neff3 +
#         0.2250*neff4 - 0.6038*C + 0.1749*omegaDEwz*(1.0 + weffa))
#     bn = @. 10.0^(-0.5642 + 0.5864*neff + 0.5716*neff2 - 1.5474*C + 0.2279*omegaDEwz*(1.0 + weffa))
#     cn = @. 10.0^(0.3698 + 2.0404*neff + 0.8161*neff2 + 0.5869*C)
#     gamman = @. 0.1971 - 0.0843*neff + 0.8460*C
#     alphan = @. abs(6.0835 + 1.3373*neff - 0.1959*neff2 - 5.5274*C)
#     betan = @. 2.0379 - 0.7354*neff + 0.3157*neff2 + 1.2490*neff3 + 0.3980*neff4 - 0.1682*C
#     mun = 0.0
#     nun = @. 10.0^(5.2105 + 3.6902*neff)
#
#     # eqns C17 and C18 for Smith et al.
#     if abs(1.0 - omegaMz) > 0.01
#         f1a = omegaMz ^ -0.0732
#         f2a = omegaMz ^ -0.1423
#         f3a = omegaMz ^ 0.0725
#         f1b = omegaMz ^ -0.0307
#         f2b = omegaMz ^ -0.0585
#         f3b = omegaMz ^ 0.0743
#         fb_frac = omegaDEwz / (1.0 - omegaMz)
#         f1 = fb_frac * f1b + (1.0 - fb_frac) * f1a
#         f2 = fb_frac * f2b + (1.0 - fb_frac) * f2a
#         f3 = fb_frac * f3b + (1.0 - fb_frac) * f3a
#     else
#         f1 = 1.0
#         f2 = 1.0
#         f3 = 1.0
#     end
#
#     # correction to betan from Bird et al., eqn A10
#     betan += (fnu * (1.081 + 0.395*neff2))
#
#     # eqns A1 - A3
#     y = k ./ ksigma
#     y2 = @. y * y
#     fy = @. y/4.0 + y2/8.0
#     DeltakL =  PkL .* delta2_norm
#
#     # correction to DeltakL from Bird et al., eqn A9
#     kh = @. k / cosmo.cosmo.h
#     kh2 = @. kh * kh
#     DeltakL_tilde_fac = @. fnu * (47.48 * kh2) / (1.0 + 1.5 * kh2)
#     DeltakL_tilde = @. DeltakL * (1.0 + DeltakL_tilde_fac)
#     DeltakQ = @. DeltakL * (1.0 + DeltakL_tilde)^betan / (1.0 + DeltakL_tilde*alphan) * exp(-fy)
#
#     DeltakHprime = @. an * y^(3.0*f1) / (1.0 + bn*y^f2 + (cn*f3*y)^(3.0 - gamman))
#     DeltakH = @. DeltakHprime / (1.0 + mun/y + nun/y2)
#
#     # correction to DeltakH from Bird et al., eqn A6-A7
#     Qnu = @. fnu * (0.977 - 18.015 * (cosmo.cosmo.Ωm - 0.3))
#     DeltakH *= @. (1.0 + Qnu)
#
#     DeltakNL = @. DeltakQ + DeltakH
#     PkNL = @. DeltakNL / delta2_norm
#
#     return PkNL
# end