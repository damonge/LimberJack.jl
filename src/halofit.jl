#=
halofit:
- Julia version: 
- Author: andrina
- Date: 2021-09-17
=#

# OPT: do we need for this to be a structure?
struct PKnonlin
    a::Array
    k::Array
    rsig::AbstractInterpolation
    sigma2::AbstractInterpolation
    neff::AbstractInterpolation
    C::AbstractInterpolation
    pk_NL::AbstractInterpolation
end

PKnonlin(cosmo::CosmoPar, z, k, PkLs) = begin
    nk = length(k)
    nz = length(z)
    logk = log.(k)
    a = reverse(@. 1.0 / (1.0 + z))
    zs = reverse(z)
    nk = length(k)
    nz = length(z)
    # OPT: check order
    PkL = reverse(transpose(PkLs), dims=1)
    
    rsigs =[get_rsigma(PkL[i, :], logk) for i in range(1, stop=nz)]
    sigma2s = [rsigma_func(rsigs[i], PkL[i, :], logk) + 1 for i in range(1, stop=nz)]
    onederiv_ints = [trapz(logk, onederiv_gauss_norm_int_func(logk, PkL[i, :], rsigs[i])) for i in range(1, stop=nz)]
    neffs = @. -rsigs/sigma2s*onederiv_ints - 3.0
    twoderiv_ints = [trapz(logk, twoderiv_gauss_norm_int_func(logk, PkL[i, :], rsigs[i])) for i in range(1, stop=nz)]
    Cs = [-(rsigs[i]^2/sigma2s[i]*twoderiv_ints[i] - rsigs[i]^2/sigma2s[i]^2*onederiv_ints[i]^2
          + rsigs[i]/sigma2s[i]*onederiv_ints[i]) for i in range(1, stop=nz)]
    
    # Interpolate linearily over a
    rsig = LinearInterpolation(a, rsigs)
    sigma2 = LinearInterpolation(a, sigma2s)
    neff = LinearInterpolation(a, neffs)
    C = LinearInterpolation(a, Cs)

    pk_NLs = [power_spectrum_nonlin(cosmo, PkL[i, :], k, a[i], rsigs[i], sigma2s[i], neffs[i], Cs[i]) for i in range(1, stop=nz)]
    pk_NLs = reduce(vcat,transpose.(pk_NLs))
    pk_NLs = transpose(reverse(pk_NLs, dims=1))
    pk_NL = LinearInterpolation((k, z), pk_NLs)

    PKnonlin(a, k, rsig, sigma2, neff, C, pk_NL)
end 

function gauss_norm_int_func(logk, pk, R)
    k = exp.(logk)
    k2 = k .^ 2
    integ = @. pk*k*k2/2.0/pi^2*exp(-k2*R^2)
    return integ
end

function onederiv_gauss_norm_int_func(logk, pk, R)
    k = exp.(logk)
    k2 = k .^ 2
    integ = @. (-2.0*R*k2)*pk*k*k2/2.0/pi^2*exp(-k2*R^2)
    return integ
end

function twoderiv_gauss_norm_int_func(logk, pk, R)
    k = exp.(logk)
    k2 = k .^ 2
    R2 = R .^ 2
    integ = @. (-2.0*k2 + 4*k2^2*R2)*pk*k*k2/2.0/pi^2*exp(-k2*R2)
    return integ
end

# function whose root is \sigma^2{rsigma, a} = 1
function rsigma_func(rsigma, pk, logk)
    result = trapz(logk, gauss_norm_int_func(logk, pk, rsigma)) - 1.0
    return result
end

function solve_for_rsigma(pk, logk)
    rlow = 1e-2
    rhigh = 1e2
#     f1 = rsig -> rsigma_func(rsig, pk, logk)
#     f2 = rsig -> trapz(logk, onederiv_gauss_norm_int_func(logk, pk, rsig))
#     rsigma = find_zero((f1, f2), 5.5, Roots.Newton())
    rsigma = find_zero(rsig -> rsigma_func(rsig, pk, logk), (rlow, rhigh))
    return rsigma
end

function solve_for_rsigma(pk::Vector{D}, logk) where D<:ForwardDiff.Dual
    pk0 = ForwardDiff.value.(pk)
    rsig0 = solve_for_rsigma(pk0, logk)
    ∂pk = ForwardDiff.gradient(pkl -> rsigma_func(rsig0, pkl, logk), pk0)
    ∂rsig = ForwardDiff.derivative(rsig -> rsigma_func(rsig, pk0, logk), rsig0)
    parts_in = zip(collect.(ForwardDiff.partials.(pk))...)
    parts_out = map(x->dot(-∂pk./∂rsig, x), parts_in)
    # (parts_out...,) from vector to (*, *, *)
    p = ForwardDiff.Partials((parts_out...,))
    return D(rsig0, p)
end

function get_rsigma(pk, logk)
    rlow = 1e-2
    rhigh = 1e2
    # we have to bound the root, otherwise return -1
    # we will fiil in any -1's in the calling routine
    flow = rsigma_func(rlow, pk, logk)
    fhigh = rsigma_func(rhigh, pk, logk)

    if flow * fhigh > 0
        return -1
    end

    rsigma = solve_for_rsigma(pk, logk)

    return rsigma
end

function power_spectrum_nonlin(cpar::CosmoPar, PkL, k, a, rsig, sigma2, neff, C)

    zs = @. 1.0/a - 1.0
    weffa = -1.0
    Ez = @. sqrt(cpar.Ωm*(1+zs)^3+cpar.Ωr*(1+zs)^4+cpar.ΩΛ)
    omegaMz = @. cpar.Ωm / (a^3) / Ez^2 
    omegaDEwz = (1 - cpar.Ωm)/Ez^2

    # not using these to match CLASS better - might be a bug in CLASS
    # weffa = gsl_spline_eval(hf->weff, a, NULL);
    # omegaMz = gsl_spline_eval(hf->omeff, a, NULL);
    # omegaDEwz = gsl_spline_eval(hf->deeff, a, NULL);

    ksigma = @. 1.0 / rsig
    neff2 = @. neff * neff
    neff3 = @. neff2 * neff
    neff4 = @. neff3 * neff

    delta2_norm = @. k*k*k/2.0/pi/pi

    # compute the present day neutrino massive neutrino fraction
    # uses all neutrinos even if they are moving fast
#     om_nu = cosmo.sum_nu_masses / 93.14 / cosmo.h / cosmo.h
#     fnu = om_nu / (cosmo.Ωm)
    fnu = 0.0

    # eqns A6 - A13 of Takahashi et al.
    an = @. 10.0^(1.5222 + 2.8553*neff + 2.3706*neff2 + 0.9903*neff3 +
        0.2250*neff4 - 0.6038*C + 0.1749*omegaDEwz*(1.0 + weffa))
    bn = @. 10.0^(-0.5642 + 0.5864*neff + 0.5716*neff2 - 1.5474*C + 0.2279*omegaDEwz*(1.0 + weffa))
    cn = @. 10.0^(0.3698 + 2.0404*neff + 0.8161*neff2 + 0.5869*C)
    gamman = @. 0.1971 - 0.0843*neff + 0.8460*C
    alphan = @. abs(6.0835 + 1.3373*neff - 0.1959*neff2 - 5.5274*C)
    betan = @. 2.0379 - 0.7354*neff + 0.3157*neff2 + 1.2490*neff3 + 0.3980*neff4 - 0.1682*C
    mun = 0.0
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
    betan += (fnu * (1.081 + 0.395*neff2))

    # eqns A1 - A3
    y = k ./ ksigma
    y2 = @. y * y
    fy = @. y/4.0 + y2/8.0
    DeltakL =  PkL .* delta2_norm

    # correction to DeltakL from Bird et al., eqn A9
    kh = @. k / cpar.h
    kh2 = @. kh * kh
    DeltakL_tilde_fac = @. fnu * (47.48 * kh2) / (1.0 + 1.5 * kh2)
    DeltakL_tilde = @. DeltakL * (1.0 + DeltakL_tilde_fac)
    DeltakQ = @. DeltakL * (1.0 + DeltakL_tilde)^betan / (1.0 + DeltakL_tilde*alphan) * exp(-fy)

    DeltakHprime = @. an * y^(3.0*f1) / (1.0 + bn*y^f2 + (cn*f3*y)^(3.0 - gamman))
    DeltakH = @. DeltakHprime / (1.0 + mun/y + nun/y2)

    # correction to DeltakH from Bird et al., eqn A6-A7
    Qnu = @. fnu * (0.977 - 18.015 * (cpar.Ωm - 0.3))
    DeltakH *= @. (1.0 + Qnu)

    DeltakNL = @. DeltakQ + DeltakH
    PkNL = @. DeltakNL / delta2_norm

    return PkNL
end
