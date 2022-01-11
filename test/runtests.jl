using Test
using LimberJack
using ForwardDiff
using Trapz
using Zygote


@testset "BMChi" begin
    cosmo = Cosmology()
    ztest = [0.1, 0.5, 1.0, 3.0]
    chi = comoving_radial_distance(cosmo, ztest)
    chi_bm = [437.18951971,
              1973.14978475,
              3451.62027484,
              6639.61804355]
    @test all(@. (abs(chi/chi_bm-1.0) < 1E-4))
end

@testset "BMGrowth" begin
    cosmo = Cosmology()
    ztest = [0.1, 0.5, 1.0, 3.0]
    Dz = growth_factor(cosmo, ztest)
    Dz_bm = [0.9496636,
             0.77319003,
             0.61182745,
             0.31889837]
    # It'd be best if this was < 1E-4...
    @test all(@. (abs(Dz/Dz_bm-1.0) < 2E-4))
end

@testset "BMPkBBKS" begin
    cosmo = Cosmology()
    ks = [0.001, 0.01, 0.1, 1.0, 10.0]
    pk = power_spectrum(cosmo, ks, 0.)
    pk_bm = [2.01570296e+04,
             7.77178497e+04,
             1.04422728e+04,
             7.50841197e+01,
             2.02624683e-01]
    # It'd be best if this was < 1E-4...
    @test all(@. (abs(pk/pk_bm-1.0) < 3E-4))
end

@testset "BMPkEisHu" begin
    cosmo = Cosmology(0.25, 0.05, 0.67, 0.96, 0.81,
                      nk=1024, tk_mode="EisHu")
    ks = [0.001, 0.01, 0.1, 1.0, 10.0]
    pk = power_spectrum(cosmo, ks, 0.)
    pk_bm = [2.12222992e+04,
             8.83444294e+04,
             1.05452648e+04,
             8.22064850e+01,
             2.41173851e-01]
    # It'd be best if this was < 1E-4...
    @test all(@. (abs(pk/pk_bm-1.0) < 3E-4))
end

@testset "BMCℓs" begin
    cosmo = Cosmology()
    z = range(0., stop=2., length=1024)
    wz = @. exp(-0.5*((z-0.5)/0.05)^2)
    t = NumberCountsTracer(z, wz, 2.)
    ℓs = [10, 30, 100, 300]
    Cℓs = [angularCℓ(cosmo, t, t, ℓ) for ℓ in ℓs]
    Cℓs_bm = [7.02850428e-05,
              7.43987364e-05,
              2.92323380e-05,
              4.91394610e-06]
    # It'd be best if this was < 1E-4...
    @test all(@. (abs(Cℓs/Cℓs_bm-1.0) < 5E-4))
end

@testset "CreateTracer" begin
    z = range(0., stop=2., length=1024)
    wz = @. exp(-0.5*((z-0.5)/0.05)^2)
    t = NumberCountsTracer(z, wz, 2.)
    integ = 1.0/(sqrt(2π)*0.05)
    @test abs(t.wnorm/integ - 1) < 1E-4
end

@testset "CreateCosmo" begin
    cosmo = Cosmology()
    @test cosmo.cosmo.Ωm == 0.3
end

@testset "IsDiff" begin
    zs = 0.02:0.02:1.0

    function f(p::T)::Array{T,1} where T<:Real
        Ωm = p
        θCMB = 2.725/2.7
        cpar = LimberJack.CosmoPar{T}(Ωm, 0.05, 0.67, 0.96, 0.81, θCMB)
        cosmo = LimberJack.Cosmology(cpar)
        chi = comoving_radial_distance(cosmo, zs)
        return chi
    end

    Ωm0 = 0.3
    g = ForwardDiff.derivative(f, Ωm0)

    dΩm = 0.02
    g1 = (f(Ωm0+dΩm)-f(Ωm0-dΩm))/2dΩm
    @test all(@. (abs(g/g1-1) < 1E-3))
end

@testset "dsigma2/dPk" begin
    zs = 0.0
    a = 1.0
    lkmin = -4
    lkmax = 2
    nk = 256
    logk = range(lkmin, stop=lkmax, length=nk)
    k = 10 .^ logk
    logk = log.(k)

    ind = 150

    cosmo = Cosmology()
    PkL = power_spectrum(cosmo, k, zs)
    rsigma = LimberJack.get_rsigma(PkL, logk)

    dsigma2 = ForwardDiff.gradient(pklin -> LimberJack.rsigma_func(rsigma, pklin, logk), PkL)

    dPkL = zeros(nk)
    dPkL[ind] = 1e-4*PkL[ind]
    PkL_hi = LimberJack.rsigma_func(rsigma, PkL .+ dPkL, logk)
    PkL_lo = LimberJack.rsigma_func(rsigma, PkL .- dPkL, logk)
    dsigma2_test = @. (PkL_hi - PkL_lo)/2dPkL[ind]
    # deltalogk/(2pi^2)k^3 e^(-k^2 R^2)
    test = (logk[151] .- logk[150]) ./ 2.0 ./ pi^2 .* k .^ 3 .* exp.(.- k .^ 2 .* rsigma .^ 2)
    println("dsigma2/dPk")
    println("analytic = ", test[ind])
    println("autodiff = ", dsigma2[ind])
    println("numerical diff = ", dsigma2_test)

    @test all(@. (abs(dsigma2[ind]/dsigma2_test-1) < 1E-3))
end

@testset "dsigma2/dR" begin
    zs = 0.0
    a = 1.0
    lkmin = -4
    lkmax = 2
    nk = 256
    logk = range(lkmin, stop=lkmax, length=nk)
    k = 10 .^ logk
    logk = log.(k)
    cosmo = Cosmology()
    PkL = power_spectrum(cosmo, k, zs)
    rsigma = LimberJack.get_rsigma(PkL, logk)

    dsigma2 = ForwardDiff.derivative(rsig -> LimberJack.rsigma_func(rsig, PkL, logk), rsigma)

    drsigma = 1e-4*rsigma
    PkL_hi = LimberJack.rsigma_func(rsigma .+ drsigma, PkL, logk)
    PkL_lo = LimberJack.rsigma_func(rsigma .- drsigma, PkL, logk)
    dsigma2_test = @. (PkL_hi - PkL_lo)/2drsigma
    test = trapz(logk, LimberJack.onederiv_gauss_norm_int_func(logk, PkL, rsigma))
    println("dsigma2/dR")
    println("analytic = ", test)
    println("autodiff = ", dsigma2)
    println("numerical diff = ", dsigma2_test)

    @test all(@. (abs(dsigma2/dsigma2_test-1) < 1E-3))
end

@testset "drsigma/dPk" begin
    zs = 0.0
    a = 1.0
    lkmin = -4
    lkmax = 2
    nk = 256
    logk = range(lkmin, stop=lkmax, length=nk)
    k = 10 .^ logk
    logk = log.(k)

    ind = 150

    cosmo = Cosmology()
    PkL = power_spectrum(cosmo, k, zs)
    drsigma = ForwardDiff.gradient(pklin -> LimberJack.get_rsigma(pklin, logk), PkL)

    dPkL = zeros(nk)
    dPkL[ind] = 1e-2*PkL[ind]
    PkL_hi = LimberJack.get_rsigma(PkL .+ dPkL, logk)
    PkL_lo = LimberJack.get_rsigma(PkL .- dPkL, logk)
    drsigma_test = @. (PkL_hi - PkL_lo)/2dPkL[ind]
    # Analytic result:
    # sigma2(Pk, R) = 1
    # dsigma2 = dsigma2/dPk dPk + dsigma2/dR dR = 0
    # dR/dPk = (-dsigma2/dPk)/(dsigma2/dR)
    # dsigma2/dPk = Deltalogk k^3/(2pi&2)exp(-k^2R(sigma2=1)^2)
    # dsigma2/dR = -2R int dlogk k^5Pk/(2pi&2)exp(-k^2R(sigma2=1)^2)
    test = (logk[2] .- logk[1]) ./ 2.0 ./ pi^2 .* k .^ 3 .* exp.(.- k .^ 2 .* LimberJack.get_rsigma(PkL, logk) .^ 2)
    test1 = (-1) .* test ./ trapz(logk, LimberJack.onederiv_gauss_norm_int_func(logk, PkL, LimberJack.get_rsigma(PkL, logk)))
    println("drsigma/dPk")
    println("analytic = ", test1[ind])
    println("autodiff = ", drsigma[ind])
    println("numerical diff = ", drsigma_test)

    @test all(@. (abs(drsigma[ind]/drsigma_test-1) < 1E-3))
end

@testset "Rkmats" begin
    zs = 0.0
    a = 1.0
    lkmin = -4
    lkmax = 2
    nk = 256
    logk = range(lkmin, stop=lkmax, length=nk)
    k = 10 .^ logk
    logk = log.(k)

    ind = 150

    cosmo = Cosmology()
    PkL = power_spectrum(cosmo, k, zs)
    Rkmat = Rkmats(cosmo, nk=nk, nz=1)

    dPkL = zeros(nk)
    dPkL[ind] = 1e-4*PkL[ind]
    PkL_hi = LimberJack._power_spectrum_nonlin_diff(cosmo, PkL .+ dPkL, k, logk, a)
    PkL_lo = LimberJack._power_spectrum_nonlin_diff(cosmo, PkL .- dPkL, k, logk, a)
    Rkmat_test = @. (PkL_hi - PkL_lo)/2dPkL[ind]

    println("Rk")
    println("analytic = ", Rkmat_test[ind])
    println("autodiff = ", Rkmat[1, ind, ind])

    @test all(@. (abs(Rkmat[1, ind, ind]/Rkmat_test[ind]-1) < 1E-3))
end

@testset "Rkkmats" begin
    zs = 0.0
    a = 1.0
    lkmin = -4
    lkmax = 2
    nk = 20
    logk = range(lkmin, stop=lkmax, length=nk)
    k = 10 .^ logk
    logk = log.(k)

    ind = 1

    cosmo = Cosmology()
    PkL = power_spectrum(cosmo, k, zs)
    Rkkmat = Rkkmats(cosmo, nk=nk, nz=1)

    dPkL = zeros(nk)
    dPkL[ind] = 1e-1*PkL[ind]
    Rkmat_hi = LimberJack._Rkmat(cosmo, PkL .+ dPkL, k, logk, a)
    Rkmat_lo = LimberJack._Rkmat(cosmo, PkL .- dPkL, k, logk, a)
    Rkkmat_test = @. (Rkmat_hi - Rkmat_lo)/2dPkL[ind]

    println("Rkk")
    println("analytic = ", Rkkmat_test[ind])
    println("autodiff = ", Rkkmat[ind, ind, ind])

    zeromask = Rkkmat_test .> 0.0

    @test all(@. (abs(Rkkmat[ind, zeromask, ind]/Rkkmat_test[zeromask]-1) < 5E-3))
end