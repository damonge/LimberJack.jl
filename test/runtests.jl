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
        cpar = LimberJack.CosmoPar{T}(Ωm, 0.05, 0.67, 0.96, 0.81)
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

#
# # @testset "test_sigma2" begin
# #     zs = 0.0
# #     a = 1.0
# #     lkmin = -4
# #     lkmax = 2
# #     nk = 256
# #     logk = range(lkmin, stop=lkmax, length=nk)
# #     k = 10 .^ logk
# #     logk = log.(k)
# #     cosmo = Cosmology()
# #     PkL = power_spectrum(cosmo, k, zs)
# #     rsigma = LimberJack.get_rsigma(PkL, logk)
# #     println(rsigma)
# #     rsigma = LimberJack.get_rsigma_test(PkL, logk)
# #     println(rsigma)
# # end
#
@testset "dsigma2/dPk" begin
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

    dsigma2 = ForwardDiff.gradient(pklin -> LimberJack.rsigma_func(rsigma, pklin, logk), PkL)

    dPkL = zeros(nk)
    dPkL[150] = 1e-4*PkL[150]
    PkL_hi = LimberJack.rsigma_func(rsigma, PkL .+ dPkL, logk)
    PkL_lo = LimberJack.rsigma_func(rsigma, PkL .- dPkL, logk)
    dsigma2_test = @. (PkL_hi - PkL_lo)/2dPkL[150]
    # deltalogk/(2pi^2)k^3 e^(-k^2 R^2)
    test = (logk[151] .- logk[150]) ./ 2.0 ./ pi^2 .* k .^ 3 .* exp.(.- k .^ 2 .* rsigma .^ 2)
    println("dsigma2/dPk")
    println("analytic = ", test[150])
    println("autodiff = ", dsigma2[150])
    println("numerical diff = ", dsigma2_test)

    @test all(@. (abs(dsigma2[150]/dsigma2_test-1) < 1E-3))
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

#     @test all(@. (abs(dsigma2[150]/dsigma2_test-1) < 1E-3))
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

    cosmo = Cosmology()
    PkL = power_spectrum(cosmo, k, zs)
    drsigma = ForwardDiff.gradient(pklin -> LimberJack.get_rsigma(pklin, logk), PkL)

    dPkL = zeros(nk)
    dPkL[150] = 1e-2*PkL[150]
    PkL_hi = LimberJack.get_rsigma(PkL .+ dPkL, logk)
    PkL_lo = LimberJack.get_rsigma(PkL .- dPkL, logk)
    drsigma_test = @. (PkL_hi - PkL_lo)/2dPkL[150]
    test = (logk[2] .- logk[1]) ./ 2.0 ./ pi^2 .* k .^ 3 .* exp.(.- k .^ 2 .* LimberJack.get_rsigma(PkL, logk) .^ 2)
    test1 = (-1) .* test ./ trapz(logk, LimberJack.onederiv_gauss_norm_int_func(logk, PkL, LimberJack.get_rsigma(PkL, logk)))
    println("drsigma/dPk")
    println("analytic = ", test1[150])
    println("autodiff = ", drsigma[150])
    println("numerical diff = ", drsigma_test)

#     @test all(@. (abs(drsigma[10]/drsigma_test-1) < 1E-3))
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

    cosmo = Cosmology()
    PkL = power_spectrum(cosmo, k, zs)
    Rkmat = Rkmats(cosmo, nk=nk, nz=1)

    dPkL = zeros(nk)
    dPkL[150] = 1e-4*PkL[150]
    PkL_hi = LimberJack._power_spectrum_nonlin_diff(cosmo, PkL .+ dPkL, k, logk, a)
    PkL_lo = LimberJack._power_spectrum_nonlin_diff(cosmo, PkL .- dPkL, k, logk, a)
    Rkmat_test = @. (PkL_hi - PkL_lo)/2dPkL[150]

    println(Rkmat[1, 150, 150])
    println(Rkmat_test[150])

    @test all(@. (abs(Rkmat[1, 1, 1]/Rkmat_test[1]-1) < 1E-3))
end