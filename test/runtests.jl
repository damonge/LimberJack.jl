using Test
using LimberJack
using ForwardDiff


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
    cosmo = Cosmology(tk_mode="Eis_Hu")
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
