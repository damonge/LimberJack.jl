using Test
using LimberJack

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
