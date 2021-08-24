using Test
using LimberJack

@testset "CreateCosmo" begin
    cosmo = Cosmology()
    @test cosmo.cosmo.Ωm == 0.3
end

@testset "CreateTracer" begin
    z = range(0., stop=2., length=1024)
    wz = @. exp(-0.5*((z-0.5)/0.05)^2)
    t = NumberCountsTracer(z, wz, 2.)
    integ = 1.0/(sqrt(2π)*0.05)
    @test abs(t.wnorm/integ - 1) < 1E-4
end

@testset "ComputeCℓ" begin
    cosmo = Cosmology()
    z = range(0., stop=2., length=1024)
    wz = @. exp(-0.5*((z-0.5)/0.05)^2)
    t = NumberCountsTracer(z, wz, 2.)
    ℓs = 2:10:1000
    Cℓs = [angularCℓ(cosmo, t, t, ℓ) for ℓ in ℓs]
end
