using Test
using LimberJack
using ForwardDiff

@testset "All tests" begin

    @testset "BMChi" begin
        cosmo = Cosmology()
        ztest = [0.1, 0.5, 1.0, 3.0]
        chi = comoving_radial_distance(cosmo, ztest)
        chi_bm = [437.1870424,
                  1973.09067532,
                  3451.41630697,
                  6638.67844433]
        @test all(@. (abs(chi/chi_bm-1.0) < 1E-4))
    end

    @testset "BMGrowth" begin
        cosmo = Cosmology()
        ztest = [0.1, 0.5, 1.0, 3.0]
        Dz = growth_factor(cosmo, ztest)
        Dz_bm = [0.94966513,
                 0.77320274,
                 0.61185874,
                 0.31898209]
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Dz/Dz_bm-1.0) < 2E-4))
    end

    @testset "BMPkBBKS" begin
        cosmo = Cosmology()
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk = lin_Pk(cosmo, 0., ks)
        print(pk)
        pk_bm = [2.01570296e+04,
                 7.77178497e+04,
                 1.04422728e+04,
                 7.50841197e+01,
                 2.02624683e-01]
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk/pk_bm-1.0) < 3E-4))
    end

    @testset "BMPkEisHu" begin
        cosmo = Cosmology(0.25, 0.05, 0.67, 0.96, 0.81, 2.725/2.7,
                          nk=1024, tk_mode="EisHu")
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk = lin_Pk(cosmo, 0., ks)
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
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo, z, nz, 2.)
        ts = WeakLensingTracer(cosmo, z, nz)
        tk = CMBLensingTracer(cosmo)
        ℓs = [10, 30, 100, 300]
        Cℓ_gg = [angularCℓ(cosmo, tg, tg, ℓ) for ℓ in ℓs]
        Cℓ_gs = [angularCℓ(cosmo, tg, ts, ℓ) for ℓ in ℓs]
        Cℓ_ss = [angularCℓ(cosmo, ts, ts, ℓ) for ℓ in ℓs]
        Cℓ_gk = [angularCℓ(cosmo, tg, tk, ℓ) for ℓ in ℓs]
        Cℓ_sk = [angularCℓ(cosmo, ts, tk, ℓ) for ℓ in ℓs]
        Cℓ_gg_bm = [7.02850428e-05, 7.43987364e-05,
                    2.92323380e-05, 4.91394610e-06]
        Cℓ_gs_bm = [7.26323570e-08, 7.29532942e-08,
                    2.65115994e-08, 4.23362515e-09]
        Cℓ_ss_bm = [1.75502191e-08, 8.22186845e-09,
                    1.52560567e-09, 1.75501782e-10]
        Cℓ_gk_bm = [1.21510532e-06, 1.28413154e-06,
                    5.04008449e-07, 8.48814786e-08]
        Cℓ_sk_bm = [2.97005787e-08, 1.62295093e-08,
                    3.53166134e-09, 4.43204907e-10]
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 5E-4))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 5E-4))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 5E-4))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 5E-4))
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 5E-4))
    end

    @testset "CreateTracer" begin
        p_of_z(x) = @. exp(-0.5*((x-0.5)/0.05)^2)

        z = range(0., stop=2., length=2048)
        pz = p_of_z(z)
        cosmo = Cosmology()
        t = NumberCountsTracer(cosmo, z, pz, 2.)

        wz1 = t.wint(cosmo.chi(0.5))
        hz = Hmpc(cosmo, 0.5)
        wz2 = p_of_z(0.5)*hz/(sqrt(2π)*0.05)

        @test abs(wz2/wz1 - 1) < 1E-4
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
            #cpar = LimberJack.CosmoPar{T}(Ωm, 0.05, 0.67, 0.96, 0.81, θCMB)
            cpar = LimberJack.CosmoPar(Ωm, 0.05, 0.67, 0.96, 0.81, θCMB)
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

end