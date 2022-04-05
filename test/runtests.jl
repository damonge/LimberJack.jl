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

    @testset "PkBBKS" begin
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81)
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk = nonlin_Pk(cosmo, ks, 0.0)
        pk_bm = [2.01570296e+04,
                 7.77178497e+04,
                 1.04422728e+04,
                 7.50841197e+01,
                 2.02624683e-01]
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk/pk_bm-1.0) < 3E-4))
    end

    @testset "PkEisHu" begin
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nk=768, tk_mode="EisHu")
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk = nonlin_Pk(cosmo, ks, 0.0)
        pk_bm = [2.12222992e+04,
                 8.83444294e+04,
                 1.05452648e+04,
                 8.22064850e+01,
                 2.41173851e-01]
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk/pk_bm-1.0) < 3E-4))
    end

    @testset "PkHalofit" begin
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nk=512, tk_mode="EisHu", 
                          Pk_mode="Halofit")
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk = nonlin_Pk(cosmo, ks, 0)
        pk_bm = [2.12015208e+04,
                 8.75109090e+04,
                 1.15273287e+04,
                 8.52170268e+02,
                 1.31682588e+01]
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk/pk_bm-1.0) < 1E-3))
    end
    
    @testset "BBKS_Cℓs" begin
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81)
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

    @testset "EisHu_Cℓs" begin
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nk=512, tk_mode="EisHu")
        z = range(0., stop=2., length=256)
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
        Cℓ_gg_bm = [7.60013901e-05, 8.33928286e-05,
                     3.05959806e-05, 4.89394772e-06]
        Cℓ_gs_bm = [7.90343478e-08, 8.04101857e-08,
                     2.73864193e-08, 4.24835821e-09]
        Cℓ_ss_bm = [1.88095054e-08, 8.11598350e-09,
                     1.51568997e-09, 1.81460354e-10]
        Cℓ_gk_bm = [1.31453613e-06, 1.43657960e-06,
                     5.24090831e-07, 8.49698739e-08]
        Cℓ_sk_bm = [3.21788793e-08, 1.61961883e-08,
                     3.50133537e-09, 4.54239420e-10]
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 5E-3))
        # The ℓ=10 point is a bit inaccurate for some reason
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 3E-3))
    end
    
    @testset "Halo_Cℓs" begin
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          tk_mode="EisHu", Pk_mode="Halofit")
        z = range(0., stop=2., length=256)
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
        Cℓ_gg_bm = [7.57445927e-05, 8.26337569e-05, 3.02763731e-05, 5.86734959e-06]
        Cℓ_gs_bm = [7.87411925e-08, 7.96026876e-08, 2.71972767e-08, 5.31526718e-09]
        Cℓ_ss_bm = [1.86729492e-08, 8.22346781e-09, 1.89325128e-09, 4.82204679e-10]
        Cℓ_gk_bm = [1.31077487e-06, 1.42425553e-06, 5.19243548e-07, 1.01852050e-07]
        Cℓ_sk_bm = [3.18928412e-08, 1.61941343e-08, 3.99846079e-09, 9.53760295e-10]
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 1E-2))
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

    @testset "IsBaseDiff" begin
        zs = 0.02:0.02:1.0

        function f(p::T)::Array{T,1} where T<:Real
            Ωm = p
            θCMB = 2.725/2.7
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81)
            chi = comoving_radial_distance(cosmo, zs)
            return chi
        end

        Ωm0 = 0.3
        g = ForwardDiff.derivative(f, Ωm0)

        dΩm = 0.02
        g1 = (f(Ωm0+dΩm)-f(Ωm0-dΩm))/2dΩm
        @test all(@. (abs(g/g1-1) < 1E-3))
    end

    @testset "IsLinPkDiff" begin
        zs = 0.02:0.02:1.0
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        function BBKS(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81)
            pk = lin_Pk(cosmo, ks, 0.)
            return pk
        end

        function EisHu(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81, tk_mode="EisHu")
            pk = lin_Pk(cosmo, ks, 0.)
            return pk
        end

        Ωm0 = 0.3
        dΩm = 0.001

        BBKS_autodiff = ForwardDiff.derivative(BBKS, Ωm0)
        EisHu_autodiff = ForwardDiff.derivative(EisHu, Ωm0)
        BBKS_anal = (BBKS(Ωm0+dΩm)-BBKS(Ωm0-dΩm))/2dΩm
        EisHu_anal = (EisHu(Ωm0+dΩm)-EisHu(Ωm0-dΩm))/2dΩm

        @test all(@. (abs(BBKS_autodiff/BBKS_anal-1) < 1E-3))
        @test all(@. (abs(EisHu_autodiff/EisHu_anal-1) < 1E-3))
    end
    
    @testset "AreClsDiff" begin
        
        function Cl_gg(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            z = range(0., stop=2., length=256)
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            tg = NumberCountsTracer(cosmo, z, nz, 2.)
            ℓs = [10, 30, 100, 300]
            Cℓ_gg = [angularCℓ(cosmo, tg, tg, ℓ) for ℓ in ℓs]
            return Cℓ_gg
        end
        
        function Cl_ss(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            z = range(0., stop=2., length=256)
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            ts = WeakLensingTracer(cosmo, z, nz)
            ℓs = [10, 30, 100, 300]
            Cℓ_ss = [angularCℓ(cosmo, ts, ts, ℓ) for ℓ in ℓs]
            return Cℓ_ss
        end
        
        function Cl_sk(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            z = range(0., stop=2., length=256)
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            ts = WeakLensingTracer(cosmo, z, nz)
            tk = CMBLensingTracer(cosmo)
            ℓs = [10, 30, 100, 300]
            Cℓ_sk = [angularCℓ(cosmo, ts, tk, ℓ) for ℓ in ℓs]
            return Cℓ_sk
        end

        Ωm0 = 0.3
        dΩm = 0.001

        Cl_gg_autodiff = ForwardDiff.derivative(Cl_gg, Ωm0)
        Cl_gg_anal = (Cl_gg(Ωm0+dΩm)-Cl_gg(Ωm0-dΩm))/2dΩm
        Cl_ss_autodiff = ForwardDiff.derivative(Cl_ss, Ωm0)
        Cl_ss_anal = (Cl_ss(Ωm0+dΩm)-Cl_ss(Ωm0-dΩm))/2dΩm
        Cl_sk_autodiff = ForwardDiff.derivative(Cl_sk, Ωm0)
        Cl_sk_anal = (Cl_sk(Ωm0+dΩm)-Cl_sk(Ωm0-dΩm))/2dΩm

        @test all(@. (abs(Cl_gg_autodiff/Cl_gg_anal-1) < 1E-2))
        @test all(@. (abs(Cl_ss_autodiff/Cl_ss_anal-1) < 1E-2))
        @test all(@. (abs(Cl_sk_autodiff/Cl_sk_anal-1) < 1E-2))
    end
    
    @testset "IsHalofitDiff" begin
        zs = 0.02:0.02:1.0
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        function Halofit(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            pk = LimberJack.nonlin_Pk(cosmo, ks, 0)
            return pk
        end

        Ωm0 = 0.3
        dΩm = 0.001

        Halofit_autodiff = ForwardDiff.derivative(Halofit, Ωm0)
        Halofit_anal = (Halofit(Ωm0+dΩm)-Halofit(Ωm0-dΩm))/2dΩm

        @test all(@. (abs(Halofit_autodiff/Halofit_anal-1) < 2E-2))
    end

    @testset "data" begin
        path = joinpath(pwd(), "data")
        datas = [Data("Dmygc", "Dmygc", 1 , 1, cl_path=path, cov_path=path),
                 Data("Dmywl", "Dmywl", 2 , 2, cl_path=path, cov_path=path),
                 Data("Dmygc", "Dmywl", 1 , 2, cl_path=path, cov_path=path)]
        Cls_metas = Cls_meta(datas, covs_path=path)
        @test Cls_metas.cls_names == ["Dmygc__1_Dmygc__1", "Dmywl__2_Dmywl__2", "Dmygc__1_Dmywl__2"]
        @test Cls_metas.tracers_names == ["Dmygc__1", "Dmywl__2"]
        @test Cls_metas.data_vector == [1, 2, 3]
        @test Cls_metas.cov_tot == [[11] [12] [13]; [12] [22] [23]; [13] [23] [33]]
    end

#=
    @testset "theory" begin
        path = joinpath(pwd(), "data")
        datas1 = [Data("Dmygc", "Dmygc", 1 , 1, cl_path=path, cov_path=path),
                  Data("Dmywl", "Dmywl", 2 , 2, cl_path=path, cov_path=path),
                  Data("Dmygc", "Dmywl", 1 , 2, cl_path=path, cov_path=path)]
        datas2 = [Data("Dmygc", "Dmygc", 1 , 1, cl_path=path, cov_path=path),
                  Data("Dmygc", "Dmywl", 1 , 2, cl_path=path, cov_path=path),
                  Data("Dmywl", "Dmywl", 2 , 2, cl_path=path, cov_path=path)]
        Cls_metas1 = Cls_meta(datas1, covs_path=path)
        Cls_metas2 = Cls_meta(datas2, covs_path=path)
        cosmo = LimberJack.Cosmology(0.3, 0.05, 0.67, 0.96, 0.81,
                                     tk_mode="EisHu", Pk_mode="Halofit")
        nuisances = Dict("b1"=> 2.0, "b2"=> 2.0)
        Nzs = [Nz(1; path=path), Nz(2; path=path), Nz(3; path=path),
               Nz(4; path=path), Nz(5; path=path)]
        theory1 = Theory(cosmo, Cls_metas1, Nzs, nuisances)
        theory2 = Theory(cosmo, Cls_metas2, Nzs, nuisances)
        match1 = [3.353603676098882e-5, 2.114314544813784e-9, 1.294707571087517e-7]
        match2 = [3.353603676098882e-5, 1.294707571087517e-7, 2.114314544813784e-9]
        tracers1 = [typeof(tracer) for tracer in theory1.tracers]
        tracers2 = [typeof(tracer) for tracer in theory2.tracers]
        @test tracers1 == tracers2
        @test all(@. (abs(theory1.Cls-match1) < 1E-5))
        @test all(@. (abs(theory2.Cls-match2) < 1E-5))
    end
=#    
end
