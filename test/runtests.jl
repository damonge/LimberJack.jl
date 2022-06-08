using Test
using LimberJack
using ForwardDiff
using PyCall 

ccl = pyimport("pyccl")
np = pyimport("numpy")

@testset "All tests" begin
    
    @testset "CreateCosmo" begin
        cosmo = Cosmology()
        @test cosmo.cosmo.Ωm == 0.3
    end
    
    @testset "BMHz" begin
        cosmo = Cosmology()
        cosmo_class = ccl.boltzmann.classy.Class()
        params = Dict("h"=>  0.67,
                      "Omega_cdm"=>  0.25,
                      "Omega_b"=>  0.05,
                      "Omega_Lambda"=>  0.6999068724497256,
                      "sigma8"=>  0.81)
        cosmo_class.set(params)
        cosmo_class.compute()
        
        c = 299792458.0
        ztest = [0.1, 0.5, 1.0, 3.0]
        H = cosmo.cosmo.h*100*Ez(cosmo, ztest)
        H_bm = [cosmo_class.Hubble(z)*c/1000 for z in ztest]
        @test all(@. (abs(H/H_bm-1.0) < 0.0005))
    end

    @testset "BMChi" begin
        cosmo = Cosmology()
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="bbks", 
                                            matter_power_spectrum="linear",
                                            Omega_g=0, Omega_k=0)
        ztest = [0.1, 0.5, 1.0, 3.0]
        chi = comoving_radial_distance(cosmo, ztest)
        chi_bm = ccl.comoving_radial_distance(cosmo_bm,
                                              1 ./ (1 .+ ztest)) 
        @test all(@. (abs(chi/chi_bm-1.0) < 0.0005))
    end

    @testset "BMGrowth" begin
        cosmo = Cosmology()
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="bbks", 
                                            matter_power_spectrum="linear",
                                            Omega_g=0, Omega_k=0)
        ztest = [0.1, 0.5, 1.0, 3.0]
        Dz = growth_factor(cosmo, ztest)
        Dz_bm = ccl.growth_factor(cosmo_bm, 1 ./ (1 .+ ztest))
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Dz/Dz_bm-1.0) < 0.0005))
    end

    @testset "PkBBKS" begin
        cosmo = Cosmology()
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="bbks", 
                                            matter_power_spectrum="linear",
                                            Omega_g=0, Omega_k=0)
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk = nonlin_Pk(cosmo, ks, 0.0)
        pk_bm = ccl.linear_matter_power(cosmo_bm, ks, 1.)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk/pk_bm-1.0) <  0.0005))
    end

    @testset "PkEisHu" begin
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nk=256, tk_mode="EisHu")
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu",
                                            matter_power_spectrum="linear",
                                            Omega_g=0, Omega_k=0)
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk = nonlin_Pk(cosmo, ks, 0.0)
        pk_bm = ccl.linear_matter_power(cosmo_bm, ks, 1.)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk/pk_bm-1.0) < 0.0005))
    end

    @testset "PkHalofit" begin
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nk=512, tk_mode="EisHu", 
                          Pk_mode="Halofit")
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu",
                                            matter_power_spectrum="halofit",
                                            Omega_g=0, Omega_k=0)
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk = nonlin_Pk(cosmo, ks, 0)
        pk_bm = ccl.nonlin_matter_power(cosmo_bm, ks, 1.)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk/pk_bm-1.0) < 1E-3))
    end
    
    @testset "CreateTracer" begin
        p_of_z(x) = @. exp(-0.5*((x-0.5)/0.05)^2)

        z = Vector(range(0., stop=2., length=200))
        nz = Vector(p_of_z(z))
        cosmo = Cosmology()
        t = NumberCountsTracer(cosmo, z, nz; bias=1.0)

        wz1 = t.wint(cosmo.chi(0.5))
        hz = Hmpc(cosmo, 0.5)
        wz2 = p_of_z(0.5)*hz/(sqrt(2π)*0.05)

        @test abs(wz2/wz1 - 1) < 1E-4
    end
    
    @testset "BBKS_Cℓs" begin
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81)
        z = Vector(range(0., stop=2., length=1024))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo, z, nz; bias=2.0)
        ts = WeakLensingTracer(cosmo, z, nz;
                               mbias=-1.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo)
        ℓs = [10.0, 30.0, 100.0, 300.0]
        Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs)
        Cℓ_gs = angularCℓs(cosmo, tg, ts, ℓs) 
        Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs) 
        Cℓ_gk = angularCℓs(cosmo, tg, tk, ℓs)
        Cℓ_sk = angularCℓs(cosmo, ts, tk, ℓs) 
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
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo, z, nz; bias=2.0)
        ts = WeakLensingTracer(cosmo, z, nz;
                               mbias=-1.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo)
        ℓs = [10.0, 30.0, 100.0, 300.0]
        Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
        Cℓ_gs = angularCℓs(cosmo, tg, ts, ℓs)
        Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs) 
        Cℓ_gk = angularCℓs(cosmo, tg, tk, ℓs) 
        Cℓ_sk = angularCℓs(cosmo, ts, tk, ℓs) 
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
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo, z, nz; bias=2.0)
        ts = WeakLensingTracer(cosmo, z, nz;
                               mbias=-1.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo)
        ℓs = [10.0, 30.0, 100.0, 300.0]
        Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs)
        Cℓ_gs = angularCℓs(cosmo, tg, ts, ℓs) 
        Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs) 
        Cℓ_gk = angularCℓs(cosmo, tg, tk, ℓs) 
        Cℓ_sk = angularCℓs(cosmo, ts, tk, ℓs) 
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
            z = Vector(range(0., stop=2., length=256))
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            tg = NumberCountsTracer(cosmo, z, nz; bias=2.0)
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
            return Cℓ_gg
        end
        
        function Cl_ss(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            z = Vector(range(0., stop=2., length=256))
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            ts = WeakLensingTracer(cosmo, z, nz;
                                   mbias=-1.0,
                                   IA_params=[0.0, 0.0])
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs)
            return Cℓ_ss
        end
        
        function Cl_sk(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            z = range(0., stop=2., length=256)
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            ts = WeakLensingTracer(cosmo, z, nz;
                                   mbias=-1.0,
                                   IA_params=[0.0, 0.0])
            tk = CMBLensingTracer(cosmo)
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_sk = angularCℓs(cosmo, ts, tk, ℓs)
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
end
