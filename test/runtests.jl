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
                          nk=500, tk_mode="EisHu")
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu",
                                            matter_power_spectrum="linear",
                                            Omega_g=0, Omega_k=0)
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk = nonlin_Pk(cosmo, ks, 0.0)
        pk_bm = ccl.linear_matter_power(cosmo_bm, ks, 1.)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk/pk_bm-1.0) < 0.001))
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
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nz=512)
        t = NumberCountsTracer(cosmo, z, nz, 1.0)

        wz1 = t.wint(cosmo.chi(0.5))
        hz = Hmpc(cosmo, 0.5)
        wz2 = p_of_z(0.5)*hz/(sqrt(2π)*0.05)

        @test abs(wz2/wz1 - 1) < 0.005
    end
    
    @testset "BBKS_Cℓs" begin
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="bbks", 
                                            matter_power_spectrum="linear",
                                            Omega_g=0, Omega_k=0)
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nz=512, nk=512)
        z = Vector(range(0., stop=2., length=1024))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo, z, nz, 2.0)
        ts = WeakLensingTracer(cosmo, z, nz)
        tk = CMBLensingTracer(cosmo)
        ℓs = [10, 30, 100, 300]
        Cℓ_gg = [angularCℓ(cosmo, tg, tg, ℓ) for ℓ in ℓs]
        Cℓ_gs = [angularCℓ(cosmo, tg, ts, ℓ) for ℓ in ℓs]
        Cℓ_ss = [angularCℓ(cosmo, ts, ts, ℓ) for ℓ in ℓs]
        Cℓ_gk = [angularCℓ(cosmo, tg, tk, ℓ) for ℓ in ℓs]
        Cℓ_sk = [angularCℓ(cosmo, ts, tk, ℓ) for ℓ in ℓs] 
        tg_bm = ccl.NumberCountsTracer(cosmo_bm, false, dndz=(z, nz), bias=(z, 2 .* np.ones_like(z)))
        ts_bm = ccl.WeakLensingTracer(cosmo_bm, dndz=(z, nz))
        tk_bm = ccl.CMBLensingTracer(cosmo_bm, z_source=1100)
        Cℓ_gg_bm = ccl.angular_cl(cosmo_bm, tg_bm, tg_bm, ℓs)
        Cℓ_gs_bm = ccl.angular_cl(cosmo_bm, tg_bm, ts_bm, ℓs)
        Cℓ_ss_bm = ccl.angular_cl(cosmo_bm, ts_bm, ts_bm, ℓs)
        Cℓ_gk_bm = ccl.angular_cl(cosmo_bm, tg_bm, tk_bm, ℓs)
        Cℓ_sk_bm = ccl.angular_cl(cosmo_bm, ts_bm, tk_bm, ℓs)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 5E-4))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 5E-4))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 5E-4))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 5E-4))
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 5E-3))
    end

    @testset "EisHu_Cℓs" begin
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu", 
                                            matter_power_spectrum="linear",
                                            Omega_g=0, Omega_k=0)
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nk=512, tk_mode="EisHu")
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo, z, nz, 2.0)
        ts = WeakLensingTracer(cosmo, z, nz)
        tk = CMBLensingTracer(cosmo)
        ℓs = [10, 30, 100, 300]
        Cℓ_gg = [angularCℓ(cosmo, tg, tg, ℓ) for ℓ in ℓs]
        Cℓ_gs = [angularCℓ(cosmo, tg, ts, ℓ) for ℓ in ℓs]
        Cℓ_ss = [angularCℓ(cosmo, ts, ts, ℓ) for ℓ in ℓs]
        Cℓ_gk = [angularCℓ(cosmo, tg, tk, ℓ) for ℓ in ℓs]
        Cℓ_sk = [angularCℓ(cosmo, ts, tk, ℓ) for ℓ in ℓs] 
        tg_bm = ccl.NumberCountsTracer(cosmo_bm, false, dndz=(z, nz), bias=(z, 2 .* np.ones_like(z)))
        ts_bm = ccl.WeakLensingTracer(cosmo_bm, dndz=(z, nz))
        tk_bm = ccl.CMBLensingTracer(cosmo_bm, z_source=1100)
        Cℓ_gg_bm = ccl.angular_cl(cosmo_bm, tg_bm, tg_bm, ℓs)
        Cℓ_gs_bm = ccl.angular_cl(cosmo_bm, tg_bm, ts_bm, ℓs)
        Cℓ_ss_bm = ccl.angular_cl(cosmo_bm, ts_bm, ts_bm, ℓs)
        Cℓ_gk_bm = ccl.angular_cl(cosmo_bm, tg_bm, tk_bm, ℓs)
        Cℓ_sk_bm = ccl.angular_cl(cosmo_bm, ts_bm, tk_bm, ℓs)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 5E-3))
        # The ℓ=10 point is a bit inaccurate for some reason
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 3E-3))
    end
    
    @testset "Halo_Cℓs" begin
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu", 
                                            matter_power_spectrum="halofit",
                                            Omega_g=0, Omega_k=0)
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          tk_mode="EisHu", Pk_mode="Halofit")
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo, z, nz, 2.0)
        ts = WeakLensingTracer(cosmo, z, nz)
        tk = CMBLensingTracer(cosmo)
        ℓs = [10, 30, 100, 300]
        Cℓ_gg = [angularCℓ(cosmo, tg, tg, ℓ) for ℓ in ℓs]
        Cℓ_gs = [angularCℓ(cosmo, tg, ts, ℓ) for ℓ in ℓs]
        Cℓ_ss = [angularCℓ(cosmo, ts, ts, ℓ) for ℓ in ℓs]
        Cℓ_gk = [angularCℓ(cosmo, tg, tk, ℓ) for ℓ in ℓs]
        Cℓ_sk = [angularCℓ(cosmo, ts, tk, ℓ) for ℓ in ℓs] 
        tg_bm = ccl.NumberCountsTracer(cosmo_bm, false, dndz=(z, nz), bias=(z, 2 .* np.ones_like(z)))
        ts_bm = ccl.WeakLensingTracer(cosmo_bm, dndz=(z, nz))
        tk_bm = ccl.CMBLensingTracer(cosmo_bm, z_source=1100)
        Cℓ_gg_bm = ccl.angular_cl(cosmo_bm, tg_bm, tg_bm, ℓs)
        Cℓ_gs_bm = ccl.angular_cl(cosmo_bm, tg_bm, ts_bm, ℓs)
        Cℓ_ss_bm = ccl.angular_cl(cosmo_bm, ts_bm, ts_bm, ℓs)
        Cℓ_gk_bm = ccl.angular_cl(cosmo_bm, tg_bm, tk_bm, ℓs)
        Cℓ_sk_bm = ccl.angular_cl(cosmo_bm, ts_bm, tk_bm, ℓs)
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
    
    @testset "AreClsDiff" begin
        
        function Cl_gg(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            z = Vector(range(0., stop=2., length=256))
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            tg = NumberCountsTracer(cosmo, z, nz, 2.0)
            ℓs = [10, 30, 100, 300]
            Cℓ_gg = [angularCℓ(cosmo, tg, tg, l) for l in ℓs]
            return Cℓ_gg
        end
        
        function Cl_ss(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            z = Vector(range(0., stop=2., length=256))
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            ts = WeakLensingTracer(cosmo, z, nz)
            ℓs = [10, 30, 100, 300]
            Cℓ_ss = [angularCℓ(cosmo, ts, ts, l) for l in ℓs]
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
            Cℓ_sk = [angularCℓ(cosmo, ts, tk, l) for l in ℓs]
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
    
end
