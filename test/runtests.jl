using Test
using LimberJack
using ForwardDiff
using PythonCall
using CondaPkg

CondaPkg.add("pyccl")

ccl = pyimport("pyccl")
np = pyimport("numpy")

@testset "All tests" begin
    
    @testset "CreateCosmo" begin
        cosmo = Cosmology()
        @test cosmo.cosmo.Ωm == 0.3
    end
    
    @testset "BMHz" begin
        cosmo = Cosmology()
        c = 299792458.0
        ztest = [0.1, 0.5, 1.0, 3.0]
        H = cosmo.cosmo.h*100*Ez(cosmo, ztest)
        H_bm = @. 67*sqrt(0.3 * (1+ztest)^3 + (1-0.3-0.69991) * (1+ztest)^4 + 0.69991)
        @test all(@. (abs(H/H_bm-1.0) < 0.0005))
    end

    @testset "BMChi" begin
        cosmo = Cosmology()
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="bbks", 
                                            matter_power_spectrum="linear",
                                            Omega_g=0, Omega_k=0)
        ztest = [0.1, 0.5, 1.0, 3.0]
        chi = comoving_radial_distance(cosmo, ztest)
        chi_bm = ccl.comoving_radial_distance(cosmo_bm,  1 ./ (1 .+ ztest))
        chi_bm = pyconvert(Vector, chi_bm)
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
        Dz_bm = pyconvert(Vector, Dz_bm)
        @test all(@. (abs(Dz/Dz_bm-1.0) < 0.0005))
    end

    @testset "linear_Pk" begin
        cosmo_BBKS = Cosmology()
        cosmo_BBKS_bm = ccl.CosmologyVanillaLCDM(transfer_function="bbks", 
                                                 matter_power_spectrum="linear",
                                                 Omega_g=0, Omega_k=0)
        cosmo_EisHu = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                                nk=500, tk_mode="EisHu")
        cosmo_EisHu_bm = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu",
                                                  matter_power_spectrum="linear",
                                                  Omega_g=0, Omega_k=0)
        cosmo_emul = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                               nk=500, tk_mode="emulator")
        cosmo_emul_bm = ccl.CosmologyVanillaLCDM(transfer_function="boltzmann_class",
                                                 matter_power_spectrum="linear",
                                                 Omega_g=0, Omega_k=0)
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk_BBKS = nonlin_Pk(cosmo_BBKS, ks, 0.0)
        pk_EisHu = nonlin_Pk(cosmo_EisHu, ks, 0.0)
        pk_emul = nonlin_Pk(cosmo_emul, ks, 0.0)
        
        pk_BBKS_bm = ccl.linear_matter_power(cosmo_BBKS_bm, ks, 1.)
        pk_EisHu_bm = ccl.linear_matter_power(cosmo_EisHu_bm, ks, 1.)
        pk_emul_bm = ccl.linear_matter_power(cosmo_emul_bm, ks, 1.)

        pk_BBKS_bm = pyconvert(Vector, pk_BBKS_bm)
        pk_EisHu_bm = pyconvert(Vector, pk_EisHu_bm)
        pk_emul_bm = pyconvert(Vector, pk_emul_bm)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk_BBKS/pk_BBKS_bm-1.0) <  0.0005))
        @test all(@. (abs(pk_EisHu/pk_EisHu_bm-1.0) <  0.0005))
        @test all(@. (abs(pk_emul/pk_emul_bm-1.0) <  0.05))
    end

    @testset "nonlinear_Pk" begin
        cosmo_BBKS = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                               nk=512, tk_mode="BBKS", 
                               Pk_mode="Halofit")
        cosmo_BBKS_bm = ccl.CosmologyVanillaLCDM(transfer_function="bbks", 
                                                 matter_power_spectrum="halofit",
                                                 Omega_g=0, Omega_k=0)
        cosmo_EisHu = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                                nk=512, tk_mode="EisHu", 
                                Pk_mode="Halofit")
        cosmo_emul = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                               nk=512, tk_mode="emulator", 
                               Pk_mode="Halofit")
        cosmo_EisHu_bm = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu",
                                            matter_power_spectrum="halofit",
                                            Omega_g=0, Omega_k=0)
        cosmo_emul_bm = ccl.CosmologyVanillaLCDM(transfer_function="boltzmann_class",
                                            matter_power_spectrum="halofit",
                                            Omega_g=0, Omega_k=0)
        ks = [0.001, 0.01, 0.1, 1.0, 10.0]
        pk_BBKS = nonlin_Pk(cosmo_BBKS, ks, 0.0)
        pk_EisHu = nonlin_Pk(cosmo_EisHu, ks, 0)
        pk_emul = nonlin_Pk(cosmo_emul, ks, 0)
        pk_BBKS_bm = ccl.nonlin_matter_power(cosmo_BBKS_bm, ks, 1.)
        pk_EisHu_bm = ccl.nonlin_matter_power(cosmo_EisHu_bm, ks, 1.)
        pk_emul_bm = ccl.nonlin_matter_power(cosmo_emul_bm, ks, 1.)
        
        pk_BBKS_bm = pyconvert(Vector, pk_BBKS_bm)
        pk_EisHu_bm = pyconvert(Vector, pk_EisHu_bm)
        pk_emul_bm = pyconvert(Vector, pk_emul_bm)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk_BBKS/pk_BBKS_bm-1.0) < 0.05))
        @test all(@. (abs(pk_EisHu/pk_EisHu_bm-1.0) < 1E-3))
        @test all(@. (abs(pk_emul/pk_emul_bm-1.0) < 0.05))
    end

    @testset "CreateTracer" begin
        p_of_z(x) = @. exp(-0.5*((x-0.5)/0.05)^2)

        z = Vector(range(0., stop=2., length=200))
        nz = Vector(p_of_z(z))
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nz=512)
        t = NumberCountsTracer(cosmo, z, nz, b=1.0)

        wz1 = t.wint(cosmo.chi(0.5))
        hz = Hmpc(cosmo, 0.5)
        wz2 = p_of_z(0.5)*hz/(sqrt(2π)*0.05)

        @test abs(wz2/wz1 - 1) < 0.005
    end

    @testset "EisHu_Cℓs" begin
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu", 
                                            matter_power_spectrum="linear",
                                            Omega_g=0, Omega_k=0)
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nk=512, tk_mode="EisHu")
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)

        tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
        ts = WeakLensingTracer(cosmo, z, nz;
                               mb=0.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo)
        ℓs = [10.0, 30.0, 100.0, 300.0]
        Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs)
        Cℓ_gs = angularCℓs(cosmo, tg, ts, ℓs)
        Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs)
        Cℓ_gk = angularCℓs(cosmo, tg, tk, ℓs)
        Cℓ_sk = angularCℓs(cosmo, ts, tk, ℓs)

        tg_bm = ccl.NumberCountsTracer(cosmo_bm, false, dndz=(z, nz), bias=(z, 1 .* np.ones_like(z)))
        ts_bm = ccl.WeakLensingTracer(cosmo_bm, dndz=(z, nz))
        tk_bm = ccl.CMBLensingTracer(cosmo_bm, z_source=1100)
        Cℓ_gg_bm = ccl.angular_cl(cosmo_bm, tg_bm, tg_bm, ℓs)
        Cℓ_gs_bm = ccl.angular_cl(cosmo_bm, tg_bm, ts_bm, ℓs)
        Cℓ_ss_bm = ccl.angular_cl(cosmo_bm, ts_bm, ts_bm, ℓs)
        Cℓ_gk_bm = ccl.angular_cl(cosmo_bm, tg_bm, tk_bm, ℓs)
        Cℓ_sk_bm = ccl.angular_cl(cosmo_bm, ts_bm, tk_bm, ℓs)
        
        Cℓ_gg_bm = pyconvert(Vector, Cℓ_gg_bm)
        Cℓ_gs_bm = pyconvert(Vector, Cℓ_gs_bm)
        Cℓ_ss_bm = pyconvert(Vector, Cℓ_ss_bm)
        Cℓ_gk_bm = pyconvert(Vector, Cℓ_gk_bm)
        Cℓ_sk_bm = pyconvert(Vector, Cℓ_sk_bm)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 5E-3))
        # The ℓ=10 point is a bit inaccurate for some reason
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 3E-3))
    end

"""
    @testset "emul_Cℓs" begin
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="boltzmann_class", 
                                            matter_power_spectrum="linear",
                                            Omega_g=0, Omega_k=0)
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          nk=512, tk_mode="emulator")
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)

        tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
        ts = WeakLensingTracer(cosmo, z, nz;
                               mb=0.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo)
        ℓs = [10.0, 30.0, 100.0, 300.0]
        Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
        Cℓ_gs = angularCℓs(cosmo, tg, ts, ℓs)
        Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs) 
        Cℓ_gk = angularCℓs(cosmo, tg, tk, ℓs) 
        Cℓ_sk = angularCℓs(cosmo, ts, tk, ℓs) 
        tg_bm = ccl.NumberCountsTracer(cosmo_bm, false, dndz=(z, nz), bias=(z, 1 .* np.ones_like(z)))
        ts_bm = ccl.WeakLensingTracer(cosmo_bm, dndz=(z, nz))
        tk_bm = ccl.CMBLensingTracer(cosmo_bm, z_source=1100)
        Cℓ_gg_bm = ccl.angular_cl(cosmo_bm, tg_bm, tg_bm, ℓs)
        Cℓ_gs_bm = ccl.angular_cl(cosmo_bm, tg_bm, ts_bm, ℓs)
        Cℓ_ss_bm = ccl.angular_cl(cosmo_bm, ts_bm, ts_bm, ℓs)
        Cℓ_gk_bm = ccl.angular_cl(cosmo_bm, tg_bm, tk_bm, ℓs)
        Cℓ_sk_bm = ccl.angular_cl(cosmo_bm, ts_bm, tk_bm, ℓs)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.05))
        # The ℓ=10 point is a bit inaccurate for some reason
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.05))
    end

    @testset "EisHu_Halo_Cℓs" begin
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu", 
                                            matter_power_spectrum="halofit",
                                            Omega_g=0, Omega_k=0)
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          tk_mode="EisHu", Pk_mode="Halofit")
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
        ts = WeakLensingTracer(cosmo, z, nz;
                               mb=0.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo)
        ℓs = [10.0, 30.0, 100.0, 300.0]
        Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs)
        Cℓ_gs = angularCℓs(cosmo, tg, ts, ℓs) 
        Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs) 
        Cℓ_gk = angularCℓs(cosmo, tg, tk, ℓs) 
        Cℓ_sk = angularCℓs(cosmo, ts, tk, ℓs) 
        tg_bm = ccl.NumberCountsTracer(cosmo_bm, false, dndz=(z, nz), bias=(z, 1 .* np.ones_like(z)))
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

    @testset "emul_Halo_Cℓs" begin
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="boltzmann_class", 
                                            matter_power_spectrum="halofit",
                                            Omega_g=0, Omega_k=0)
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          tk_mode="emulator", Pk_mode="Halofit")
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
        ts = WeakLensingTracer(cosmo, z, nz;
                               mb=0.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo)
        ℓs = [10.0, 30.0, 100.0, 300.0]
        Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs)
        Cℓ_gs = angularCℓs(cosmo, tg, ts, ℓs) 
        Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs) 
        Cℓ_gk = angularCℓs(cosmo, tg, tk, ℓs) 
        Cℓ_sk = angularCℓs(cosmo, ts, tk, ℓs) 
        tg_bm = ccl.NumberCountsTracer(cosmo_bm, false, dndz=(z, nz), bias=(z, 1 .* np.ones_like(z)))
        ts_bm = ccl.WeakLensingTracer(cosmo_bm, dndz=(z, nz))
        tk_bm = ccl.CMBLensingTracer(cosmo_bm, z_source=1100)
        Cℓ_gg_bm = ccl.angular_cl(cosmo_bm, tg_bm, tg_bm, ℓs)
        Cℓ_gs_bm = ccl.angular_cl(cosmo_bm, tg_bm, ts_bm, ℓs)
        Cℓ_ss_bm = ccl.angular_cl(cosmo_bm, ts_bm, ts_bm, ℓs)
        Cℓ_gk_bm = ccl.angular_cl(cosmo_bm, tg_bm, tk_bm, ℓs)
        Cℓ_sk_bm = ccl.angular_cl(cosmo_bm, ts_bm, tk_bm, ℓs)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.05))
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
        ks = [0.001, 0.01, 0.1, 1.0, 7.0]
        
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
        
        function emul(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.04, 0.67, 0.96, 0.81, tk_mode="emulator")
            pk = lin_Pk(cosmo, ks, 0.)
            return pk
        end

        Ωm0 = 0.3
        dΩm = 0.001

        BBKS_autodiff = ForwardDiff.derivative(BBKS, Ωm0)
        EisHu_autodiff = ForwardDiff.derivative(EisHu, Ωm0)
        emul_autodiff = ForwardDiff.derivative(emul, Ωm0)
        BBKS_anal = (BBKS(Ωm0+dΩm)-BBKS(Ωm0-dΩm))/2dΩm
        EisHu_anal = (EisHu(Ωm0+dΩm)-EisHu(Ωm0-dΩm))/2dΩm
        emul_anal = (emul(Ωm0+dΩm)-emul(Ωm0-dΩm))/2*0.005

        @test all(@. (abs(BBKS_autodiff/BBKS_anal-1) < 1E-3))
        @test all(@. (abs(EisHu_autodiff/EisHu_anal-1) < 1E-3))
        #@test all(@. (abs(emul_autodiff/emul_anal-1) < 0.5))
    end
    

    @testset "IsEisHuHalofitDiff" begin

        zs = 0.02:0.02:1.0
        ks = [0.001, 0.01, 0.1, 1.0, 7.0]
        
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

    @testset "IsemulHalofitDiff" begin

        zs = 0.02:0.02:1.0
        ks = [0.001, 0.01, 0.1, 1.0, 7.0]
        
        function Halofit(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.04, 0.67, 0.96, 0.81,
                                         tk_mode="emulator", Pk_mode="Halofit")
            pk = LimberJack.nonlin_Pk(cosmo, ks, 0)
            return pk
        end

        Ωm0 = 0.3
        dΩm = 0.000S5

        Halofit_autodiff = ForwardDiff.derivative(Halofit, Ωm0)
        Halofit_anal = (Halofit(Ωm0+dΩm)-Halofit(Ωm0-dΩm))/2dΩm

        @test all(@. (abs(Halofit_autodiff/Halofit_anal-1) < 0.5))
    end
    
    @testset "AreClsDiff" begin
        
        function Cl_gg(p::T)::Array{T,1} where T<:Real
            Ωm = p
            cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            z = Vector(range(0., stop=2., length=256))
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
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
                                   mb=0.0,
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
                                   mb=0.0,
                                   IA_params=[0.0, 0.0])
            tk = CMBLensingTracer(cosmo)
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_sk = angularCℓs(cosmo, ts, tk, ℓs)
            return Cℓ_sk
        end

        Ωm0 = 0.3
        dΩm = 0.0001

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

    @testset "Nuisances" begin
        cosmo_bm = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu", 
                                            matter_power_spectrum="halofit",
                                            Omega_g=0, Omega_k=0)
        cosmo = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81,
                          tk_mode="EisHu", Pk_mode="Halofit")
        z = Vector(range(0., stop=2., length=1024))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg_b = NumberCountsTracer(cosmo, z, nz; b=2.0)
        ts_m = WeakLensingTracer(cosmo, z, nz; mb=1.0)
        ts_IA = WeakLensingTracer(cosmo, z, nz; IA_params=[0.1, 0.1])
        ℓs = [10.0, 30.0, 100.0, 300.0]
        Cℓ_gg_b = angularCℓs(cosmo, tg_b, tg_b, ℓs)
        Cℓ_ss_m = angularCℓs(cosmo, ts_m, ts_m, ℓs)
        Cℓ_ss_IA = angularCℓs(cosmo, ts_IA, ts_IA, ℓs)
        IA_corr = @.(0.1*((1 + z)/1.62)^0.1 * (0.0134*cosmo.cosmo.Ωm/cosmo.Dz(z)))
        tg_b_bm = ccl.NumberCountsTracer(cosmo_bm, false, dndz=(z, nz), bias=(z, 2.0 .* np.ones_like(z)))
        ts_m_bm = ccl.WeakLensingTracer(cosmo_bm, dndz=(z, nz))
        ts_IA_bm = ccl.WeakLensingTracer(cosmo_bm, dndz=(z, nz), ia_bias=(z, IA_corr))
        Cℓ_gg_b_bm = ccl.angular_cl(cosmo_bm, tg_b_bm, tg_b_bm, ℓs)
        Cℓ_ss_m_bm = (1.0 + 1.0).^2 .* ccl.angular_cl(cosmo_bm, ts_m_bm, ts_m_bm, ℓs)
        Cℓ_ss_IA_bm = ccl.angular_cl(cosmo_bm, ts_IA_bm, ts_IA_bm, ℓs)
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg_b/Cℓ_gg_b_bm-1.0) < 5E-3))
        @test all(@. (abs(Cℓ_ss_m/Cℓ_ss_m_bm-1.0) < 1E-2))
        @test all(@. (abs(Cℓ_ss_IA/Cℓ_ss_IA_bm-1.0) < 1E-2))
    end

    @testset "AreNuisancesDiff" begin
        
        function bias(p::T)::Array{T,1} where T<:Real
            cosmo = LimberJack.Cosmology(0.3, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            cosmo.settings.cosmo_type = typeof(p)
            z = Vector(range(0., stop=2., length=256))
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            tg = NumberCountsTracer(cosmo, z, nz; b=p)
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
            return Cℓ_gg
        end
        
        function dz(p::T)::Array{T,1} where T<:Real
            cosmo = LimberJack.Cosmology(0.3, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            cosmo.settings.cosmo_type = typeof(p)
            z = Vector(range(0., stop=2., length=256)) .- p
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            tg = NumberCountsTracer(cosmo, z, nz)
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
            return Cℓ_gg
        end
        
        function mbias(p::T)::Array{T,1} where T<:Real
            cosmo = LimberJack.Cosmology(0.3, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            cosmo.settings.cosmo_type = typeof(p)
            z = range(0., stop=2., length=256)
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            ts = WeakLensingTracer(cosmo, z, nz;
                                   mb=p,
                                   IA_params=[0.0, 0.0])
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_sk = angularCℓs(cosmo, ts, ts, ℓs)
            return Cℓ_sk
        end
        
        function IA_A(p::T)::Array{T1,} where T<:Real
            cosmo = LimberJack.Cosmology(0.3, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            cosmo.settings.cosmo_type = typeof(p)
            z = range(0., stop=2., length=256)
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            ts = WeakLensingTracer(cosmo, z, nz;
                                   IA_params=[p, 0.1])
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs)
            return Cℓ_ss
        end
        
        function IA_alpha(p::T)::Array{T,1} where T<:Real
            cosmo = LimberJack.Cosmology(0.3, 0.05, 0.67, 0.96, 0.81,
                                         tk_mode="EisHu", Pk_mode="Halofit")
            cosmo.settings.cosmo_type = typeof(p)
            z = range(0., stop=2., length=256)
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            ts = WeakLensingTracer(cosmo, z, nz;
                                   IA_params=[0.3, p])
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs)
            return Cℓ_ss
        end

        d = 0.00005
        b_autodiff = ForwardDiff.derivative(bias, 2.0)
        b_anal = (bias(2.0+d)-bias(2.0-d))/2d
        dz_autodiff = ForwardDiff.derivative(dz, -0.1)
        dz_anal = (dz(-0.1+d)-dz(-0.1-d))/2d
        mb_autodiff = ForwardDiff.derivative(mbias, 2.0)
        mb_anal = (mbias(2.0+d)-mbias(2.0-d))/2d
        IA_A_autodiff = ForwardDiff.derivative(IA_A, 0.3)
        IA_A_anal = (IA_A(0.3+d)-IA_A(0.3-d))/2d
        IA_alpha_autodiff = ForwardDiff.derivative(IA_alpha, 0.1)
        IA_alpha_anal = (IA_alpha(0.1+d)-IA_alpha(0.1-d))/2d

        @test all(@. (abs(b_autodiff/b_anal-1) < 1E-2))
        @test all(@. (abs(dz_autodiff/dz_anal-1) < 1E-2))
        @test all(@. (abs(mb_autodiff/mb_anal-1) < 1E-2))
        @test all(@. (abs(IA_A_autodiff/IA_A_anal-1) < 1E-2))
        @test all(@. (abs(IA_alpha_autodiff/IA_alpha_anal-1) < 1E-2))
    end

"""
end
