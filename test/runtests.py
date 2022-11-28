import numpy as np
import pyccl as ccl

test_results = {}

cosmo_bbks = ccl.CosmologyVanillaLCDM(transfer_function="bbks", 
                                    matter_power_spectrum="linear",
                                    Omega_g=0, Omega_k=0)
cosmo_eishu = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu", 
                                    matter_power_spectrum="linear",
                                    Omega_g=0, Omega_k=0)
cosmo_camb = ccl.Cosmology(Omega_c=0.255, Omega_b=0.045, h=0.67, n_s=0.96, sigma8=0.81,
                           Omega_g=0, Omega_k=0,
                           transfer_function="boltzmann_camb",
                           matter_power_spectrum="linear")

cosmo_bbks_nonlin = ccl.CosmologyVanillaLCDM(transfer_function="bbks",
                                              matter_power_spectrum="halofit",
                                              Omega_g=0, Omega_k=0)
cosmo_eishu_nonlin = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu",
                                              matter_power_spectrum="halofit",
                                              Omega_g=0, Omega_k=0)
cosmo_camb_nonlin = ccl.Cosmology(Omega_c=0.255, Omega_b=0.045, h=0.67, n_s=0.96, sigma8=0.81,
                                  Omega_g=0, Omega_k=0,
                                  transfer_function="boltzmann_camb",
                                  matter_power_spectrum="halofit")
# =====
ztest = np.array([0.1, 0.5, 1.0, 3.0])
test_results["Chi"] = ccl.comoving_radial_distance(cosmo_bbks,  1 / (1 + ztest))
# =====
ztest = np.array([0.1, 0.5, 1.0, 3.0])
test_results["Dz"] = ccl.growth_factor(cosmo_bbks, 1 / (1 + ztest))
test_results["fz"] = ccl.growth_rate(cosmo_bbks, 1 / (1 + ztest))
# =====
ks = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
test_results["pk_BBKS"] = ccl.linear_matter_power(cosmo_bbks, ks, 1.)
test_results["pk_EisHu"] = ccl.linear_matter_power(cosmo_eishu, ks, 1.)
test_results["pk_emul"] = ccl.linear_matter_power(cosmo_camb, ks, 1.)
# =====
lks = np.linspace(-3, 2, 20)
ks = np.exp(lks)
test_results["pk_BBKS_nonlin"] = ccl.nonlin_matter_power(cosmo_bbks_nonlin, ks, 1.)
test_results["pk_EisHu_nonlin"] = ccl.nonlin_matter_power(cosmo_eishu_nonlin, ks, 1.)
test_results["pk_emul_nonlin"] = ccl.nonlin_matter_power(cosmo_camb_nonlin, ks, 1.)
# =====
z = np.linspace(0., 2., num=256)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
ℓs = np.array([10.0, 30.0, 100.0, 300.0])
tg = ccl.NumberCountsTracer(cosmo_eishu, False, dndz=(z, nz), bias=(z, 1 * np.ones_like(z)))
ts = ccl.WeakLensingTracer(cosmo_eishu, dndz=(z, nz))
tk = ccl.CMBLensingTracer(cosmo_eishu, z_source=1100)
test_results["cl_gg_eishu"] = ccl.angular_cl(cosmo_eishu, tg, tg, ℓs)
test_results["cl_gs_eishu"] = ccl.angular_cl(cosmo_eishu, tg, ts, ℓs)
test_results["cl_ss_eishu"] = ccl.angular_cl(cosmo_eishu, ts, ts, ℓs)
test_results["cl_gk_eishu"] = ccl.angular_cl(cosmo_eishu, tg, tk, ℓs)
test_results["cl_sk_eishu"] = ccl.angular_cl(cosmo_eishu, ts, tk, ℓs)
# =====
z = np.linspace(0., 2., num=256)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
ℓs = np.array([10.0, 30.0, 100.0, 300.0])
tg = ccl.NumberCountsTracer(cosmo_camb, False, dndz=(z, nz), bias=(z, 1 * np.ones_like(z)))
ts = ccl.WeakLensingTracer(cosmo_camb, dndz=(z, nz))
tk = ccl.CMBLensingTracer(cosmo_camb, z_source=1100)
test_results["cl_gg_camb"] = ccl.angular_cl(cosmo_camb, tg, tg, ℓs)
test_results["cl_gs_camb"] = ccl.angular_cl(cosmo_camb, tg, ts, ℓs)
test_results["cl_ss_camb"] = ccl.angular_cl(cosmo_camb, ts, ts, ℓs)
test_results["cl_gk_camb"] = ccl.angular_cl(cosmo_camb, tg, tk, ℓs)
test_results["cl_sk_camb"] = ccl.angular_cl(cosmo_camb, ts, tk, ℓs)
# =====
z = np.linspace(0., 2., num=256)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
ℓs = np.array([10.0, 30.0, 100.0, 300.0])
tg = ccl.NumberCountsTracer(cosmo_eishu_nonlin, False, dndz=(z, nz), bias=(z, 1 * np.ones_like(z)))
ts = ccl.WeakLensingTracer(cosmo_eishu_nonlin, dndz=(z, nz))
tk = ccl.CMBLensingTracer(cosmo_eishu_nonlin, z_source=1100)
test_results["cl_gg_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tg, tg, ℓs)
test_results["cl_gs_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tg, ts, ℓs)
test_results["cl_ss_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, ts, ts, ℓs)
test_results["cl_gk_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tg, tk, ℓs)
test_results["cl_sk_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, ts, tk, ℓs)
# =====
z = np.linspace(0.01, 2., num=1024)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
ℓs = np.array([10.0, 30.0, 100.0, 300.0])
tg = ccl.NumberCountsTracer(cosmo_camb_nonlin, False, dndz=(z, nz), bias=(z, 1 * np.ones_like(z)))
ts = ccl.WeakLensingTracer(cosmo_camb_nonlin, dndz=(z, nz))
tk = ccl.CMBLensingTracer(cosmo_camb_nonlin, z_source=1100)
test_results["cl_gg_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, tg, tg, ℓs)
test_results["cl_gs_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, tg, ts, ℓs)
test_results["cl_ss_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, ts, ts, ℓs)
test_results["cl_gk_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, tg, tk, ℓs)
test_results["cl_sk_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, ts, tk, ℓs)
# =====
z = np.linspace(0.01, 2., num=256)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
ℓs = np.array([10.0, 30.0, 100.0, 300.0])
Dz = ccl.comoving_radial_distance(cosmo_bbks,  1 / (1 + z))
IA_corr = (0.1*((1 + z)/1.62)**0.1 * (0.0134*0.3/Dz))
tg_b = ccl.NumberCountsTracer(cosmo_eishu_nonlin, False, dndz=(z, nz), bias=(z, 2 * np.ones_like(z)))
ts_m = ccl.WeakLensingTracer(cosmo_eishu_nonlin, dndz=(z, nz))
ts_IA = ccl.WeakLensingTracer(cosmo_eishu_nonlin, dndz=(z, nz), ia_bias=(z, IA_corr))
test_results["cl_gg_b"] = ccl.angular_cl(cosmo_eishu_nonlin, tg_b, tg_b, ℓs)
test_results["cl_ss_m"] = (1.0 + 1.0)**2  * ccl.angular_cl(cosmo_eishu_nonlin, ts_m, ts_m, ℓs)
test_results["cl_ss_IA"] = ccl.angular_cl(cosmo_eishu_nonlin, ts_IA, ts_IA, ℓs)

np.savez("test_results.npz", **test_results)