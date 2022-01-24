import pyccl as ccl
import numpy as np
cosmo = ccl.Cosmology(Omega_c=0.26+0.05, Omega_b=0.05,
                      h=0.67, n_s=0.96, sigma8=0.811,
                      transfer_function='eisenstein_hu',
                      matter_power_spectrum='halofit', Omega_k=0)
z_bg_test = np.array([0.1, 0.5, 1.0, 3.0])
print("Dist: ", ccl.comoving_radial_distance(cosmo, 1./(1+z_bg_test)))
print("Growth: ", ccl.growth_factor(cosmo, 1./(1+z_bg_test)))
ks_test = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
print("Pk: ", ccl.nonlin_matter_power(cosmo, ks_test, 1.))
print("lin_Pk: ", ccl.linear_matter_power(cosmo, ks_test, 1.))
z = np.linspace(0., 3., 256)
wz = np.exp(-0.5*((z-0.5)/0.05)**2)
tg = ccl.NumberCountsTracer(cosmo, False, dndz=(z, wz), bias=(z, 2.*np.ones_like(z)))
ts = ccl.WeakLensingTracer(cosmo, dndz=(z, wz))
tk = ccl.CMBLensingTracer(cosmo, z_source=1100)
ls_test = np.array([10, 30, 100, 300])
cls_gg = ccl.angular_cl(cosmo, tg, tg, ls_test)
cls_gs = ccl.angular_cl(cosmo, tg, ts, ls_test)
cls_ss = ccl.angular_cl(cosmo, ts, ts, ls_test)
cls_gk = ccl.angular_cl(cosmo, tg, tk, ls_test)
cls_sk = ccl.angular_cl(cosmo, ts, tk, ls_test)
print("Cls_gg: ", cls_gg)
print("Cls_gs: ", cls_gs)
print("Cls_ss: ", cls_ss)
print("Cls_gk: ", cls_gk)
print("Cls_sk: ", cls_sk)
