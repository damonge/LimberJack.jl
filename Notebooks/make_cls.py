import pyccl as ccl
import numpy as np

cosmo1 = ccl.CosmologyVanillaLCDM(transfer_function='bbks', matter_power_spectrum='linear', Omega_g=0, Omega_k=0)
cosmo2 = ccl.CosmologyVanillaLCDM(transfer_function='eisenstein_hu', matter_power_spectrum='linear', Omega_g=0, Omega_k=0)
z_bg_test = np.array([0.1, 0.5, 1.0, 3.0])
print("Dist: ", ccl.comoving_radial_distance(cosmo1, 1./(1+z_bg_test)))
print("Growth: ", ccl.growth_factor(cosmo1, 1./(1+z_bg_test)))
ks_test = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
print("Pk0-1: ", ccl.linear_matter_power(cosmo1, ks_test, 1.))
print("Pk0-2: ", ccl.linear_matter_power(cosmo2, ks_test, 1.))

z = np.linspace(0., 2., 1024)
wz = np.exp(-0.5*((z-0.5)/0.05)**2)
t = ccl.NumberCountsTracer(cosmo1, False, dndz=(z, wz), bias=(z, 2.*np.ones_like(z)))
ls_test = np.array([10, 30, 100, 300])
cls = ccl.angular_cl(cosmo1, t, t, ls_test)
print("Cls: ", cls)

z = np.linspace(0., 2., 1024)
wz = np.exp(-0.5*((z-0.5)/0.05)**2)
t = ccl.NumberCountsTracer(cosmo2, False, dndz=(z, wz), bias=(z, 2.*np.ones_like(z)))
ls_test = np.array([10, 30, 100, 300])
cls = ccl.angular_cl(cosmo2, t, t, ls_test)
print("Cls: ", cls)