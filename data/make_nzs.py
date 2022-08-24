import numpy as np
import sacc
import yaml
    
s = sacc.Sacc().load_fits("DESY1_cls/cls_covG_new.fits")
tracer_code = {'DESgc': 1, 'DESwl': 2}
for n, t in s.tracers.items():
    if n.startswith('DES'):
        tracer = tracer_code[n[:5]]
        binn = n[-1]
        np.savez(f'nz_{tracer}{binn}'+'.npz',
                 z=np.array(t.z),
                 dndz=np.array(t.nz))
        