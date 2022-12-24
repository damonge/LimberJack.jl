import numpy as np
import sacc
import yaml

def apply_scale_cuts(s, config):
    indices = []
    for cl in config['order']:
        t1, t2 = cl['tracers']
        lmin, lmax = cl['ell_cuts']
        spin1 = get_spin(s, t1)
        spin2 = get_spin(s, t2)
        cl_name = 'cl_%s%s' % (spin1, spin2)
        if cl_name == 'cl_e0':
            cl_name = 'cl_0e'
        ind = s.indices(cl_name, (t1, t2),
                        ell__gt=lmin, ell__lt=lmax)
        indices += list(ind)
    s.keep_indices(indices)
    return s

def get_type(sacc_file, tracer_name):
    return sacc_file.tracers[tracer_name].quantity

def get_spin(sacc_file, tracer_name):
    tt = sacc_file.tracers[tracer_name].quantity
    if tt == "galaxy_shear":
        spin = 'e'
    elif tt == "galaxy_density":
        spin = "0"
    elif tt == "cmb_convergence":
        spin = "0"
    return spin

sacc_path = "FD/cls_FD_covG.fits"
yaml_path = "DESY1/gcgc_gcwl_wlwl"
nzs_path = None
fname = "DESY1/gcgc_gcwl_wlwl"

s = sacc.Sacc().load_fits(sacc_path)
with open(yaml_path+".yml") as f:
    config = yaml.safe_load(f)

# Apply scale cuts
s = apply_scale_cuts(s, config)

cls = []
ls = []
indices = []
pairs = []
for cl in config['order']:
    t1, t2 = cl['tracers']
    spin1 = get_spin(s, t1)
    spin2 = get_spin(s, t2)
    cl_name = 'cl_%s%s' % (spin1, spin2)
    if cl_name == 'cl_e0':
        cl_name = 'cl_0e'
    l, c_ell, ind = s.get_ell_cl(cl_name, t1, t2,
                                 return_cov=False,
                                 return_ind=True)
    indices += list(ind)
    cls += list(c_ell)
    ls.append(l)
    pairs.append([t1, t2])

names = np.unique(pairs).flatten()
cov = s.covariance.dense[list(indices)][:, list(indices)]
w, v = np.linalg.eigh(cov)
cov = np.dot(v, np.dot(np.diag(np.fabs(w)), v.T))
cov = np.tril(cov) + np.triu(cov.T, 1)
inv_cov = np.linalg.inv(cov)

lengths = np.array([len(l) for l in ls])
lengths = np.concatenate([[0], lengths])
idx  = np.cumsum(lengths)

types = [get_type(s, name) for name in names]

types = np.array(types)
indices = np.array(indices)
pairs = np.array(pairs)
cls = np.array(cls)
idx = np.array(idx)
cov = np.array(cov)
inv_cov = np.array(inv_cov)

dict_save = {'names': names, 'pairs': pairs,
             'types': types, 'cls': cls, 'idx': idx,
             'cov': cov, 'inv_cov': inv_cov}

np.savez(fname+"_meta.npz", **dict_save)

###########

dict_save = {}

for pair, l in zip(pairs, ls):
    t1, t2 = pair
    print(t1, t2, len(l))
    dict_save[f'ls_{t1}_{t2}'] = np.array(l)

for name, tracer in s.tracers.items():
    if name in names:
        if nzs_path is None:
            if tracer.quantity != "cmb_convergence":
                z=np.array(tracer.z)
                dndz=np.array(tracer.nz)
                dict_save[f'nz_{name}'] = np.array([z, dndz])
        else:
            nzs = np.load(nzs_path+f'nz_{name}.npz')
            z = nzs["z"]
            dndz = nzs["dndz"]
            dict_save[f'nz_{name}'] = np.array([z, dndz])

np.savez(fname+"_files.npz", **dict_save)
