import numpy as np
import sacc
import yaml

def get_type(name):
    if ('DESwl' in name) or ('KiDS1000' in name):
        return 'e'
    elif "PLAcv" in name:
        return 'k'
    else:
        return '0'

s = sacc.Sacc().load_fits("FD/cls_FD_covG.fits")
fname = "FD/K1K_DELS_DESY1_eBOSS"
with open(fname+".yml") as f:
    config = yaml.safe_load(f)

# Apply scale cuts
indices = []
for cl in config['order']:
    t1, t2 = cl['tracers']
    lmin, lmax = cl['ell_cuts']
    cl_name = 'cl_%s%s' % (get_type(t1), get_type(t2))
    ind = s.indices(cl_name, (t1, t2),
                    ell__gt=lmin, ell__lt=lmax)
    indices += list(ind)
s.keep_indices(indices)

cls = []
ls = []
indices = []
pairs = []
for cl in config['order']:
    t1, t2 = cl['tracers']
    type1 = get_type(t1)
    type2 = get_type(t2)
    cl_name = 'cl_%s%s' % (type1, type2)
    l, c_ell, ind = s.get_ell_cl(cl_name, t1, t2,
                                 return_cov=False,
                                 return_ind=True)
    indices += list(ind)
    cls += list(c_ell)
    ls.append(l)
    pairs.append([t1+'_'+type1, t2+'_'+type2])

tracers = np.unique(pairs).flatten()
cov = s.covariance.dense[list(indices)][:, list(indices)]
w, v = np.linalg.eigh(cov)
cov = np.dot(v, np.dot(np.diag(np.fabs(w)), v.T))
cov = np.tril(cov) + np.triu(cov.T, 1)
inv_cov = np.linalg.inv(cov)

lengths = np.array([len(l) for l in ls])
lengths = np.concatenate([[0], lengths])
idx  = np.cumsum(lengths)

pairs_ids = []
for pair in pairs:
    t1, t2 = pair
    id1 = list(tracers).index(t1)
    id2 = list(tracers).index(t2)
    ids = [id1+1, id2+1]
    pairs_ids.append(ids)

indices = np.array(indices)
pairs = np.array(pairs)
pairs_ids = np.array(pairs_ids)
cls = np.array(cls)
idx = np.array(idx)
cov = np.array(cov)
inv_cov = np.array(inv_cov)

dict_save = {'tracers': tracers, 'pairs': pairs,
             'pairs_ids': pairs_ids, 'cls': cls, 'idx': idx,
             'cov': cov, 'inv_cov': inv_cov}

np.savez(fname+"_meta.npz", **dict_save)

###########

dict_save = {}

for pair, l in zip(pairs, ls):
    t1, t2 = pair
    dict_save[f'ls_{t1}_{t2}'] = np.array(l)

for name, tracer in s.tracers.items():
    if name != "PLAcv":
        z=np.array(tracer.z)
        dndz=np.array(tracer.nz)
        dict_save[f'nz_{name}'+'_'+get_type(name)]  = np.array([z, dndz]) 

np.savez(fname+"_files.npz", **dict_save)