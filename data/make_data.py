import numpy as np
import sacc
import yaml

def get_type(name, mode="write"):
    if ('DESwl' in name) or ('KiDS1000' in name):
        return 'e'
    elif "PLAcv" in name:
        if mode=="write":
            return 'k'
        if mode=="read":
            return "0"
    else:
        return '0'

s = sacc.Sacc().load_fits("DESY1/cls_covG_new.fits")
fname = "DESY1/wlwl"
with open(fname+".yml") as f:
    config = yaml.safe_load(f)

# Apply scale cuts
indices = []
for cl in config['order']:
    t1, t2 = cl['tracers']
    lmin, lmax = cl['ell_cuts']
    type1 = get_type(t1, mode="read")
    type2 = get_type(t2, mode="read")
    cl_name = 'cl_%s%s' % (type1, type2)
    if cl_name == 'cl_e0':
        cl_name = 'cl_0e'
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
    type1 = get_type(t1, mode="read")
    type2 = get_type(t2, mode="read")
    cl_name = 'cl_%s%s' % (type1, type2)
    if cl_name == 'cl_e0':
        cl_name = 'cl_0e'
    l, c_ell, ind = s.get_ell_cl(cl_name, t1, t2,
                                 return_cov=False,
                                 return_ind=True)
    indices += list(ind)
    cls += list(c_ell)
    ls.append(l)
    type1 = get_type(t1, mode="write")
    type2 = get_type(t2, mode="write")
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

np.savez(fname+"binned20_meta.npz", **dict_save)

###########

dict_save = {}

for pair, l in zip(pairs, ls):
    t1, t2 = pair
    print(t1, t2, len(l))
    dict_save[f'ls_{t1}_{t2}'] = np.array(l)

nzs_path = "DESY1/binned_20_nzs/"
for name, tracer in s.tracers.items():
    name = name+'_'+get_type(name, mode="write")
    if name in tracers:
        if nzs_path is None:       
            z=np.array(tracer.z)
            dndz=np.array(tracer.nz)
            dict_save[f'nz_{name}'] = np.array([z, dndz])
        else:
            nzs = np.load(nzs_path+f'nz_{name}.npz')
            z = nzs["z"]
            dndz = nzs["dndz"]
            dict_save[f'nz_{name}'] = np.array([z, dndz])

np.savez(fname+"binned20_files.npz", **dict_save)
