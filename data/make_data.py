import numpy as np
import sacc
import yaml

def get_type(name):
    if 'wl' in name:
        return 'e'
    else:
        return '0'

s = sacc.Sacc().load_fits("DESY1_cls/cls_covG_new.fits")
fname = "DESY1_cls/wlwl"
with open(fname+".yml") as f:
    config = yaml.safe_load(f)

#Apply scale cuts 
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
    cl_name = 'cl_%s%s' % (get_type(t1), get_type(t2))
    l, c_ell, ind = s.get_ell_cl(cl_name, t1, t2,
                                 return_cov=False,
                                 return_ind=True)
    indices += list(ind)
    cls += list(c_ell)
    ls.append(l)
    pairs.append([t1, t2])

tracer_code = {'DESgc': 1, 'DESwl': 2}
tracers = []
for tracer in np.unique(pairs).flatten():
    t, n = tracer_code[tracer[:5]], int(tracer[-1])
    tracers.append([t, n])

indices = np.array(indices)
cls = np.array(cls)
print(len(indices))
cov = s.covariance.dense[indices][:, indices]
w, v = np.linalg.eigh(cov)
cov = np.dot(v, np.dot(np.diag(np.fabs(w)), v.T))
cov = np.tril(cov) + np.triu(cov.T, 1)
inv_cov = np.linalg.inv(cov)

pairs_id = []
pairs_coded = []
for pair, l in zip(pairs, ls):
    t1, t2 = pair
    typ1 = tracer_code[t1[:5]]
    bin1 = int(t1[-1])
    typ2 = tracer_code[t2[:5]]
    bin2 = int(t2[-1])
    id1 = tracers.index([typ1, bin1])
    id2 = tracers.index([typ2, bin2])
    ids = [id1+1, id2+1]
    pair = [typ1, bin1, typ2, bin2]
    pairs_id.append(ids)
    pairs_coded.append(pair)

lengths = [len(l) for l in ls]
lengths = np.concatenate([[0], lengths])
idx  = np.cumsum(lengths)

dict_save = {'tracers': tracers, 'pairs': pairs_coded, 'pairs_ids': pairs_id,
             'cls': cls, 'cov': cov, 'inv_cov': inv_cov, 'idx': idx}

for pair, l in zip(pairs, ls):
    t1, t2 = pair
    bin1 = int(t1[-1])
    t1 = tracer_code[t1[:5]]
    t1 = int(f'{t1}{bin1}')
    bin2 = int(t2[-1])
    t2 = tracer_code[t2[:5]]
    t2 = int(f'{t2}{bin2}')
    dict_save[f'ls_{t1}{t2}'] = l

np.savez(fname+".npz", **dict_save)
