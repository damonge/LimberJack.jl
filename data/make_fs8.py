import numpy as np
import os
from scipy.linalg import block_diag

def get_Wigglez(path):
    dataset_name = 'Wigglez.npz'
    z_Wigglez = np.array([0.44, 0.60, 0.73])
    fs8_Wigglez = np.array([0.413, 0.390, 0.437])
    Wigglez_cov = 10**(-3)*np.array([[6.4, 2.57, 0], 
                                    [2.57, 3.969, 2.54], 
                                    [0, 2.54, 5.184]])
    Wigglez_err = np.sqrt(np.diag(Wigglez_cov))
    np.savez(os.path.join(path, dataset_name),  
            data=fs8_Wigglez,
            z=z_Wigglez,
            cov=Wigglez_cov,
            err=Wigglez_err)
    return np.load(os.path.join(path, dataset_name))
    
def get_Vipers(path):
    dataset_name = 'Vipers.npz'
    z_Vipers = np.array([0.60, 0.86])
    fs8_Vipers = np.array([0.55, 0.40])
    Vipers_cov = np.array([[0.12**2, 0], 
                          [0, 0.11**2]])
    Vipers_err = np.sqrt(np.diag(Vipers_cov))
    np.savez(os.path.join(path, dataset_name),  
            data=fs8_Vipers,
            z=z_Vipers,
            cov=Vipers_cov,
            err=Vipers_err)
    return np.load(os.path.join(path, dataset_name)) 

def get_6dF(path):
    dataset_name = '6dF.npz'
    z_6dF = np.array([0.067])
    fs8_6dF = np.array([0.423])
    cov_6dF = np.array([[0.055**2]])
    err_6dF = np.array([0.055])
    np.savez(os.path.join(path, dataset_name),  
            data = fs8_6dF,
            z=z_6dF,
            cov=cov_6dF,
            err=err_6dF)
    return np.load(os.path.join(path, dataset_name))

def get_FastSound(path):
    dataset_name = 'FastSound.npz'
    z_FastSound = np.array([1.4])
    fs8_FastSound = np.array([0.482])
    cov_FastSound = np.array([[0.116**2]])
    err_FastSound = np.array([0.116])
    np.savez(os.path.join(path, dataset_name),  
            data = fs8_FastSound,
            z=z_FastSound,
            cov=cov_FastSound,
            err=err_FastSound)
    return np.load(os.path.join(path, dataset_name))

def get_DSS(path):
    dataset_name = 'DSS.npz'
    z_DSS = np.array([0])
    fs8_DSS = np.array([0.39])
    DSS_cov = np.array([[0.022**2]])
    DSS_err = np.array([0.022])
    np.savez(os.path.join(path, dataset_name),  
            data = fs8_DSS,
            z=z_DSS,
            cov=DSS_cov,
            err=DSS_err)
    return np.load(os.path.join(path, dataset_name))

def get_eBOSS(path):
    dataset_name = 'eBOSS.npz'
    z_eBOSS = np.array([1.48]) 
    fs8_eBOSS = np.array([0.462])
    eBOSS_cov = np.array([[0.045**2]])
    eBOSS_err = np.array([[0.045]])
    rd_eBOSS = 147.3
    np.savez(os.path.join(path, dataset_name),  
            data=fs8_eBOSS,
            z=z_eBOSS,
            cov=eBOSS_cov,
            err=eBOSS_err,
            rd=rd_eBOSS)
    return np.load(os.path.join(path, dataset_name))

def get_BOSS(path):
    dataset_name = 'BOSS.npz'
    z_BOSS = np.array([0.38, 0.51, 0.61])
    fs8_BOSS = np.array([0.49749, 0.457523, 0.436148])
    BOSS_cov = np.array([
           [2.03355e-03, 8.11829e-04, 2.64615e-04],
           [8.11829e-04, 1.42289e-03, 6.62824e-04],
           [2.64615e-04, 6.62824e-04, 1.18576e-03]])
    BOSS_err = np.sqrt(np.diag(BOSS_cov))
    rd_BOSS = 147.78
    np.savez(os.path.join(path, dataset_name),  
            data=fs8_BOSS,
            z=z_BOSS,
            cov=BOSS_cov,
            err=BOSS_err,
            rd=rd_BOSS)
    return np.load(os.path.join(path, dataset_name))

path = "fs8s"
BOSS = get_BOSS(path)
eBOSS = get_eBOSS(path)
Wigglez = get_Wigglez(path)
Vipers = get_Vipers(path)
sixdF = get_6dF(path)
FastSound = get_FastSound(path)

datadict = {'BOSS': BOSS,
            'eBOSS': eBOSS,
            'Wigglez': Wigglez,
            'Vipers': Vipers,
            '6dF': sixdF,
            'FastSound': FastSound}

zs = np.array([])
data = np.array([])
data_cov = np.array([])
for dataset_name in datadict.keys():
    print(datadict[dataset_name])
    dataset = datadict[dataset_name]
    zs = np.concatenate([zs, dataset['z']])
    data = np.concatenate([data, dataset['data']])
    data_cov = block_diag(data_cov, dataset['cov'])
data_cov = data_cov[1:]

np.savez(os.path.join(path, 'fs8s'),  
         data=data,
         z=zs,
         cov=data_cov,
         inv_cov=np.linalg.inv(data_cov))
