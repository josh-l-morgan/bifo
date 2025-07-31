# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 18:18:23 2025

@author: jlmorgan
"""

from scipy.io import loadmat
import h5py
import numpy as np
import pickle
import time
from types import SimpleNamespace
from bifo.tools import dtypes as dt


def file_convert_vastSubs_to_vast_subs(mat_path, mat_key, h5_path, group_name):
    with h5py.File(h5_path, 'w') as hf:
        h_group = hf.create_group(group_name)
        with h5py.File(mat_path,'r') as vf:
            vast_subs_refs = vf[mat_key][:]
            for i, ref in enumerate(vast_subs_refs):
                print(f'{i} of {len(vast_subs_refs)}')
                sub = np.array(vf[ref.item()],dtype='float32')
                if sub.shape[0] == 3:
                    sub = sub.swapaxes(0,1)
                h_group.create_dataset(f'{i}', data=sub)
            
            
def mat_struct_to_dict(obj):
    """Recursively convert MATLAB structs to nested Python dicts."""
    if isinstance(obj, np.ndarray):
        return [mat_struct_to_dict(o) for o in obj]
    elif hasattr(obj, '_fieldnames'):
        return {f: mat_struct_to_dict(getattr(obj, f)) for f in obj._fieldnames}
    else:
        return obj
    
    
def dict_to_namespace(d):
    """Recursively convert a dictionary (or struct-like object) to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, (list, np.ndarray)) and all(isinstance(i, dict) for i in d):
        # Convert list of dicts to list of namespaces
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def file_convert_obI_mat_to_pkl(obI_mat_path,obI_pkl_path):
    obI_scpy = loadmat(obI_mat_path, struct_as_record=False, squeeze_me=True) 
    obI_dict = dt.mat_struct_to_dict(obI_scpy['obI'])  # or obI.get('obI')   
    obI = dt.dict_to_namespace(obI_dict)
    with open(obI_pkl_path, 'wb') as f:
        pickle.dump(obI, f)       
   
def run_convert_vastSubs(mat_dir, py_dir):
    vastSubs_mat_path = mat_dir + 'vastSubs.mat'
    vast_subs_h5_path = py_dir + 'vast_subs.h5'
    group_name = 'subs'
    mat_key = 'vastSubs' 
    file_convert_vastSubs_to_vast_subs(vastSubs_mat_path, mat_key, vast_subs_h5_path, group_name)
   
    return vast_subs_h5_path

def run_convert_obI(mat_dir, py_dir):
    obI_mat_path = mat_dir + 'obI.mat'
    obI_pkl_path = py_dir + 'obI.pkl'
    file_convert_obI_mat_to_pkl(obI_mat_path,obI_pkl_path)
   
    return obI_pkl_path

def convert_Merge_mat_to_py(mat_dir, py_dir):
   
    vast_subs_h5_path = run_convert_vastSubs(mat_dir, py_dir)
    obI_pkl_path = run_convert_obI(mat_dir, py_dir) 
    
    paths = {
        'vast_subs': vast_subs_h5_path,
        'obI': obI_pkl_path
    }
    
    return paths
    









if __name__ == '__main__':

    
    mat_dir = '//storage1.ris.wustl.edu/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/CellNav_KxR/Volumes/v1/Merge/'  
    py_dir = 'G:/TrainingSegmentations/KxR_vast_subs/'
    
     
    if 1: 
        paths = convert_Merge_mat_to_py(mat_dir, py_dir)  
    
    if 0:
        vast_subs_h5_path = run_convert_vastSubs(mat_dir, py_dir)
        
    if 0:
        obI_pkl_path = run_convert_obI(mat_dir, py_dir) 
        with open(obI_pkl_path, 'rb') as f:
            obI = pickle.load(f)
        dt.explore_namespace(obI,3)    
         
                
             
    ## file open time test
    if 0:
        t = np.zeros((3,2))
        
        t[0,0] = time.time()
        for i in range(5753):
            with h5py.File(paths['vast_subs'], 'r') as hf:
                a = hf['subs'][f'{i}'][()]
                #print(a.shape)
        t[0,1] = time.time()
            
            
        t[1,0] = time.time()
        with h5py.File(paths['vast_subs'], 'r') as hf:
            vast_subs = [hf['subs'][key][()] for key in hf['subs']]               
        t[1,1] = time.time()
    
        
        td = t[:,1] - t[:,0]
        print(f'single {td[0]}, full = {td[1]}')
            
        
    
    

    
    
    
    
    
    
    
    