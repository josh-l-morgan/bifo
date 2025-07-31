# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:18:41 2025
@author: jlmorgan
Save Vast-matlab datastructures (vast_subs) to zarr
"""
import h5py
import numpy as np
import pickle
import zarr
import os
import matplotlib.pyplot as plt

from bifo.tools import dtypes
from bifo import zarrTools
from bifo.vast import readMergeMats
from bifo.tools import display as dsp

plt.ion()
en = dtypes.explore_namespace # get namespace explorer

merge_py_dir = 'G:/TrainingSegmentations/KxR_vast_subs/'
z_shape = [1400, 72000, 50400] 
use_mip = 4
sub_to_zar_dim = (2,0,1)
chunk_shape = [64, 256, 256]

vast_subs_path = merge_py_dir + 'vast_subs.h5'
obI_pkl_path = merge_py_dir + 'obI.pkl'

if not os.path.isfile(vast_subs_path): #or choose to update
    print('converting mat files to h5 and pkl')
    mat_dir = 'G:/KxR_cellNav/Volumes/v1/Merge/'  
    paths = readMergeMats.convert_Merge_mat_to_py(mat_dir, merge_py_dir)  

with open(obI_pkl_path, 'rb') as f:
        obI = pickle.load(f)   

# Make zarr
print('Initializing target zarr')
z_path = merge_py_dir + 'cn.zarr'
z_group = 'vast_subs'
zarrTools.makeMultiscaleGroup(z_path=z_path, group_name=z_group, zarr_shape=z_shape, 
                              z_chunk_shape = chunk_shape)
zarr_root = zarr.open_group(z_path, mode='a')
zrm = zarr_root[z_group][f'{use_mip}'] 

# write vast_subs to zarr
if 0:
    with h5py.File(vast_subs_path, 'r') as hf:
        vast_subs = [hf['subs'][key][()] for key in hf['subs']]  
    zarrTools.list_of_subs_to_zarr(vast_subs, zrm, sub_to_zar_dim = (2,0,1))

## show
fig = dsp.figure()   
num_col = 10000
cmap = dsp.col_map('rand',num_col)

if 0:
    pw = np.array([[0,1400],[0,3000],[0,1500]]) 
    w_shape = np.diff(pw,1).flatten() + 1
    max_i = zarrTools.zarr_max(zrm, dim=0, p_window=pw)

else:
    pw = None
    w_shape = zrm.shape
    max_i = zarrTools.zarr_max(zrm, dim=0, p_window=pw)

fig.make_img(i_shape=w_shape[1:], cmap=cmap, vmax=num_col)
fig.ax[0].img.set_data(max_i)
fig.update()
  
    
        
        
        
        