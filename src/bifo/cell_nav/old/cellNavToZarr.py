# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:18:41 2025

@author: jlmorgan
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

                    
if __name__ == '__main__':
        
    en = dtypes.explore_namespace # get namespace explorer
    
    merge_py_dir = 'G:/TrainingSegmentations/KxR_vast_subs/'
    z_shape = [1400, 72000, 50400] 
    use_mip = 4
    sub_to_zar_dim = (2,0,1)
    block_shape = [128, 128, 128]
    
    vast_subs_path = merge_py_dir + 'vast_subs.h5'
    obI_pkl_path = merge_py_dir + 'obI.pkl'
    
    
    if not os.path.isfile(vast_subs_path): #or choose to update
        print('converting mat files to h5 and pkl')
        mat_dir = 'G:/KxR_cellNav/Volumes/v1/Merge/'  
        paths = readMergeMats.convert_Merge_mat_to_py(mat_dir, merge_py_dir)  
    
    with h5py.File(vast_subs_path, 'r') as hf:
        vast_subs = [hf['subs'][key][()] for key in hf['subs']]  
    
    with open(obI_pkl_path, 'rb') as f:
        obI = pickle.load(f)
     
    
    # Make zarr
    print('Initializing target zarr')
    z_path = merge_py_dir + 'cn.zarr'
    z_group = 'vast_subs'
    zarrTools.makeMultiscaleGroup(z_path=z_path, group_name=z_group, zarr_shape=z_shape)
    zarr_root = zarr.open_group(z_path, mode='a')
    zrm = zarr_root[z_group][f'{use_mip}'] 
    
    
    zarrTools.list_of_subs_to_zarr(vast_subs, zrm, block_shape=[128, 128, 128], sub_to_zar_dim = (2,0,1))
    
    
    ## show
    plt.ion()
    
    fig = dsp.figure()
    samp_y = np.array([200, 800])
    samp_x = np.array([800, 1600])
    i_shape = np.diff([samp_y, samp_x], axis=1)
    
    num_col = 10000
    cmap = dsp.col_map('rand',num_col)
    fig.make_img(i_shape=i_shape, cmap=cmap, vmax=num_col)
    
    for z in range(400,500):
        
        print(z)
        z_samp = zrm[z,samp_y[0]: samp_y[1],samp_x[0]: samp_x[1]]
        fig.ax[0].img.set_data(z_samp)
        fig.update()
       
            
       
    
   
    
   
    
   
    
   
    
   
    
    
    
        
        
    
    