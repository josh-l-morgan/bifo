# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:18:41 2025

@author: jlmorgan
"""
from scipy.io import loadmat
import h5py
import numpy as np
import pickle
import time
import zarr
import os
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm


from bifo.tools import dtypes
from bifo import zarrTools
from bifo.vast import readMergeMats
from bifo.tools import dtypes
from bifo.tools import display as dsp

# import importlib
# importlib.reload(zarrTools)


# def isIn3(sub, lower, upper = None):
#     if upper is None:
#         upper = lower
#     # upper inclusive
#     is_in =  (sub[:,0]>=lower[0]) & (sub[:,0] <= (upper[0])) & \
#                 (sub[:,1]>=lower[1]) & (sub[:,1] <= (upper[1])) & \
#                 (sub[:,2]>=lower[2]) & (sub[:,2] <= (upper[2]))
#     return is_in



# def list_of_subs_to_zarr(vast_subs, zrm, block_shape=[128, 128, 128], sub_to_zar_dim = (2,0,1)):
    
#     num_o = len(vast_subs)

    
#     # get bounding boxes
#     print('finding bounding boxes')
#     bbox = np.zeros((num_o,3,2),int)
#     v_has_dat = np.zeros(num_o,'bool')
#     for i,s in enumerate(vast_subs):
#         bbox[i,:,0] = s.min(0)
#         bbox[i,:,1] = s.max(0)
#         if len(s.shape) >1:
#             v_has_dat[i] = True
            
#     bbox = bbox[:,sub_to_zar_dim,:]        
#     good_o = np.where(v_has_dat)[0]
#     bad_o = np.where(~v_has_dat) [0]       
        
    
#     vol_box = np.stack((bbox[good_o,:,0].min(0),bbox[good_o,:,1].max(0)),0)
    
#     # calculate blocks to run
#     check_s = []
#     for d in range(3):
#         check_s.append(np.arange(0,vol_box[1,d],block_shape[d]))
    
#     # pre scan blocks
#     b_info = []
#     for s0 in check_s[0]:
#         for s1 in check_s[1]:
#             for s2 in check_s[2]:
#                 lower = np.array([s0, s1, s2])
#                 upper = np.array([s0, s1, s2]) + block_shape - 1
#                 in_box_min = isIn3(bbox[:,:,0], lower, upper)
#                 in_box_max = isIn3(bbox[:,:,1], lower, upper)
#                 in_box_list = np.where(in_box_min + in_box_max)[0]
#                 in_box_list = np.setdiff1d(in_box_list,bad_o) #remove empty
#                 if in_box_list.shape[0]:
#                     b_info.append({'blk':np.array((s0,s1,s2),int),
#                                    'vs': in_box_list})
    
#     print('writing to zarr')
#     tk = dtypes.ticTocDic() # get timer
#     tk.m(['block', 'check is_in','sub_to_block','write']) #initialize timer dictionary
#     block = np.zeros(block_shape,'int32')
#     for i,bi in enumerate(b_info):
#         tk.b('block')
#         block = block * 0
#         lower = bi['blk']
#         upper = lower + block_shape - 1
#         for iv,v in enumerate(bi['vs']):
#             tk.b('check is_in')
#             is_in = np.where(isIn3(vast_subs[v][:,sub_to_zar_dim], lower, upper))[0]
#             tk.e('check is_in')
#             if is_in.shape[0]:
#                 tk.b('sub_to_block')
#                 sub = vast_subs[v][is_in,:]
#                 sub = sub[:,sub_to_zar_dim] - bi['blk']
#                 block[sub[:,0].astype(int),sub[:,1].astype(int), sub[:,2].astype(int)] = v
#                 tk.e('sub_to_block')
        
#         tk.b('write')
#         b_clip =  np.min((zrm.shape-lower,block_shape),0) 
#         z_clip = lower + b_clip -1
#         zrm[lower[0]:z_clip[0]+1, lower[1]:z_clip[1]+1, lower[2]:z_clip[2]+1] = \
#             block[0:b_clip[0],0:b_clip[1],0:b_clip[2]]
#         tk.e('write')
          
#         tk.e('block')
#         print(f'block {i} of {len(b_info)}.')
#         tk.pt()        
                    
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
       
            
       
    
   
    
   
    
   
    
   
    
   
    
    
    
        
        
    
    