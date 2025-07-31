# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 19:45:37 2025

@author: jlmorgan

Confirm that contents of zarr file are consistent with vast_subs

"""
from bifo.fetchChunckFromMips import fetchDiced
import numpy as np
import zarr
import bifo.tools.voxTools as vt
import bifo.tools.display as dsp
import time

def vast_to_map_request(map_request,
        vast_location=(15163, 15397, 373),
        check_size=[4, 256, 256]
        ):
   
    fov_center = np.array(vast_location) 
    fov_center = fov_center[[2,1,0]]
    fov_corner_1 = fov_center - (np.array(check_size) * map_request['mip_level'] / 2) # find lower corner

    map_request["volume_lower_corner_zyx"] = fov_corner_1.astype(int)
    map_request['volume_size_zyx'] = check_size
    
    return map_request


merge_py_dir = 'G:/TrainingSegmentations/KxR_vast_subs/'

# Open vast segmentation zar
z_path = merge_py_dir + 'cn.zarr'
zarr_seg_root = zarr.open_group(z_path, mode='a')
zrm_seg = zarr_seg_root['vast_subs']['4'] 



# set paths
paths = {        
    'diced_dir':"F:/KxR_P11_LGN/mips/",
    'cp_path': "G:/Checkpoints/UnetPP_256_01/checkpoints/model_ns12_256by256_rgcDet_round1_itr_10000.pth",
    'results_dir': "G:/Results/pred/",
    'z_name': 'KxR_ms.zarr',
    'z_group': 'pred'
}
   

map_request = {
    "diced_dir": paths['diced_dir'],
    "mip_level": 1,
    "analyze_section": [400], 
    "volume_lower_corner_zyx": [0,0,0],
    "volume_size_zyx": [4, 256, 256], # size of volume at mip level to be processed
    "chunk_shape_zyx": [16, 128, 128], # size to be fed to model
    "chunk_overlap": [0, 0, 0],
    'full_data_shape': [1400, 72000, 50400]
    }

map_request = fetchDiced.vast_to_map_request(map_request, 
                                  vast_location=(30134, 21565, 360),
                                  check_size = [4, 1024, 1024])

rw = np.array([map_request["volume_lower_corner_zyx"],
    map_request["volume_lower_corner_zyx"] + map_request['volume_size_zyx'] - 1])
pw = rw.copy()
pw[:,1:] = pw[:,1:] / (2 ** 1)
vol_shape = pw[1,:] - pw[0,:]
vol = np.zeros(vol_shape)

fd = fetchDiced(map_request)

        
vol = fd.full_vol()
       

sw = rw.copy()
sw[:,1:] = sw[:,1:] / (2 **4)
vol_s = zrm_seg[sw[0,0]:sw[1,0], sw[0,1]:sw[1,1], sw[0,2]:sw[1,2]] 

        
f1 = dsp.figure()
f1.fig.clear()
f1.ax[0] = f1.fig.add_subplot(1,2,1)
f1.ax.append(f1.fig.add_subplot(1,2,2))
num_col = 10000
cmap = dsp.col_map('rand',num_col)

f1.make_img(ax_id=0, i_shape=vol.shape[1:])
f1.make_img(ax_id=1, i_shape=vol_s.shape[1:],cmap=cmap, vmax=num_col)

for z in range(vol.shape[0]):
    f1.ax[0].img.set_data(vol[z,:,:])
    f1.ax[1].img.set_data(vol_s[z,:,:])
    f1.update()
    time.sleep(1)




# ## show
# fig = dsp.figure()   
# num_col = 10000
# cmap = dsp.col_map('rand',num_col)

# if 0:
#     pw = np.array([[0,1400],[0,3000],[0,1500]]) 
#     w_shape = np.diff(pw,1).flatten() + 1
#     max_i = zarrTools.zarr_max(zrm, dim=0, p_window=pw)

# else:
#     pw = None
#     w_shape = zrm.shape
#     max_i = zarrTools.zarr_max(zrm, dim=0, p_window=pw)

# fig.make_img(i_shape=w_shape[1:], cmap=cmap, vmax=num_col)
# fig.ax[0].img.set_data(max_i)
# fig.update()




# map_request = vast_to_map_request(map_request_default,
#                                vast_location=(15163, 15397, 373),
#                                check_size=[4, 256, 256]
#                                )

