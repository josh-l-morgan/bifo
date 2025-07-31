# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 13:49:42 2025

@author: jlmorgan
"""
import os
import bifo.zarrTools as zt
from PIL import Image
import numpy as np
import zarr
import bifo.tools.display as dsp
import matplotlib.pyplot as plt
import bifo.tools.voxTools as vt
import bifo.zarrTools as zt

plt.ion()

source_path =  "//storage1.ris.wustl.edu/jlmorgan/Active/morganLab/DATA/LGN_Adult/LGNs1/tweakedImageVolume/LGNs1_P32_smallHighres/"
em_source_path = '//storage1.ris.wustl.edu/jlmorgan/Active/morganLab/DATA/LGN_Adult/LGNs1/tweakedImageVolume/em_mip0_export/'
z_path = '//storage1.ris.wustl.edu/jlmorgan/Active/morganLab/DATA/LGN_Adult/LGNs1/tweakedImageVolume/linda.zarr/'
em_shape = [176, 2116, 2360]
z_chunk_shape = [32, 256, 256]
dsamp= [1,2,2]

# EM
zt.makeMultiscaleGroup(z_path=z_path, group_name='em', zarr_shape=em_shape,
                        z_chunk_shape = z_chunk_shape, num_scales=4, dsamp=dsamp)
zarr_root = zarr.open_group(z_path, mode='a')
zrm_em = zarr_root['em']['0'] 

if 0:
    file_names = [d.name for d in os.scandir(em_source_path)]
    em_pat = 'tweakedImageVolume2_export_s'
    em_files = [f for f in file_names if f.startswith(em_pat)]
    zs = np.zeros(len(em_files))
    for i, f in enumerate(em_files):
        d_string = f.split('_s')[1].split('.png')[0]
        zs[i] = int(d_string)   
    z_idx = np.argsort(zs)
    zs = zs[z_idx].astype(int)
    
    em_vol = np.zeros(em_shape)
    print('reading em images')
    for z, idx in enumerate(z_idx):
        print(f'reading em z = {i}')
        image_path = source_path + em_files[idx]
        img = Image.open(image_path)
        em_vol[z,:,:] = img
    print('writing em images')
    zrm_em[:,:,:] = em_vol
    #em_vol.clear()  
 
# Segmentation
zt.makeMultiscaleGroup(z_path=z_path, group_name='seg', zarr_shape=em_shape,
                        z_chunk_shape = z_chunk_shape, num_scales=4, dsamp= dsamp)
zarr_root = zarr.open_group(z_path, mode='a')
zrm_seg = zarr_root['seg']['0'] 

if 0:
    # make segmentation volume
    seg_vol = np.zeros(em_shape)
    
    print('reading segmentation')
    base_num = 256
    for z in zs:
        print(f'reading segmenation z = {z}')
        image_name = f'Segmentation1-LX_8-14.vsseg_export_s{int(z):03}.png'
        image_path = source_path + image_name
        img = np.array(Image.open(image_path),'float64')
        img_flat = img[:,:,0] * (base_num ** 2) + \
            img[:,:,1] * (base_num ** 1) + img[:,:,2] * (base_num ** 0)
        seg_vol[z,:,:] = img_flat
    print('writing segmentation')
    zrm_seg[:,:,:] = seg_vol
    #seg_vol.clear()


# Down sample
if 0:
    zrm_group_seg = zrm_seg = zarr_root['seg']
    zrm_group_em = zrm_seg = zarr_root['em']
    zt.fill_multiscale_from_high_to_low(zrm_group_seg, 'max', dsamp=[1,2,2])
    zt.fill_multiscale_from_high_to_low(zrm_group_em, 'mean', dsamp=[1,2,2])
    

# review results
zup_em = zrm_group_em[f'{0}']
zdn_em = zrm_group_em[f'{2}']
   
zup_seg = zrm_group_seg[f'{0}']
zdn_seg = zrm_group_seg[f'{2}']
   
## View
plt.ion()
f1 = dsp.figure()
f1.fig.clear()
f1.ax[0] = f1.fig.add_subplot(2,2,1)
f1.ax.append(f1.fig.add_subplot(2,2,2))
f1.ax.append(f1.fig.add_subplot(2,2,3))
f1.ax.append(f1.fig.add_subplot(2,2,4))
cmap = dsp.col_map(c_type='rand', num_col= 10000)
f1.make_img(ax_id=0, i_shape=zup_em.shape[1:])
f1.make_img(ax_id=1, i_shape=zdn_em.shape[1:])
f1.make_img(ax_id=2, i_shape=zup_seg.shape[1:],vmax=10000, cmap=cmap)
f1.make_img(ax_id=3, i_shape=zdn_seg.shape[1:],vmax=10000, cmap=cmap)

seg_max = seg_vol.max(0)
f1.ax[1].img.set_data(seg_max)

for z in range(em_shape[0]):
    print(z)
    f1.ax[0].img.set_data(zup_em[z,:,:])
    f1.ax[1].img.set_data(zdn_em[z,:,:])
    f1.ax[2].img.set_data(zup_seg[z,:,:])
    f1.ax[3].img.set_data(zdn_seg[z,:,:])
    f1.update
    plt.pause(.1)










