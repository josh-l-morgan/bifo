# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:29:56 2025

@author: jlmorgan

Convert em volume that has been diced to be read by VAST into a zarr file


Example input
paths = {        
    'diced_dir':"//storage1.ris.wustl.edu/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/diced/",
    'z_path': "//storage1.ris.wustl.edu\jlmorgan\Active\morganLab\DATA\LGN_Developing\KxR_P11LGN\zarrs\KxR_ms_em_seg.zarr/",
    'z_group': 'em'
}

transfer_request = {
    'save_progress_every': 10,
    'start_fresh': 0}

map_request = {
    "diced_dir": paths['diced_dir'],
    "mip_level": 1,
    "analyze_section": [400], 
    "volume_lower_corner_zyx": [0,0,0],
    "volume_size_zyx": None, # size of volume at mip level to be processed, if None, then all
    "chunk_shape_zyx": [64, 512, 512], # size to be fed to model
    "chunk_overlap": [0, 0, 0],
    'full_data_shape': None
    }

zarr_request = {
    'zarr_shape':[1400, 72000, 50400],
    'chunk_shape':[64, 256, 256],
    'dsamp':[1,2,2]
    }

dicedToZarr.diced_to_zarr(paths,
                          transfer_request, 
                          map_request, 
                          zarr_request)


"""
import zarr
import numpy as np
import dill as pickle
import os

from bifo.tools import dtypes as dt
from bifo.tools import zarr_tools as zt
from bifo.tools import vox_tools as vt
from bifo.diced import fetchChunckFromMips as fetch


def diced_to_zarr(paths,tr,mr,zr):
    
    for key in paths:
        paths[key] = paths[key].replace('\\','/')
    
    # find zrm shape  
    zrm_shape = [int(-(-z//d**mr['mip_level'])) for z,d in zip(zr['zarr_shape'], 
                                              zr['dsamp'])]
    if mr['volume_size_zyx'] is None:
        mr["volume_size_zyx"] = zrm_shape
    
    
    if not os.path.isfile(paths['z_path'] + paths['z_group']):
        zt.makeMultiscaleGroup(z_path=paths['z_path'], group_name=paths['z_group'], 
                               zarr_shape=zr['zarr_shape'], z_chunk_shape = zr['chunk_shape'], 
                               num_scales=9, dsamp=zr['dsamp'])
            
    zarr_root = zarr.open_group(paths['z_path'], mode='a')
    zarr_group = zarr_root[paths['z_group']]
    zrm = zarr_group[f"{mr['mip_level']}"]
    
    project_path = paths['z_path'] + paths['z_group'] + '/'
    progress_file = project_path + 'last_chunk.pkl'
    
    print('mapping diced')
    fd = fetch.fetchDiced(mr)
    
    #fd_file = project_path + 'fetch_dice_object.pkl'
    # if (os.path.isfile(fd_file)  & ~start_fresh):
    #     print('loading previous pickled fd, fetch map')
    #     with open(fd_file, 'rb') as f:
    #         fd = pickle.load(f)
    # else:
    #     print('mapping diced dataset to produce list of chunks for processing')
    #     fd = fetch.fetchDiced(map_request)
    #     with open(fd_file, 'wb') as f:
    #          pickle.dump(fd, f)  
    
    if (os.path.isfile(progress_file) & ~tr['start_fresh']):
        print('loading previous progress')
        with open(progress_file, 'rb') as f:
            progress = pickle.load(f)
    else:
        print('starting new progress file')
        progress = {'last_chunk': - 1}
        with open(progress_file, 'wb') as f:
             pickle.dump(progress, f) 
    
    # get data from fd to determine positions of prediction assignments
    c1 = fd.chunk_list["clipped_corners_1"]
    c2 = fd.chunk_list["clipped_corners_2"] 
    
    num_chunk = c1.shape[0]
    
    tk = dt.ticTocDic()
    tk.m(['run','read','write','meta'])
    tk.b('run')
    last_times = np.zeros(1000)
    pw = np.array([[0, 0, 0], zrm_shape])
    for c in range(progress['last_chunk']+1,num_chunk):
        tk.b('read')
        print(f'reading chunk {c} of {num_chunk}')
        fd.readChunk(c) #reads chunk from cache or disk into fd.chunk
        tk.e('read')
        
        tk.b('write')
        cw = np.array([c1[c,:], c2[c,:]+1])
        tc = vt.block_to_window(pw,cw)
        zrm[tc['full'][0,0]:tc['full'][1,0], 
            tc['full'][0,1]:tc['full'][1,1], 
            tc['full'][0,2]:tc['full'][1,2]] = \
        fd.chunk[tc['ch'][0,0]:tc['ch'][1,0], 
            tc['ch'][0,1]:tc['ch'][1,1], 
            tc['ch'][0,2]:tc['ch'][1,2]]
        tk.e('write')
        
        
        tk.pl()
        tk.b('meta')
        last_times = np.roll(last_times,1)
        last_times[0] = tk.td['read'][2] + tk.td['write'][2]
        mean_time = np.mean(last_times[last_times>0])
        remaining_time = (num_chunk-c)*mean_time
        remaining_hours = remaining_time/60/60
        print(f'Estimated {remaining_hours:.2f} hours remaining')
    
        if not c % tr['save_progress_every']:
            progress['last_chunk'] = c
            with open(progress_file, 'wb') as f:
                pickle.dump(progress, f)  
        tk.e('meta')
    
    




