# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:28:28 2025

@author: jlmorgan
"""


import numpy as np


def vast_pos_to_zarr(v_pos, mip=0, dsamp=[1,2,2]):
    
    vast_pos = np.array(v_pos)
    dsamp = np.array(dsamp)
    zarr_pos = vast_pos[[2,1,0]]
    zarr_pos = zarr_pos / (dsamp ** mip)
    return zarr_pos
   
    