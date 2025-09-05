# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 12:20:25 2025

@author: jlmorgan
"""

def sk_watershed(aff, vox_size=None, thresh=0):
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed
    from skimage.morphology import local_maxima  # or h_minima
    from skimage.measure import label

    
    
    if vox_size is None:
        vox_size = (1, 1, 1)
    mask = aff > thresh
    dist = distance_transform_edt(mask, sampling=vox_size)
    seeds = local_maxima(dist, connectivity=3)      # boolean
    markers = label(seeds, connectivity=3)
    sv = watershed(aff, markers=markers, mask=mask)  # supervoxels
    return sv

