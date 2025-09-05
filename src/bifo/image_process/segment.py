# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 12:20:25 2025

@author: jlmorgan
"""

import numpy as np

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

def fuse_connected_sv(sv, fuse_dim = [0, 1, 2]):
    
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    
    pairs = []
    for d in range(sv.ndim):
       s1 = [slice(None)] * sv.ndim
       s2 = [slice(None)] * sv.ndim
       s1[d] = slice(1, None)     # 1..end
       s2[d] = slice(None, -1)    # 0..end-1

       x = sv[tuple(s1)]
       y = sv[tuple(s2)]

       m = (x != y) & (x > 0) & (y > 0)   # exclude background & self
       if np.any(m):
           a = x[m].ravel()
           b = y[m].ravel()
           # normalize unordered pairs
           ab = np.stack((np.minimum(a,b), np.maximum(a,b)), axis=1)
           pairs.append(ab)

    if not pairs:
        return sv.copy()

    pairs = np.concatenate(pairs, axis=0)
    # keep unique unordered pairs
    pairs = np.unique(pairs, axis=0)

    # Build sparse undirected graph on labels [0..max]
    max_lab = int(max(sv.max(), pairs.max()))
    # edges in both directions
    rows = np.concatenate([pairs[:,0], pairs[:,1]])
    cols = np.concatenate([pairs[:,1], pairs[:,0]])
    data = np.ones_like(rows, dtype=np.uint8)

    G = coo_matrix((data, (rows, cols)), shape=(max_lab+1, max_lab+1))
    n_comp, comp_labels = connected_components(G, directed=False, return_labels=True)  # :contentReference[oaicite:2]{index=2}

    # Map every voxel label to its component id; keep background at 0
    comp_labels[0] = 0
    sv_f = comp_labels[sv]
    return sv_f
    
    
    
    
    
    # a = []
    # b = []
    # for d in fuse_dim:
    #     svr = np.roll(sv,1,d)
    #     svd = (svr - sv) !=0
    #     svd = svd & (sv>0) & (svr>0)
    #     a.append(sv[svd])
    #     b.append(svr[svd])
        
    # if a:
    #     a = np.concatenate(a, axis=0)
    #     b = np.concatenate(b, axis=0)
    # else:
    #     a = np.array([])
    #     b = np.array([])
        
    # pairs = np.stack((a, b), axis=1)
    # pairs = np.sort(pairs,1)
    # pairs = np.unique(pairs, axis=0)
    # max_val = sv.max()
    # lookup = np.arange(0, max_val+1)
    # lookup[pairs[:,1]] = pairs[:,0]
    # last_sum = -1
    # while last_sum != lookup.sum():
    #     last_sum = lookup.sum()
    #     lookup = lookup[lookup]
    
    # sv_f = lookup[sv]
    # return sv_f









