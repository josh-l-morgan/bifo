"""
Created on Fri Jul 18 15:59:48 2025

@author: jlmorgan
"""
import numpy as np
from scipy.stats import mode
import torch

def is_in3(sub, lower, cap = None):
    if cap is None:
        cap = lower+1
    # upper inclusive
    is_in =  (sub[:,0] >= lower[0]) & (sub[:,0] < cap[0]) & \
             (sub[:,1] >= lower[1]) & (sub[:,1] < cap[1]) & \
             (sub[:,2] >= lower[2]) & (sub[:,2] < cap[2])
    return is_in

def is_in_win(sub, win):
    # upper inclusive
    is_in = sub[:,0] *  0 + 1
    for d in range(sub.shape[1]):
        is_in = is_in * (sub[:,d] >= win[0,d]) * (sub[:,d] < win[1,d])
    return is_in


def is_in1(sub, lower, cap=None):
    if cap is None:
        cap = lower + 1
    # upper inclusive
    is_in =  (sub >= lower) & (sub < cap)
              
    return is_in

def list_chunks(pw, chunk_shape):    
    """
    Use case: you want to retrieve 3D voxels from one source and deliver them to 
    another source. You only want to fill a limited target window and you dont
    want to retrieve all the voxels at once. Resulting chunk list should be 
    registered to 0 for zarr chunking
    
    
    Provides list of chunk dictionaries for filling a target window.
    Each dictionary includes the coordinates
    for for the chunk in reference space and the tr transfer dictionary for 
    moving the chunk into the target window

    Parameters
    ----------
    pw : 2 x 3 array 
         Target window to fill
         pw[0,:] is the lower corner of a window
         pw[1,:] is the cap
    chunk_shape : 3, array 
         shape of chunks used to fill pw
       

    Returns
    -------
    chkl : list of dictionaries
        chunks will assume they should be placed relative to reference 
        voxel [0,0,0]. Chunks will cover target window and will include all the
        transfer clipping information for moving data from chunk to window
    
    """
    # pw is exclusive
    pw = np.array(pw).astype(int)
    chunk_shape = np.array(chunk_shape).astype(int)
    chunk_start = np.astype((np.floor(pw[0,:] / chunk_shape) * chunk_shape),int)
    
    check_s = []
    for d in range(3):
        check_s.append(np.arange(chunk_start[d], pw[1,d], chunk_shape[d]))
          
    chkl = []
    for r in check_s[1]:
        for c in check_s[2]:
            for z in check_s[0]:
                lower = np.array([z, r, c]).flatten()
                cap = np.min((lower + chunk_shape,pw[1,:]),0).flatten()
                shape = cap - lower
                
                ch = {'lower':lower, 'upper': cap-1,
                     'cap':cap, 'shape': shape}
                cw = np.array([lower,cap])
                ch['tr'] = block_to_window(pw,cw)                 
                chkl.append(ch)     
                
    return chkl            
         
def list_chunks_overlap(pw, chunk_shape, overlap=None, aw=None):    
    """
    Use case: you want to retrieve 3D voxels from one source and deliver them to 
    another source. You only want to fill a limited target window and you dont
    want to retrieve all the voxels at once. Resulting chunk list should be 
    registered to 0 for zarr chunking
    
    
    Provides list of chunk dictionaries for filling a target window.
    Each dictionary includes the coordinates
    for for the chunk in reference space and the tr transfer dictionary for 
    moving the chunk into the target window

    Parameters
    ----------
    pw : 2 x 3 array 
         Target window to fill
         pw[0,:] is the lower corner of a window
         pw[1,:] is the cap
    chunk_shape : 3, array 
         shape of chunks used to fill pw
       

    Returns
    -------
    chkl : list of dictionaries
        chunks will assume they should be placed relative to reference 
        voxel [0,0,0]. Chunks will cover target window and will include all the
        transfer clipping information for moving data from chunk to window
    
    """
    
    chunk_shape = np.array(chunk_shape, int)

    if overlap is None:
        overlap = chunk_shape * 0
    else:
        overlap = np.array(overlap,int)
    
    # pw is exclusive
    pw = np.array(pw, int)
    if aw is None:
        aw = pw.copy()
        
    chunk_start = ((np.floor(pw[0,:]-overlap) / chunk_shape) * chunk_shape).astype(int) # includes pw but registered to 0,0,0 + n * chunk_shape
    clip_shape = (chunk_shape - overlap * 2).astype(int)
    step = chunk_shape - overlap
    chunk_stop = (np.floor((pw[1,:] + overlap) / chunk_shape) * chunk_shape).astype(int)
   
    chunk_complete_w = np.array([[0, 0, 0], chunk_shape],int)
    clip_coomplete_w = np.array([[0,0,0], clip_shape], int)
    
    check_s = []
    for d in range(3):
        check_s.append(np.arange(chunk_start[d], chunk_stop[d]+1, step[d]))
          
    chkl = []
    for r in check_s[1]:
        for c in check_s[2]:
            for z in check_s[0]:
                lower = np.array([z, r, c]).flatten()
                cap_complete = lower + chunk_shape
                cap = np.min((cap_complete, pw[1,:]), 0).flatten()
                # shape = cap - lower
                # ch = {'lower':lower, 'upper': cap-1,
                #      'cap':cap, 'shape': shape}
                cw = np.array([lower,cap])
                chunk_complete = {'full': np.array([lower,cap_complete]),
                                  'ch': chunk_complete_w}
                chunk = block_to_window(aw,cw)                 
                
                clip_lower = lower + overlap
                clip_cap = cap - overlap
                # ch['clip_shape'] = clip_cap- clip_lower
                                
                clip_cw = np.array([clip_lower, clip_cap])
                clip_complete = {'full': clip_cw,
                                 'ch': clip_coomplete_w}
                
                clip = block_to_window(aw, clip_cw)                 
                ch = {'chunk_complete': chunk_complete,
                      'clip_complete': clip_complete,
                      'chunk': chunk, 
                      'clip': clip,
                      }
                chkl.append(ch)   
                
    return chkl                   
                 
def block_to_window(pw,cw):
    """
    Provides coordinates for placing voxels from cw in pw. Assumes both sets
    of coordinates refer to the same reference frame

    Parameters
    ----------
    pw : 2 x 3 array 
         Target window to fill
         pw[0,:] is the lower corner of a window
         pw[1,:] is the cap
    cw : 2 x 3 array 
        DESCRIPTION.

    Returns
    -------
    transfer_info : dictionary
        information required for transfering voxels from cw to pw
        includes the lower corner and cap for reference space, source chunk,
        and target window. Provides the shape of the transfered clipped volume.
        Provides boolean for whether pw or cp overlap (has_data)

    """
    #transfer coords
    z_lw = np.max([[pw[0,:]], [cw[0,:]]],0).flatten() #full space
    w_lw = z_lw - pw[0,:] #window space
    c_lw = z_lw - cw[0,:] #chunk space
    
    z_cp = np.min([[pw[1,:]], [cw[1,:]]],0).flatten() #full space
    w_cp = z_cp - pw[0,:]
    c_cp = z_cp - cw[0,:]
    
    t_shape = c_cp - c_lw
    has_data = (t_shape<1).sum() == 0  
   
    transfer_info = {'full': np.array([z_lw,z_cp]).astype(int),
            'win': np.array([w_lw, w_cp]).astype(int),
            'ch': np.array([c_lw, c_cp]).astype(int),
            'shape': t_shape.astype(int),
            'has_data': has_data}    
    
    return transfer_info

def get_chunk(arr, ch):
    if arr.ndim == ch.shape[1]:
        return arr[ch['lower'][0]:ch['cap'][0], 
            ch['lower'][1]:ch['cap'][1],
            ch['lower'][2]:ch['cap'][2]]
    else:
        return arr[ch['lower'][0]:ch['cap'][0], 
            ch['lower'][1]:ch['cap'][1],
            ch['lower'][2]:ch['cap'][2],
            :]

def get_win_brute(arr, win):
    if arr.ndim == win.shape[1]:
        return arr[win[0,0]:win[1,0], 
            win[0,1]:win[1,1],
            win[0,2]:win[1,2]]
    else:
        return arr[win[0,0]:win[1,0], 
               win[0,1]:win[1,1],
               win[0,2]:win[1,2],
               :]
    
def get_win(arr1, win1):
    
    dim1 = arr1.ndim
    
    win_f1 = np.array([np.zeros(dim1),arr1.shape],int)
    win_f1[:,-win1.shape[1]:] = win1    
    slice1 = win_to_slices(win_f1, ndim=None)
   
    return arr1[slice1]
    

def put_win_brute(arr1, win1, arr2, win2=None):
    if win2 is None:
        win2 = np.array([np.zeros(arr2.ndim), arr2.shape]).astype(int)
    
    if arr1.ndim == win1.shape[1]:
        arr1[win1[0,0]:win1[1,0], 
             win1[0,1]:win1[1,1],
             win1[0,2]:win1[1,2]] = \
             arr2[win2[0,0]:win2[1,0], 
                  win2[0,1]:win2[1,1],
                  win2[0,2]:win2[1,2]]  
        
    else:
        arr1[win1[0,0]:win1[1,0], 
             win1[0,1]:win1[1,1],
             win1[0,2]:win1[1,2],
             :] = \
             arr2[win2[0,0]:win2[1,0], 
                  win2[0,1]:win2[1,1],
                  win2[0,2]:win2[1,2],
                  :]   
    
    if isinstance(arr1, np.ndarray):
        return arr1
    
def put_win(arr1, win1, arr2, win2=None):
    if win2 is None:
        win2 = np.array([np.zeros(arr2.ndim), arr2.shape]).astype(int)
    
    dim1 = arr1.ndim
    dim2 = arr2.ndim
    
    win_f1 = np.array([np.zeros(dim1),arr1.shape],int)
    win_f1[:,-win1.shape[1]:] = win1
    win_f2 = np.array([np.zeros(dim2),arr2.shape],int)
    win_f2[:,-win2.shape[1]:] = win2
    
    slice1 = win_to_slices(win_f1, ndim=None)
    slice2 = win_to_slices(win_f2, ndim=None)
   
    arr1[slice1] = arr2[slice2]
    
    if isinstance(arr1, np.ndarray):
        return arr1
    
    
def win_to_slices(win, ndim=None):
    """win: shape (2, K) with start/stop per axis; K<=ndim."""
    if ndim is None:
        ndim = win.shape[1]
    start, stop = win
    sl = [slice(int(s), int(t)) for s, t in zip(start, stop)]
    # pad with ":" for remaining axes (e.g., channels)
    while len(sl) < ndim:
        sl.append(slice(None))
    return tuple(sl)

def put_win_confused(dst, w_dst, src, w_src=None, *, casting='no'):
    """
    Copy src[w_src] into dst[w_dst]. Supports arbitrary ndim and trailing channel dims.
    - w_*: array-like shape (2, K) [start_row..., stop_row...], K<=ndim
    - casting: passed to np.copyto (e.g., 'no', 'safe', 'unsafe')
    """
    if w_src is None:
        w_src = np.array([np.zeros(src.ndim), src.shape], dtype=int)

    # normalize to np.int
    w_dst = np.asarray(w_dst, dtype=int)
    w_src = np.asarray(w_src, dtype=int)

    s_dst = win_to_slices(w_dst, dst.ndim)
    s_src = win_to_slices(w_src, src.ndim)

    # fast copy (memcpy if both sides contiguous & same dtype)
    np.copyto(dst[s_dst], src[s_src], casting=casting)
    
    if isinstance(dst, np.ndarray):
        return dst
    
    
def array_to_tensor(arr1, win1, arr2, win2=None):
    if win2 is None:
        win2 = np.array([np.zeros(arr2.ndim), arr2.shape]).astype(int)
    
    ## reformat win to tensor shape
   
    full_win1 = np.array([np.zeros(arr1.ndim), np.ones(arr1.ndim)],int)
    full_win1[:,-win1.shape[1]:] = win1
    full_win2 =  np.array([np.zeros(arr2.ndim), np.ones(arr2.ndim)],int)
    full_win2[:,-win2.shape[1]:] = win2
    
    slices_1 = tuple(slice(c, s) for c, s in zip(full_win1[0,:], full_win1[1,:]))
    slices_2 = tuple(slice(c, s) for c, s in zip(full_win2[0,:], full_win2[1,:]))
    
    arr1[slices_1] = torch.from_numpy(arr2[slices_2])
    
    #return arr1
    
    
def downsample_win(win, dsamp):
    
    """
    Down sample window by rounding lower corner and adding a ceilinged 
    downsampled size
    
    """
    win = np.array(win,float)
    dsamp = np.array(dsamp,float)
    shape = win[1,:] - win[0,:]
    shape_ds = -(-shape // dsamp) ## half shape rounded up
    lower_ds = np.round(win[0,:] / dsamp)
    win_ds = np.array([lower_ds, lower_ds + shape_ds]).astype(int)
    return win_ds

def upsample_array(arr, rsamp):
    """
    Upsample a NumPy array by repeating elements 

    Args:
        arr (np.ndarray): The input array.
        usamp (list or tuple): List of scale factors, e.g., [1, 2, 2].

    Returns:
        np.ndarray: The upsampled array.
    """
    
    
    ndim = arr.ndim
    usamp = np.ones(ndim,int)
    nscale = len(rsamp)
    usamp[ndim-nscale:] = np.array(rsamp)

    for d in range(len(usamp)):
        arr = np.repeat(arr, usamp[d], axis=d)

    return arr
                 
def downsample_3d_kernel(arr, dsamp, method='mean'):
    """
    Downsample 3D volume using a 3D kernal with shape = dsamp
    Source will be expanded so that every voxel is used. Edge voxels that dont 
    match dsamp will be duplicated so that 
    final shape = ceiling(source.shape/dsamp)

    Parameters
    ----------
    arr : 3D np array
        DESCRIPTION.
    dsamp : 3, array 
        shape of downsample kernel.
    method : string, optional
        method of downsampling within dsamp kernel   
        'mean' = np.mean
        'max' = np.max
        'min' = np.min
        
    Returns
    -------
    arr : 3D np array
        Downsampled array.

    """
    
    #duplicat edges to fill for downsample
    raw_shape = np.array(arr.shape)
    valid_shape = (np.ceil(arr.shape[0]/dsamp[0]) * dsamp[0],
                 np.ceil(arr.shape[1]/dsamp[1]) * dsamp[1],
                 np.ceil(arr.shape[2]/dsamp[2]) * dsamp[2])
    shape_dif = valid_shape - raw_shape
    arr = np.concatenate((arr,np.repeat(arr[:,raw_shape[1]-1:raw_shape[1],:], shape_dif[1], 1)),1)
    arr = np.concatenate((arr,np.repeat(arr[:,:,raw_shape[2]-1:raw_shape[2]], shape_dif[2], 2)),2)
    arr = np.concatenate((arr,np.repeat(arr[raw_shape[0]-1:raw_shape[0],:,:], shape_dif[0], 0)),0)
    
    # reshape
    new_shape = (arr.shape[0]//dsamp[0], dsamp[0],
                 arr.shape[1]//dsamp[1], dsamp[1],
                 arr.shape[2]//dsamp[2], dsamp[2])
    
    # apply method
    match method:
        case 'mean':
            arr = arr.reshape(new_shape).mean(axis=(1, 3, 5))
        case 'max':
            arr = arr.reshape(new_shape).max(axis=(1, 3, 5))
        case 'min':
            arr = arr.reshape(new_shape).min(axis=(1, 3, 5))

    return arr

def downsample_array(arr, dsamp, method='mean'):
    arr = np.asarray(arr)
    dz, dy, dx = map(int, dsamp)

    # pad by edge-repeat so each dim divisible by dsamp
    Z, Y, X = arr.shape
    Z2 = -(-Z // dz) * dz; Y2 = -(-Y // dy) * dy; X2 = -(-X // dx) * dx
    arr = np.pad(arr, ((0, Z2-Z), (0, Y2-Y), (0, X2-X)), mode='edge')

    # reshape into blocks
    va = arr.reshape(Z2//dz, dz, Y2//dy, dy, X2//dx, dx)  # (Z', dz, Y', dy, X', dx)

    # move block dims (1,3,5) to the front and collapse them
    va = va.transpose(1, 3, 5, 0, 2, 4)                    # (dz, dy, dx, Z', Y', X')
    va = va.reshape(dz*dy*dx, Z2//dz, Y2//dy, X2//dx)      # (K, Z', Y', X') where K=dz*dy*dx

    # reduce along the collapsed block axis (axis=0)
    if method == 'mean':
        out = va.mean(axis=0)
    elif method == 'max':
        out = va.max(axis=0)
    elif method == 'min':
        out = va.min(axis=0)
    elif method == 'mode':
        out = mode_of_array_dim_0(va)
    else:
        raise ValueError("method must be one of: mean, max, min, mode")

    return out

def mode_of_array_dim_0(va):
    num_v = va.shape[0]
    counts = va.copy() * 0
    for vi in range(num_v):
        counts[vi,:] = (va == va[vi,:]).sum(0)
    is_max = counts == counts.max(0)
    max_subs = np.where(is_max)
    mode_array = np.zeros(va.shape[1:])
    mode_array[max_subs[1], max_subs[2], max_subs[3]] =  va[max_subs]
    return mode_array




def downsample_3d_mode(arr, dsamp):
    """
    Version of downsample_3d_kernel meant to work with mode.
    Currently too slow to be worth while

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    dsamp : TYPE
        DESCRIPTION.

    Returns
    -------
    mode_vals : TYPE
        DESCRIPTION.

    """
    #downsample by averaging, might shrink
    #####NOTE: too slow, need to fix
    arr = arr[:arr.shape[0]//dsamp[0] * dsamp[0],
          :arr.shape[1]//dsamp[1] * dsamp[1],
          :arr.shape[2]//dsamp[2] * dsamp[2]]
    
    new_shape = (arr.shape[0]//dsamp[0], dsamp[0],
                 arr.shape[1]//dsamp[1], dsamp[1],
                 arr.shape[2]//dsamp[2], dsamp[2])
    
    reshaped = arr.reshape(new_shape)
        
    reshaped = reshaped.transpose(0, 2, 4, 1, 3, 5)
   
    # Flatten last three axes (the block) and compute mode
    block_flat = reshaped.reshape(*reshaped.shape[:3], -1)
   
    mode_vals, _ = mode(block_flat, axis=-1, keepdims=False)
   
    return mode_vals
    
def center_window(s1,s2):
    # get window for shape 2 in center of shape 1
    s1 = np.array(s1)
    s2 = np.array(s2)
    
    lower = (s1 - s2) // 2
    cap = lower + s2
    return np.array([lower, cap])    

def fill_edges(arr, clip, value=0):
    """
    arr = np.array
    clip = list of integers indicating how many border voxels to paint with value
    """

    ndim = arr.ndim
    if len(clip) == 1: # apply one clip to all dims
        clip = clip * ndim
            
    if len(clip) < ndim: # shift clip to last dims
        clip = [0] * (ndim - len(clip)) + clip
    
    
    slice_all = [slice(None) for _ in clip]
    slice_e = [[slice(0, int(cp)) for cp in clip],
               [slice(int(-cp), None) for cp in clip]]
    for di in range(ndim):
        for ei in range(2):
            slice_u = slice_all.copy()
            slice_u[di] = slice_e[ei][di]
            arr[tuple(slice_u)] = value
    
    return arr

def check_chkl(chkl):
    """
    turning chkl into arrays
    """
    num_chunks = len(chkl)
    crs = np.zeros((num_chunks, 3))
    clrs = np.zeros((num_chunks, 3))
    
    cchs = np.zeros((num_chunks, 3))
    cch_shapes = np.zeros((num_chunks, 3))
    cfs = np.zeros((num_chunks, 3))
    cf_shapes = np.zeros((num_chunks, 3))
    
    for ci in range(num_chunks):
        ch = chkl[ci]
        cr = ch['chunk_request']
        clr = ch['clip_request']
        crs[ci,:] = cr[0,:]
        clrs[ci,:] = clr[0,:]
        chunk_shape = cr[1,:] - cr[0,:]
        cw = ch['chunk']['ch']
        cchs[ci,:] = cw[0,:]
        cch_shapes[ci,:] = cw[1,:] - cw[0,:]
        cfw = ch['chunk']['full']
        cfs[ci,:] = cfw[0,:]
        cf_shapes[ci,:] = cfw[1,:] - cfw[0,:]
        
        print(f'chunk request shape = {chunk_shape}')
        
    
    cr_dif = crs - np.roll(crs, 1, 0)
    clr_dif = clrs - np.roll(clrs, 1, 0)
    cr_clr_dif = clrs - crs
    
    cch_dif = cchs - np.roll(cchs, 1, 0)
    cf_dif = cfs - np.roll(cfs, 1, 0)
    
    cr_clr_dif = clrs - crs



def _axis_ramp(n, clip, overlap, mode="linear"):
    """
    n: axis length (int)
    clip: non-negative int
    overlap: non-negative int (tile overlap used by your tiler)
    mode: "linear" or "cosine"
    """
    idx = np.arange(n, dtype=np.float32)
    d = np.minimum(idx, (n - 1) - idx)  # distance to nearest edge

    ramp_end = overlap - clip           # (= your "overlap minus clip")
    ramp_len = ramp_end - clip          # = overlap - 2*clip

    w = np.zeros_like(idx, dtype=np.float32)

    # if ramp_len <= 0, there is no room for a ramp; step from 0→1 right after clip
    if ramp_len <= 0:
        w[d >= clip] = 1.0
        return w

    # 0 in [0, clip)
    in_ramp = (d >= clip) & (d < ramp_end)
    # 1 in [ramp_end, center]
    w[d >= ramp_end] = 1.0

    t = (d[in_ramp] - clip) / ramp_len  # t in [0,1)
    if mode == "cosine":
        # smooth Hann-like edge
        w[in_ramp] = 0.5 - 0.5 * np.cos(np.pi * t)
    else:
        # linear
        w[in_ramp] = t.astype(np.float32)

    return w

def make_blend_mask(tile_shape, clip_vec, overlap_vec, mode="linear"):
    """
    tile_shape: (Z, Y, X)
    clip_vec:   (cz, cy, cx)
    overlap_vec:(oz, oy, ox)
    returns a (Z,Y,X) float32 mask in [0,1]
    """
    z, y, x = map(int, tile_shape)
    cz, cy, cx = map(int, clip_vec)
    oz, oy, ox = map(int, overlap_vec)

    wz = _axis_ramp(z, cz, oz, mode=mode)[:, None, None]
    wy = _axis_ramp(y, cy, oy, mode=mode)[None, :, None]
    wx = _axis_ramp(x, cx, ox, mode=mode)[None, None, :]

    return (wz * wy * wx).astype(np.float32)











 