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
    want to retrieve all the voxels at once.
    
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

def get_win(arr, win):
    if arr.ndim == win.shape[1]:
        return arr[win[0,0]:win[1,0], 
            win[0,1]:win[1,1],
            win[0,2]:win[1,2]]
    else:
        return arr[win[0,0]:win[1,0], 
               win[0,1]:win[1,1],
               win[0,2]:win[1,2],
               :]
    

def put_win(arr1, win1, arr2, win2=None):
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

def upsample_array(arr, usamp):
    """
    Upsample a NumPy array by repeating elements 

    Args:
        arr (np.ndarray): The input array.
        usamp (list or tuple): List of scale factors, e.g., [1, 2, 2].

    Returns:
        np.ndarray: The upsampled array.
    """
    usamp = np.array(usamp).astype(int).flatten()
    ndim = arr.ndim
    nscale = len(usamp)

    for d in range(nscale):
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















 