import zarr
import numpy as np
import bifo.tools.voxTools as vt
import bifo.tools.dtypes as dtypes




def makeMultiscaleGroup(z_path=None, group_name=None, zarr_shape=None,
                        z_chunk_shape = [32, 256, 256], num_scales=9, dsamp= [1,2,2]):
        """
        Make multiscale zarr with specific group name and number of mip levels (downsampl = 2 ** level)
        if zarr path exists, new group or new mip levels will be added
        
        makeMultiscaleGroup(
           z_path = "G:/././..../name_of_zarr.zarr",
           group_name = 'name_of_group',
           zarr_shape = [1400, 7200, 50400], # shape at mip 0
           z_chunk_shape = [8, 128, 128], # default is [32, 256, 256]
           num_scales = 9 # default is 9
        )
    
        """ 
       
        zarr_root = zarr.open_group(z_path, mode='a')
        zarr_ms_group = zarr_root.require_group(group_name)
        
        zarr_ms_group.attrs['multiscales'] = [{
            'version': '0.1',
            'name': group_name,
            'datasets': [{'path': f'{i}'} for i in range (num_scales)],
            'axes': [
                {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
                {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                {'name': 'x', 'type': 'space', 'unit': 'micrometer'},
                ]
        }]
        
        
        for level in range(num_scales):
            scale = np.array(dsamp) ** level
            down_shape = tuple(np.ceil(np.array(zarr_shape) / scale).astype(int))
            down_chunks = tuple(np.minimum(z_chunk_shape, down_shape))
            
            zarr_ms_group.require_dataset(
                name= str(level),
                shape=down_shape,
                chunks=down_chunks,
                dtype='float32',
                compressor=zarr.Blosc(cname='zstd'))
            
# def list_chunks(zrm, p_window=None):    
#     z_shape = zrm.shape
#     z_chunks = zrm.chunks
    
#     check_s = []
#     for d in range(3):
#         check_s.append(np.arange(0,z_shape[d],z_chunks[d]))
    
#     if p_window is not None:   
#         pw = np.array(p_window)
#         for d in range(3):
#             # is_in_low = vt.is_in1(check_s[d], pw[d][0], pw[d][1])
#             # is_in_high = vt.is_in1(check_s[d] + z_chunks[d]-1, pw[d][0], pw[d][1])
#             # check_s[d] = check_s[d][is_in_low & is_in_high]
#             check_s[d] = check_s[d][
#                 (check_s[d] <= pw[d][1]) & 
#                 ((check_s[d] + z_chunks[d]-1) >= pw[d][0])]
    
#     chkl = []
#     for r in check_s[1]:
#         for c in check_s[2]:
#             for z in check_s[0]:
#                 lower = np.array([z, r, c]).flatten()
#                 upper = np.min((lower + z_chunks-1,z_shape),0).flatten()
#                 shape = upper - lower + 1
#                 chkl.append({'lower':lower, 'upper': upper,
#                              'cap':upper + 1, 'shape': shape})
    
#     return chkl

def chunk_from_ch(zrm,ch):
    chunk = zrm[ch['lower'][0]:ch['cap'][0],
                ch['lower'][1]:ch['cap'][1],
                ch['lower'][2]:ch['cap'][2]]
    return chunk

def zarr_max(zrm, dim=0, p_window=None):
    kds = np.setdiff1d([0,1,2],dim)
    chkl = vt.list_chunks(zrm, p_window)
    if p_window is None:
        w_low = np.array([0,0,0])
        w_shape = zrm.shape
        w_cap = w_low + w_shape
    else:
        pw = np.array(p_window)
        w_low = pw[:,0] 
        w_shape = np.diff(pw,1).flatten() + 1
        w_cap = w_low + w_shape
        
    max_i = np.zeros((w_shape[kds[0]], w_shape[kds[1]]))
    print('reading chunks for max')
    for i,ch in enumerate(chkl):
        z_lw = np.max([[w_low], [ch['lower']]],0).flatten()
        w_lw = z_lw - w_low
        c_lw = z_lw - ch['lower']
        
        z_cp = np.min([[w_cap], [ch['cap']]],0).flatten()
        w_cp = z_cp - w_low
        c_cp = z_cp - ch['lower']
        
        # tr = block_to_window(vol_box, ch)
                        
        # zrm[tr['full'][0,0]:tr['full'][1,0], 
        #     tr['full'][0,1]:tr['full'][1,1], 
        #     tr['full'][0,2]:tr['full'][1,2]] = \
        #     block[tr['ch'][0,0]:tr['ch'][1,0],
        #           tr['ch'][0,1]:tr['ch'][1,1],
        #           tr['ch'][0,2]:tr['ch'][1,2]]

        
        chunk = chunk_from_ch(zrm,ch)
        
        #Project
        max_chunk = chunk[c_lw[0]:c_cp[0], c_lw[1]:c_cp[1], c_lw[2]:c_cp[2]].max(dim)
        max_i[w_lw[kds[0]]:w_cp[kds[0]],
              w_lw[kds[1]]:w_cp[kds[1]]]  = \
              np.max([max_i[w_lw[kds[0]]:w_cp[kds[0]],
              w_lw[kds[1]]:w_cp[kds[1]]], 
              max_chunk],dim)
              
    return max_i
            
def list_of_subs_to_zarr(vast_subs, zrm, sub_to_zar_dim = [2,0,1]):
    
    num_o = len(vast_subs)
    sub_to_zar_dim = np.array(sub_to_zar_dim)
    # get bounding boxes
    print('finding bounding boxes')
    bbox = np.zeros((num_o,3,2),int)
    v_has_dat = np.zeros(num_o,'bool')
    vox_count = 0
    for i,s in enumerate(vast_subs):
        bbox[i,:,0] = s.min(0)
        bbox[i,:,1] = s.max(0) + 1
        if len(s.shape) >1:
            v_has_dat[i] = True
            vox_count = vox_count + s.shape[0]
      
    bbox = bbox[:,sub_to_zar_dim,:]        
    good_o = np.where(v_has_dat)[0]
            
    vol_box = np.stack((bbox[good_o,:,0].min(0),bbox[good_o,:,1].max(0)),0)
    
    # calculate blocks to run   
    pw = vol_box.transpose()
    chkl = vt.list_chunks(zrm.shape, pw)
    #chkl = list_chunks(zrm)

    b_info = []
    for ch in chkl:
              
        in_box_list = np.where( 
            np.sum((bbox[:,:,1] >= ch['lower']) &  
                        (bbox[:,:,0] <= ch['cap']),1) == 3
            )[0]

        ch['vs'] = in_box_list
        
        if in_box_list.shape[0]:
            b_info.append(ch)
        
    in_box_nums = [d['vs'].shape[0] for d in b_info]
    in_box_sum = np.sum(in_box_nums)
    print(f'{in_box_sum} found in chkl. {num_o} objects total')

    print('writing to zarr')
    tk = dtypes.ticTocDic() # get timer
    tk.m(['block', 'check is_in','sub_to_block','write']) #initialize timer dictionary
    block_shape = zrm.chunks
    block = np.zeros(block_shape,'int32')
    total_vox_in_blocks = 0
    total_vox_is_in = 0
    for i,bi in enumerate(b_info):
        tk.b('block')
        block = block * 0
        lower = bi['lower']
        cap = bi['cap']
        for iv,v in enumerate(bi['vs']):
            tk.b('check is_in')
            is_in = np.where(vt.is_in3(vast_subs[v][:,sub_to_zar_dim], 
                    lower, cap))[0]
            total_vox_is_in = total_vox_is_in + is_in.shape[0]
 
            tk.e('check is_in')
            if is_in.shape[0]:
                tk.b('sub_to_block')
                sub = vast_subs[v][is_in,:]  
                sub = sub[:,sub_to_zar_dim]
                b_sub = sub.astype(int) - lower
                block[b_sub[:,0], b_sub[:,1], b_sub[:,2]] = v
                tk.e('sub_to_block')
        
        tk.b('write')
        tr = vt.block_to_window(vol_box, bi)
        print(tr['full'])
        sum_vox = np.sum(block>0)       
        total_vox_in_blocks = total_vox_in_blocks + sum_vox
        print(f'block sum = {sum_vox}, total vox = {total_vox_in_blocks}')
              
        zrm[tr['full'][0,0]:tr['full'][1,0], 
            tr['full'][0,1]:tr['full'][1,1], 
            tr['full'][0,2]:tr['full'][1,2]] = \
            block[tr['ch'][0,0]:tr['ch'][1,0],
                  tr['ch'][0,1]:tr['ch'][1,1],
                  tr['ch'][0,2]:tr['ch'][1,2]]
                  
        tk.e('write')          
        tk.e('block')
        print(f'block {i} of {len(b_info)}.')
        tk.pl()
        tk.pt() 
        
    percent_filled = total_vox_in_blocks / np.prod(vol_box) * 100
    print(f'{percent_filled:.8f}% filled')
    percent_vox_found = total_vox_is_in / vox_count * 100
    print(f'{percent_vox_found}% of known voxes found')
    
    
# def block_to_window(pw,ch):
#     z_lw = np.max([[pw[0,:]], [ch['lower']]],0).flatten()
#     w_lw = z_lw - pw[0,:]
#     c_lw = z_lw - ch['lower']
    
#     z_cp = np.min([[pw[1,:]+1], [ch['cap']]],0).flatten()
#     w_cp = z_cp - pw[0,:]
#     c_cp = z_cp - ch['lower']
    
#     t_shape = c_cp - c_lw
#     has_data = t_shape.sum()>0  

#     transfer_info = {'full': np.array([z_lw,z_cp]),
#                      'win': np.array([w_lw, w_cp]),
#                      'ch': np.array([c_lw, c_cp]),
#                      'shape': t_shape,
#                      'has_data': has_data}    
    
#     return transfer_info
    
def fill_multiscale_from_high_to_low(zrm_group, method='mean', 
                                     from_mip=0, to_mip=None, 
                                     dsamp=[1,2,2]):
    
    dsamp = np.array(dsamp)
    # run to end of zarr group by default
    if to_mip is None:
       ds =  zrm_group.attrs['multiscales'][0]['datasets']
       to_mip = len(ds)-1
    
    for fm in range(to_mip - from_mip):
                
        tm = fm+1 
        zup = zrm_group[f'{fm}']
        zdn = zrm_group[f'{tm}']
        zup_chunk = zup.chunks
        zup_shape = np.array(zup.shape)
        grab_shape = zup_chunk * np.array(dsamp)
        z_window = np.array([np.array([0,0,0]),zup_shape])
        chkl = vt.list_chunks(z_window, grab_shape)
        
        print(f'downsampling mip {fm} to mip {tm}') 
        for c in chkl:
           
           cw = c['tr']['full']
           cw_down = np.ceil(cw / dsamp).astype(int)
        
           zup_vol = zup[cw[0,0]:cw[1,0], cw[0,1]:cw[1,1], cw[0,2]:cw[1,2]]       
           ds_vol = vt.downsample_3d_kernel(zup_vol, dsamp, method)
           zdn[cw_down[0,0]:cw_down[1,0], 
                 cw_down[0,1]:cw_down[1,1], 
                 cw_down[0,2]:cw_down[1,2]] = ds_vol    
            
    
    
    
    