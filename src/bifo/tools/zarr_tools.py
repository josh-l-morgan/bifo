"""

Created on  Jul 2025

@author: jlmorgan

Tools for making and manipulating zarr files

"""
import zarr
import numpy as np
import bifo.tools.vox_tools as vt
import bifo.tools.dtypes as dtypes
import matplotlib.pyplot as plt



def makeMultiscaleGroup(z_path=None, group_name=None, zarr_shape=None,
                        z_chunk_shape = [32, 256, 256], num_scales=9, dsamp= [1,2,2], dtype='float32'):
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
            down_shape = np.ceil(np.array(zarr_shape) / scale).astype(int)
            down_chunks = np.minimum(z_chunk_shape, down_shape)
            down_shape = down_shape.astype(int).tolist()
            down_chunks = down_chunks.astype(int).tolist()
            
            zarr_ms_group.require_dataset(
                name= str(level),
                shape=down_shape,
                chunks=down_chunks,
                dtype=dtype)
            #zarr.Blosc(cname='zstd')





def chunk_from_ch(zrm,ch):
    """
    Return a 3D volume form the zarr file 'zrm' useing a window defined 
    by a ch dictionary keys 'lower' and 'cap'

    Parameters
    ----------
    zrm : TYPE
        DESCRIPTION.
    ch : TYPE
        DESCRIPTION.

    Returns
    -------
    chunk : TYPE
        DESCRIPTION.

    """
    chunk = zrm[ch['lower'][0]:ch['cap'][0],
                ch['lower'][1]:ch['cap'][1],
                ch['lower'][2]:ch['cap'][2]]
    return chunk

def zarr_max(zrm, dim=0, p_window=None):
    """
    Create a maximum value projection image through a zar

    Parameters
    ----------
    zrm : TYPE
        DESCRIPTION.
    dim : TYPE, optional
        DESCRIPTION. The default is 0.
    p_window : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    max_i : TYPE
        DESCRIPTION.

    """
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
    """
    Write a list composed of n x3 arrays of subs into z zarr file.  The values
    written into the zarr file is the list index of the array
    
    Parameters
    ----------
    vast_subs : TYPE
        DESCRIPTION.
    zrm : TYPE
        DESCRIPTION.
    sub_to_zar_dim : TYPE, optional
        DESCRIPTION. The default is [2,0,1].

    Returns
    -------
    None.

    """
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
        
def fill_multiscale_from_high_to_low(zrm_group, method='mean', 
                                     from_mip=0, to_mip=None, 
                                     dsamp=[1,2,2],
                                     z_window=None,
                                     quiet=0):
    """
    Fill the mip levels of a multiscale zarr file by downsampling
    the zarr group zrm_group/from_mip must already be filled

    Parameters
    ----------
    zrm_group : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is 'mean'.
    from_mip : TYPE, optional
        DESCRIPTION. The default is 0.
    to_mip : TYPE, optional
        DESCRIPTION. The default is None.
    dsamp : TYPE, optional
        DESCRIPTION. The default is [1,2,2].

    Returns
    -------
    None.

    """
    
    dsamp = np.array(dsamp)
    # run to end of zarr group by default
    if to_mip is None:
       ds =  zrm_group.attrs['multiscales'][0]['datasets']
       to_mip = len(ds)-1
        
    if z_window is None:
        zup = zrm_group[f'{from_mip}']
        z_window = np.array([[0,0,0],zup.shape])
    
    move_mips = [r for r in range(from_mip, to_mip)]
    #mip_dif = [dsamp ** (r-from_mip) for r in move_mips]
    
    for m in range(len(move_mips)):
        
        fm = move_mips[m]
        tm = fm+1 
        zup = zrm_group[f'{fm}']
        zdn = zrm_group[f'{tm}']
        zup_chunk = zup.chunks
        grab_shape = zup_chunk * dsamp
        zwn = np.floor(z_window// (dsamp ** fm)) #  -(-z_window//mip_dif[m])
        chkl = vt.list_chunks(zwn, grab_shape)
        num_chunk = len(chkl)
       
        for ci in range(num_chunk):
            
            if not quiet:
                print(f'downsampling mip {fm} to mip {tm}, chunk {ci} of {num_chunk}')            
            
            ch = chkl[ci]
            cw_up_pull = ch['tr']['full']

            zup_vol = vt.get_win(zup, cw_up_pull)
            ds_vol = vt.downsample_3d_kernel(zup_vol, dsamp, method)
                        
            cw_z_push = vt.downsample_win(ch['tr']['full'], dsamp)
            cw_ds_pull = cw_z_push - cw_z_push[0,:]
     
            try:
                vt.put_win(zdn, cw_z_push, ds_vol, cw_ds_pull)
            except:
                print('bark')
                breakpoint()
          
def fill_multiscale_from_label(zrm_group, upsamp_method='nearest', 
                                     downsamp_method='mode',
                                     from_mip=4, 
                                     to_mip = None,
                                     dsamp=[1,2,2],
                                     z_window=None,
                                     quiet=0):
    """
    Fill the mip levels of a multiscale zarr file by downsampling
    the zarr group zrm_group/from_mip must already be filled

    Parameters
    ----------
    zrm_group : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is 'mean'.
    from_mip : TYPE, optional
        DESCRIPTION. The default is 0.
    to_mip : TYPE, optional
        DESCRIPTION. The default is None.
    dsamp : TYPE, optional
        DESCRIPTION. The default is [1,2,2].

    Returns
    -------
    None.

    """
    
    dsamp = np.array(dsamp)
    if z_window is None:
        zup = zrm_group[f'{from_mip}']
        z_window = np.array([[0,0,0],zup.shape])
        z_window = z_window * dsamp ** from_mip
    
    z_type = zrm_group[f'{from_mip}'].dtype
    
    if to_mip is None:
        ds =  zrm_group.attrs['multiscales'][0]['datasets']
        to_mip = [0,len(ds)-1]
    
    ## Going down
    move_mips = np.arange(from_mip, to_mip[1]+1).astype(float)
    
    for mi in range(len(move_mips)-1):
        
        fm = move_mips[mi]
        tm = move_mips[mi+1] 
        zfrom = zrm_group[f'{fm:.0f}']
        zto = zrm_group[f'{tm:.0f}']
        grab_shape = zfrom.chunks  * dsamp
        
        zwn = np.ceil(z_window / (dsamp ** fm)).astype(int)
        zwn[0,:] = np.max(np.array([[0,0,0], zwn[0,:]]),0)
        zwn[1,:] = np.min(np.array([zfrom.shape[-3:], zwn[1,:]]),0)
        chkl = vt.list_chunks(zwn, grab_shape)
        num_chunk = len(chkl)
       
        for ci in range(num_chunk):
            
            if not quiet:
                print(f'downsampling mip {fm} to mip {tm}, chunk {ci} of {num_chunk}')            
            
            
            ch = chkl[ci]
            cw_from = ch['tr']['full']

            zfrom_vol = vt.get_win(zfrom, cw_from)
            zto_vol = vt.downsample_array(zfrom_vol, dsamp, 'mode').astype(z_type) #'mode' is ideal but slower
            cw_to = vt.downsample_win(cw_from, dsamp)
     
            # vt.put_win(zto, cw_to, zto_vol)
            vt.put_win_brute(zto, cw_to, zto_vol)
            print(f'from {zfrom_vol.max()} to {zto_vol.max()}')
              
    ## going up
    move_mips = np.arange(from_mip, to_mip[0]-1, -1).astype(float)
    
    for mi in range(len(move_mips)-1):
        
        fm = move_mips[mi]
        tm = move_mips[mi+1] 
        zfrom = zrm_group[f'{fm:.0f}']
        zto = zrm_group[f'{tm:.0f}']
        grab_shape = zfrom.chunks * 1
        zwn = np.ceil(z_window / (dsamp ** fm)).astype(int)
        zwn[0,:] = np.max(np.array([[0,0,0], zwn[0,:]]),0)
        zwn[1,:] = np.min(np.array([zfrom.shape[-3:], zwn[1,:]]),0)
        chkl = vt.list_chunks(zwn, grab_shape)
        num_chunk = len(chkl)
       
        for ci in range(num_chunk):
            
            if not quiet:
                print(f'upsampling mip {fm} to mip {tm}, chunk {ci} of {num_chunk}')            
            
            ch = chkl[ci]
            cw_from = ch['tr']['full']

            zfrom_vol = vt.get_win(zfrom, cw_from)
            zto_vol = vt.upsample_array(zfrom_vol, dsamp).astype(z_type)
                        
            cw_to = vt.downsample_win(cw_from, 1/dsamp)
     
            try:
                vt.put_win(zto, cw_to, zto_vol)
            except:
                print('bark')
                breakpoint()
    
             
    
def scan_zarr_to_fig(zarr_group, pw=None, cw=[1, 512, 512], scan_mips=[3,2,1], dsamp=[1, 2, 2],  use_channel=None, mode=None):
    """
    Scan a window (pw) of a multiresolution zar-group.  Choose which mip levels
    to compare (scan_mips)

    Parameters
    ----------
    zarr_group : TYPE
        DESCRIPTION.
    pw : TYPE, optional
        DESCRIPTION. The default is None.
    cw : TYPE, optional
        DESCRIPTION. The default is [1, 512, 512].
    scan_mips : TYPE, optional
        DESCRIPTION. The default is [1].
    dsamp : TYPE, optional
        DESCRIPTION. The default is [1, 2, 2].

    Returns
    -------
    None.

    """
    import bifo.tools.display as dsp
    print(mode)
    num_mip = len(scan_mips)
    num_groups = len(zarr_group)
    dsamp = np.array(dsamp)
    
    if mode is None:
        mode = ['max'] * num_groups
    
    zrms = []
    for gi, g in enumerate(zarr_group):
        zrm = []
        for m in scan_mips:
            zrm.append(zarr_group[gi][f'{m}'])
        zrms.append(zrm)
    
    if pw is None:
        pw = np.array([[0, 0, 0], zrms[0][0].shape])
    else:
        pw = np.array(pw) 
    pw_f = pw.copy()
    cw_f = cw * dsamp ** scan_mips[0]

    chkl = vt.list_chunks(pw_f,cw_f)
    
    fig = dsp.figure(num_mip * num_groups, 'scan_zarr_window', rows=num_groups) 
    z_mean = np.zeros((cw[1], cw[2]))
    for a in range(len(fig.ax)):
        fig.ax[a].img = fig.ax[a].imshow(z_mean, cmap='gray', vmax=1)
    
    for ci, ch in enumerate(chkl):
        
        print(f"chunk {ci} of {len(chkl)}, corner={ch['lower']}")
        zw = ch['tr']['full']
        
        for gi in range(num_groups):
            for ai in range(num_mip):
                zw_d = (zw / (dsamp ** scan_mips[ai])).astype(int)
                if (use_channel is None) | zrms[gi][ai].ndim < 4:
                    z_samp =  zrms[gi][ai][zw_d[0,0]:zw_d[1,0], 
                               zw_d[0,1]:zw_d[1,1],
                               zw_d[0,2]:zw_d[1,2]]
                else:
                    z_samp =  zrms[gi][ai][use_channel[gi], zw_d[0,0]:zw_d[1,0], 
                               zw_d[0,1]:zw_d[1,1],
                               zw_d[0,2]:zw_d[1,2]]
                                        
                if mode[gi]=='mean':
                    z_mean = z_samp.mean(0)
                elif mode[gi] == 'max': 
                    z_mean = z_samp.max(0)
                    
                max_mean = z_mean.max()
                if max_mean:
                    z_mean = z_mean/max_mean +.3
                    
                ax_id = ai + gi * num_mip
                fig.ax[ax_id].img.set_data(z_mean)
                    
        fig.update() 
               



def write_ngff_v04_attrs_broken(z_path, group_name, base_voxel_size_um=(0.04, 0.004, 0.004),
                         dsamp=(1,2,2), has_channel=False, num_scales=9):
    """
    base_voxel_size_um = (z_um, y_um, x_um) at level 0
    dsamp = per-axis downsample factors between levels, e.g. (1,2,2)
    has_channel = True if your arrays are CZYX (else ZYX)
    """
    zarr_root = zarr.open_group(z_path, mode='a')
    zarr_ms_group = zarr_root.require_group(group_name)
    
    
    
    # enumerate existing levels
    levels = sorted([k for k,v in zarr_ms_group.items() if isinstance(v, zarr.Array)],
                    key=lambda s: int(s))
    zyx0 = np.array(base_voxel_size_um, float)
    dsamp = np.array(dsamp, float)

    # # axes
    # axes = []
    # if has_channel:
    #     axes.append({"name": "c", "type": "channel"})
    # axes += [
    #     {"name": "z", "type": "space", "unit": "micrometer"},
    #     {"name": "y", "type": "space", "unit": "micrometer"},
    #     {"name": "x", "type": "space", "unit": "micrometer"},
    # ]

    axes =  [
        {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'x', 'type': 'space', 'unit': 'micrometer'},
        ]



    # datasets = []
    # for i,lev in enumerate(levels):
    #     # physical scale at this level = base_voxel_size * (dsamp ** level_index)
    #     # prepend channel scale 1 if there's a channel axis
    #     scale_zyx = (zyx0 * (dsamp ** i)).tolist()
    #     scale = ([1.0] if has_channel else []) + scale_zyx
    #     datasets.append({
    #         "path": lev,
    #         "coordinateTransformations": [{"type": "scale", "scale": scale}]
    #     })
        
    datasets =  [{'path': f'{i}'} for i in range (num_scales)],


    zarr_ms_group.attrs["multiscales"] = [{
        "version": "0.1",
        "name": group_name, #group.name.rsplit("/",1)[-1] or "image",
        "axes": axes,
        "datasets": datasets
    }]

    # group.attrs["omero"] = {
    #     "channels": [
    #         {"label": "ch0", "color": "FFFFFF", "window": {"start": 0.0, "end": 1.0}},
    #         # add one dict per channel
    #     ]
    # }
    
def reset_multiscale_attrs(z_path, group_name, num_scales=9):
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
    
    
   
def write_ngff_v04_attrs(z_path, group_name, base_voxel_size_um=(1, 1, 1),
                         dsamp=(1,2,2), has_channel=False, num_scales=9):
    
    root = zarr.open_group(z_path, mode='a')
    g = root.require_group(group_name)
  
    
    
    # enumerate existing levels
    levels = sorted([k for k,v in g.items() if isinstance(v, zarr.Array)],
                    key=lambda s: int(s))
    zyx0 = np.array(base_voxel_size_um, float)
    dsamp = np.array(dsamp, float)
  
    
    axes =  [
        {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'x', 'type': 'space', 'unit': 'micrometer'},
        ]
    
    
    
    datasets = []
    for mi, lev in enumerate(levels):
        # physical scale at this level = base_voxel_size * (dsamp ** level_index)
        # prepend channel scale 1 if there's a channel axis
        
        scale_zyx = (zyx0 * (dsamp ** int(lev))).tolist()
        scale = ([1.0] if has_channel else []) + scale_zyx
        datasets.append({
            "path": lev,
            "coordinateTransformations": [{"type": "scale", "scale": scale}]
        })
    
    
    #datasets =  [{'path': f'{i}'} for i in range (num_scales)]

 
    
    g.attrs['multiscales'] = [{
        'version': '0.1',
        'name': group_name,
        'datasets': datasets,
        'axes': axes
    }]
     
    
    ms = g.attrs["multiscales"][0]
    print("version:", ms["version"])
    print("axes:", [a["name"] for a in ms["axes"]])
    print("paths:", [d["path"] for d in ms["datasets"]])
    print("missing:", [p for p in [d["path"] for d in ms["datasets"]] if p not in g.array_keys()])
        
    
    

def diagnose_ngff_v04(group_path):
    
    g = zarr.open_group(group_path, mode="r")
    ms_all = g.attrs.get("multiscales", [])
    print("arrays under group:", list(g.array_keys()))
    if not ms_all:
        print("❌ no multiscales in this group"); return
    ms = ms_all[0]
    print("version:", ms.get("version"))
    axes = ms.get("axes", [])
    print("axes:", [a.get("name") for a in axes])
    ds = ms.get("datasets", [])
    print("dataset paths:", [d.get("path") for d in ds])

    # 1) paths must be relative; must exist as arrays
    bad_paths = [d["path"] for d in ds if not isinstance(d.get("path"), str) or d["path"].startswith("/")]
    missing   = [d["path"] for d in ds if d.get("path") not in g.array_keys()]
    print("leading-slash paths:", bad_paths)
    print("missing arrays:", missing)

    # 2) scale length must equal len(axes)
    for i, d in enumerate(ds):
        ct = d.get("coordinateTransformations", [])
        if not ct:
            print(f"⚠️ dataset {d.get('path')} has no coordinateTransformations")
            continue
        for t in ct:
            if t.get("type") == "scale":
                scale = t.get("scale")
                print(f"scale[{d.get('path')}]:", scale)
                if not isinstance(scale, (list, tuple)) or len(scale) != len(axes):
                    print(f"❌ scale length {len(scale)} != #axes {len(axes)} at dataset {d.get('path')}")

    # # 3) sanity: sample a few voxels from declared datasets
    # for d in ds[:2]:  # first 2 datasets
    #     arr = g[d["path"]]
    #     z,y,x = np.array(arr.shape)//2
    #     print(f"sample {d['path']} shape={arr.shape} dtype={arr.dtype} mid={arr[z,y,x]}")
    #     # quick nonzero probe
    #     nz = np.any(arr[z, ::max(1,arr.shape[1]//128), ::max(1,arr.shape[2]//128)])
    #     print(f"  nonzero slice probe @z={z}: {nz}")


    
        