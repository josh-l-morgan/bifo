# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 09:58:28 2025

@author: jlmorgan
"""
import numpy as np
import bifo.tools.vox_tools as vt
import torch
from scipy.ndimage import rotate as nd_rotate
import copy
import bifo.tools.display as dsp
import matplotlib.pyplot as plt
from bifo.tools.dtypes import ticTocDic


class Dataset:
    def __init__(self, group, use_mips, v_vox_sizes):
        self.group = group
        self.use_mips = use_mips
        self.vox_sizes = np.array(v_vox_sizes)
        self.arr = []
        self.shapes = []
        self.fovs = []
        for mi, m in enumerate(use_mips):
            self.arr.append(group[f"{m}"])
            self.shapes.append(np.array(self.arr[-1].shape).astype(int))
            self.fovs.append(self.vox_sizes[mi] * self.shapes[-1])

class ZarrDictionaryManager:
    def __init__(self, tr):
        self.datasets = {}
        self.glob = {'use_mips': tr['use_mips'],
                       'v_vox_sizes': tr['v_vox_sizes'],
            }
        
    def add_dataset(self, key, group, use_mips=None, v_vox_sizes=None):
        if key in self.datasets:
            print(f"Key '{key}' already was overwritten.")
        use_mips = use_mips.copy() if use_mips is not None else self.glob['use_mips']
        v_vox_sizes = v_vox_sizes.copy() if v_vox_sizes is not None else self.glob['v_vox_sizes']
        self.datasets[key] = Dataset(group, use_mips, v_vox_sizes) 

    def get_dataset(self, key):
        return self.datasets.get(key, None)

    def __getitem__(self, key):
        return self.datasets[key]

    def __setitem__(self, key, value):
        self.datasets[key] = value



## Replace GP
class BifPowder():
    """
    
    source
    augment
    imfilter
    
    """
    def __init__(self, train_request):
        
        self.interpret_train_request(train_request)
        self.zm = ZarrDictionaryManager(self.tr)
        self.make_requests()
        self.jit = {'do': 0 }
        self.tk = ticTocDic()
        self.tk.m(['pull','batchify','augment', 'loss','other'])
        
    def interpret_train_request(self, train_request):
        tr = copy.deepcopy(train_request)
        
        tr['v_vox_sizes'] = []
        for vi, m in enumerate(tr['use_mips']):
            tr['v_vox_sizes'].append(np.array(tr['mip0_vox_size']) 
                                          * np.array(tr['zarr_dsamp']) ** m)
            
            ## crops 
            tr['crop_v'] = []
            for k in range(len(tr['use_mips'])):
                tr['crop_v'].append([0, 0, tr['crop_loss'][k][0], tr['crop_loss'][k][1], tr['crop_loss'][k][2]])

        self.tr = copy.deepcopy(tr)   
        self.num_v = len(tr['use_mips'])

    def make_requests(self):
        self.requests = {}
        self.pulled = {}
        self.pulled_b = {}
        self.pulled_c = {}
        self.output_info = {}
        self.output = {}
        self.output_b = {} 
        self.data = {}
        
        request = {}
        request['source_key'] = None
        request['target_shapes'] = np.array(self.tr['v_shapes'])
        request['use_mips'] = self.tr['use_mips']
        request['request_type'] = 'zarr'
        request['pull_fovs'] = []
        request['pull_shapes'] = []
        request['batch_size'] = self.tr['batch_size']
        request['is_input'] = True
        request['vox_sizes']= [np.array([1,1,1])]
        request['device'] = self.tr['device']
        
        request['buf_fac']  = 1
        request['buf_add']  = 0
        if self.tr['augment_rotate']:
             request['buf_fac'] *= (2 ** (1/2))
        if self.tr['jitter_frequency']:
            request['buf_add'] += (self.tr['jitter_magnitude'] * 2) ## allow for 2 std
           
        self.request_glob = request.copy()
        
                   
    def add_request(self, request_key, source_key, use_mips=None, 
                    v_shapes=None, buf_frac=None, channels=1, 
                    is_input=True, request_type='zarr',
                    request_data = None):
        request = copy.deepcopy(self.request_glob)
        request['source_key'] = source_key
        request['request_type'] = request_type
        request['is_input'] = is_input
        request['target_shapes'] = v_shapes if v_shapes is not None else request['target_shapes']
        request['use_mips'] = use_mips if use_mips is not None else request['use_mips']
        request['buf_fac'] = buf_frac if buf_frac is not None else request['buf_fac']
        request['vox_sizes'] = self.zm.datasets[source_key].vox_sizes
        
        if request['device'] == 'cpu':
            pin_memory=True
        else:
            pin_memory=False
            
        self.pulled_b[request_key] = []
        self.pulled[request_key] = []
        self.pulled_c[request_key] = []
        for m in range(len(request['use_mips'])):
            targ_shape = request['target_shapes'][m]
            add_buf = np.array([0, request['buf_add'], request['buf_add']]) 
            add_buf = add_buf * (2 / request['vox_sizes'][m])
            buf_shape =  targ_shape * np.array([1, request['buf_fac'], request['buf_fac']])
            buf_shape = buf_shape + add_buf
            fix_shape =  np.ceil(buf_shape) 
            request['pull_shapes'].append(fix_shape.copy().astype(int))
            request['pull_fovs'].append( (fix_shape * request['vox_sizes'][m]).astype(int))
            
            if request_type=='zarr':
                self.pulled_b[request_key].append(torch.empty(
                    (request['batch_size'], channels, int(targ_shape[0]), 
                     int(targ_shape[1]), int(targ_shape[2])), 
                    dtype=torch.float32, pin_memory=pin_memory, device=request['device']))
                self.pulled[request_key].append(torch.empty(
                    (1, channels, int(fix_shape[0]), int(fix_shape[1]), 
                     int(fix_shape[2])), dtype=torch.float32, 
                    pin_memory=pin_memory, device=request['device']))
                self.pulled_c[request_key].append(torch.empty(
                    (1, channels, int(targ_shape[0]), int(targ_shape[1]), 
                     int(targ_shape[2])), dtype=torch.float32, 
                    pin_memory=pin_memory, device=request['device']))
                
            else:
                self.pulled_b[request_key].append([''] * request['batch_size'])
                self.pulled[request_key].append('')
                self.pulled_c[request_key].append('')
    
            
        self.requests[request_key] = request.copy()

        ## record largest fov
        max_fov = np.zeros(3)
        for k in self.requests.keys():
            for m in self.requests[k]['pull_fovs']:
                max_fov = np.max(np.array([max_fov, m]),0)
        self.max_fov = max_fov.copy()      
   
        if request_type=='points':
            self.data['raw_pts'] = request_data
   
        
    def pull(self):
        self.tk.b('pull')
    
        for ri, k in enumerate(self.requests.keys()):
            request = self.requests[k]
            if request['is_input']:
                for vi in range(self.num_v):
                    pull_corner = np.floor(self.center_voxes[vi] - request['pull_shapes'][vi][-3:] / 2)
                    pull_win = np.array([pull_corner, pull_corner +  request['pull_shapes'][vi][-3:]]).astype(int)
                    if request['request_type']=='points':
                        pts = self.data['raw_pts'].copy() / request['vox_sizes'][vi]
                        is_in = np.where(self.pts_in_win(pts, pull_win ))[0] ## select in fov space
                        pts_in_fov = pts[is_in,:] - (pull_corner )
                        self.pulled[k][vi] =  pts_in_fov 
                    else:
                        self.pulled[k][vi][:] = 0
                        source_win = np.array([[0, 0, 0], self.zm.datasets[request['source_key']].shapes[vi]])
                        transfer = vt.block_to_window(pull_win, source_win)
                        if transfer['has_data']:
                            vt.array_to_tensor(self.pulled[k][vi], transfer['win'], 
                                       self.zm.datasets[request['source_key']].arr[vi], transfer['ch'])
                        else:
                                print('data window empty')
                        
        self.tk.e('pull')
            
                        

    def random_location(self, mode='full', key='raw', fov_win=None, ref_v=0):
        if key is None:
            keys = list(self.requests.keys())
            key = keys[0]
        
        vs = self.zm.datasets[key].vox_sizes
        if fov_win is None:
            fov = self.zm.datasets[key].fovs[ref_v]
            fov_win = np.array([[0, 0, 0], fov]).astype(int)
            
        if mode == 'valid':
           crop_fov = -(-self.max_fov // 2)
           cut_window = np.array([fov_win[0,:] + crop_fov, 
                                   fov_win[1,:] - crop_fov])
           window_exists = (cut_window[1,:] - cut_window[0,:]) > 0
           fov_win[:, window_exists] = cut_window[:, window_exists]
        
        fov_shape = fov_win[1,:]-fov_win[0,:]
        rand_shift = (fov_shape) * np.random.rand(3) 
        center = fov_win[0,:] + rand_shift
        
        self.center_voxes = []
        for mi in range(len(vs)):
            self.center_voxes.append(np.round(center / vs[mi]))
            
            
            
            
    def location_list(self, key, pos, mode='random'):
        """
        pos should be n x 3 array of positions in z, y, x mip0 voxel space
        """
        if mode=='random':
            pick = np.floor(np.random.rand() * pos.shape[0]).astype(int)
            center = pos[pick,:]

        vs = self.tr['v_vox_sizes']
        self.center_voxes = []
        for mi in range(len(vs)):
            self.center_voxes.append(np.round(center / vs[mi]))
        
        
    def batchify(self, batch_num=0, device='cpu'):
        self.tk.b('batchify')
        ## perform cropping
        self.crop_pulled() ## schould integrate with batch formation?
        
        keys = list(self.pulled.keys())
        for k in keys:
            if  self.requests[k]['request_type']=='points':
                for mi, m in enumerate(self.pulled_c[k]): 
                    self.pulled_b[k][mi][batch_num] = m.copy()
            else:
                for mi, m in enumerate(self.pulled_c[k]):
                    self.pulled_b[k][mi][batch_num:batch_num+1,:].copy_(m, non_blocking=False)
        self.tk.e('batchify')
        
### Display

    def show_key(self, key, show_plane=None):
        mip_num = len(self.pulled[key])
        fig = dsp.figure(mip_num,'show_key')
        
        shift_z = []
        for k in range(mip_num):
            shift_z.append((self.pulled[key][k].shape[-3]-self.pulled[key][-1].shape[-3])//2)
        
        midpoints = [round(self.pulled[key][k].shape[-3]/2) for k in range(len(self.pulled[key]))]
        if show_plane is None:
            showPlane = [midpoints[-1]]
        elif show_plane == 'all':
            showPlane = np.arange(self.pulled[key][-1].size(2))
            showPlane = np.concatenate((showPlane[::-1], showPlane[1:midpoints[-1]]))
                
        for i in showPlane:        
            i_input = []
            for k in range(len(self.pulled[key])):
                i_input.append(self.pulled[key][k][0, 0, i+shift_z[k]].detach().cpu().numpy())
            # Normalize
            for i,inp in enumerate(i_input):
                i_input[i] = inp/inp.max()
                
            # Display    
            if hasattr(fig.ax[0], 'im'):
                for a in range(3):
                    fig.ax[a].im.set_data(i_input[a])
                    fig.ax[a].set_title(f'v{a}')
            else:
                for a in range(3):
                    fig.ax[a].im = fig.ax[a].imshow(i_input[a], cmap='gray', vmax=1)
                    fig.ax[a].set_title(f'v{a}')
                    
            fig.fig.canvas.flush_events()
            fig.fig.canvas.draw()
            plt.pause(0.01)   
            
            
##### Processing
    def crop_center(self, arr, crop):
        """Crop x (B, C, D, H, W) to shape=(D,H,W) centered."""
        new_shape = np.array(arr.shape).copy()
        new_shape[-len(crop):] = new_shape[-len(crop):] - crop * 2
        slices = tuple(slice(c, c + s) for c, s in zip(crop, new_shape))
        arr2 = arr[slices]
        
        return arr2  
   
    
    def get_center_pts(self, old_pts, old_shape, new_shape):
                
        old_mid = (old_shape[-3:]) / 2
        new_mid = (new_shape[-3:]) / 2
        new_range = (new_shape[-3:]) / 2
        pts = old_pts - old_mid
        is_in = ((pts >= -new_range) & (pts < new_range)).sum(1) == pts.shape[1]
        pts = pts[is_in,:]
        pts = pts + new_mid
        return pts
    
    def pts_in_win(self, pt, win_pt_space):
        # upper inclusive
        #win_pt_space = win - 0.5
        is_in = pt[:,0] *  0 + 1
        for d in range(pt.shape[1]):
            is_in = is_in * (pt[:,d] >= win_pt_space[0,d]) * (pt[:,d] < win_pt_space[1,d])
        return is_in
    
    def center_window(self, s1,s2):
        s1 = np.array(s1)
        s2 = np.array(s2)
        lower = (s1 - s2) // 2
        cap = lower + s2
        return np.array([lower, cap])
    
    def get_center(self, arr, target_shape):
        """Crop x (B, C, D, H, W) to shape=(D,H,W) centered."""
        
        new_shape = np.array(arr.shape).copy()
        new_shape[-len(target_shape):] = target_shape
        dif = np.array(arr.shape) - new_shape
        crop = dif // 2
        
        slices = tuple(slice(c, c + s) for c, s in zip(crop, new_shape))
        arr2 = arr[slices]
        
        return arr2  
        
    
    def upsample_array(self, arr, scale_factors = [1, 2, 2]):
              
        # Apply repeat_interleave along each axis in reverse order
        ndim = arr.ndim
        scale_factors = np.array(scale_factors).astype(int).flatten()
        nscale = len(scale_factors)
        for d in range(nscale):
            if type(arr) == 'torch.tensor':
                arr = arr.repeat_interleave(scale_factors[d], dim= ndim-nscale+d)  # 
            else:
                arr = arr.repeat(scale_factors[d], axis= ndim-nscale+d)  # 
       
        return arr
                            
    def downsample_3d_kernel(self, arr, dsamp, method='mean'):
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
                arr2 = arr.reshape(new_shape).mean(axis=(1, 3, 5))
            case 'max':
                arr2 = arr.reshape(new_shape).max(axis=(1, 3, 5))
            case 'min':
                arr2 = arr.reshape(new_shape).min(axis=(1, 3, 5))
                
        return arr2


    def affinities(self, key_in, key_out='aff', merge=True, shift = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]):
        """
        Assumes second dim is a channel of shape 1 that is 
        available for stacking affinities
        
        """
        
        for vi in range(self.num_v):
            affm, affd = self.calc_affinities(self.pulled[key_in][vi], shift[vi])
            if merge:
                self.pulled[key_out][vi][:] = affm
            else:
                self.pulled[key_out][vi][:] = affd
            
    def calc_affinities(self, m, shift):
            
            affd = m.new_empty((1,3, m.shape[2], m.shape[3], m.shape[4]))
            affm = m.new_empty((1,1, m.shape[2], m.shape[3], m.shape[4]))
            m_stable = m.clone() 
            for di in range(3):
                d  = -(di + 1)
                dif = m_stable -  torch.roll(m, shifts=shift[di], dims=-(di+1))
                affd[0,di,:] = dif == 0 ## add affinity to channel
                
                ## reflect
                affd[0,di,:] = affd[0,di,:] * torch.roll(affd[0,di,:], shifts=-shift[di], dims=d)
                affd[0,di,:] = affd[0,di,:]  * (m[0,0,:] > 0)
                
                ## negate invalid
                low1 = np.array((1,di,0, 0, 0),int)
                high1 = np.array((1,di,m.shape[2], m.shape[3], m.shape[4]),int)
                low2 = low1.copy()
                high2 = high1.copy()
                low2[d] = high1[d]
                high2[d] = low1[d]
                slice1 = tuple(slice(c, s) for c, s in zip(low1, high2))
                slice2 = tuple(slice(c, s) for c, s in zip(low2, high1))
                affd[slice1] = 0
                affd[slice2] = 0
            affm = affd.prod(1) 
            return affm, affd
    
    def relabel(self, key_in='labels', key_out='relabels', lookup=None  ):
        
        if lookup is None:
            for vi in range(self.num_v):
                self.pulled[key_out][vi][:] = self.pulled[key_in][vi]
                
            for vi in range(self.num_v):
                self.pulled[key_out][vi][:] = lookup[self.pulled[key_in][vi]]
    
    def aff_weights(self, key_in, key_out='aff_wieghts'):
        print('affinity weights not done')
       
        for mi, m in enumerate(self.pulled[key_in]):
            label_shape = np.array(self.pulled[key_in][mi].shape)
            vox_num = label_shape[-3] * label_shape[-2] * label_shape[-1]
            vox_sum = self.pulled[key_in][mi][0,-1,:,:,:].sum()
            self.pulled[key_out][mi][:]  =   self.pulled[key_in][mi][0,-1,:,:,:] * 0 + 1 / vox_sum
        
            

#### Augments
    
  
    def binarize(self, key):
        for mi, m in enumerate(self.pulled[key]):
                self.pulled[key][mi] = m > 0
            
    def normalize(self, keys, constant=255):

        if constant == 'range':      
            for k in keys:
                v_max = 0
                v_min = 10**10
                for m in self.pulled[k]:
                    v_max = np.max([v_max,m.max()])
                    v_min = np.min([v_max,m.min()])
        else:
            v_max = constant
            v_min = 0
            
        for k in keys:
            for mi, m in enumerate(self.pulled[k]):
                self.pulled[k][mi] = (m - v_min) / (v_max - v_min)
                
    def transpose_half(self):
        if np.random.rand() < .5:
            keys = self.pulled.keys()
            for k in keys:
                if self.requests[k]['request_type']=='points':
                    for mi, m in enumerate(self.pulled[k]):
                        self.pulled[k][mi][:,[-1,-2]] = self.pulled[k][mi][:,[-2,-1]]
                else:
                    for mi, m in enumerate(self.pulled[k]):
                        self.pulled[k][mi] = torch.transpose(m, -2, -1)
                    
    def section_augment_info(self, key, frequency, magnitude, mag_type = 'sigma'):
    
            aug_seg = {}
            aug_seg['frequency'] = frequency
            aug_seg['magnitude'] = magnitude
            aug_seg['vox_sizes'] = np.array(self.zm.datasets[key].vox_sizes)
            aug_seg['shapes'] = np.array(self.requests[key]['pull_shapes'])
            
            aug_seg['full_rat'] = np.zeros((len(aug_seg['vox_sizes']),3))
            for vi in range(len(aug_seg['vox_sizes'])):
                aug_seg['full_rat'] [vi,:] = aug_seg['vox_sizes'][-1,:] /  aug_seg['vox_sizes'][vi,:] 
            aug_seg['vox_rat'] = aug_seg['full_rat'] [:,-1]
            
            aug_seg['max_fov'] = self.max_fov
            fov_max_y = aug_seg['max_fov'][1] 
           
            z_num = aug_seg['shapes'][:,0]
            max_z = z_num.max()
            z_offset = (max_z -z_num)//2
           
            if mag_type == 'sigma':
                aug_seg['r_sigs'] = np.abs(np.random.randn(max_z, 2)) * magnitude
            else:
                aug_seg['r_sigs'] = np.abs(np.random.rand(max_z, 2)) * magnitude * 2 - magnitude
            aug_seg['r_hit_full'] = np.where(np.random.rand(max_z) <= frequency)[0]
            aug_seg['r_hits'] = []
            for r in range(len(z_offset)):
                aug_seg['r_hits'].append(aug_seg['r_hit_full'] - z_offset[r])
                aug_seg['r_hits'][-1][np.where(aug_seg['r_hits'][-1] >= z_num[r])[0]] = -1   
            
            return aug_seg
           
        
    def gaussian_fft_filter_tensor(self, x, sigma_h, sigma_w=None, pad_mult=5):
    
        # x: (N, C, H, W), float32
        if sigma_w is None: sigma_w = sigma_h
        dev = x.device
        dt = x.dtype
        
        # Pad to reduce circular artifacts: ~3σ on each spatial edge
        pad_vec = np.zeros(x.ndim, int)
        pad_vec[-2:] = np.ceil(np.array((sigma_h, sigma_w)) * pad_mult + 0.5).astype(int)
        old_shape = np.array(x.shape, int)
        pad_shape = old_shape + pad_vec
        slices = tuple(slice(c, c + s) for c, s in zip(pad_vec, old_shape))
        
        xpad = x.new_zeros(tuple(pad_shape)) + x.mean()
                
        xpad[slices] = x
                
        # Xf is from rfft2(xpad): shape (..., H, W_r)
        Xf = torch.fft.rfft2(xpad)              # (..., H, W_r), W_r = W//2 + 1
        
        H  = xpad.shape[-2]
        W  = xpad.shape[-1]
        fy = torch.fft.fftfreq(H,  d=1.0, device=dev, dtype=dt)[:, None]     # (H, 1)
        fx = torch.fft.rfftfreq(W, d=1.0, device=dev, dtype=dt)[None, :]     # (1, W_r)
        
        # Gaussian frequency response on the rFFT grid: (H, W_r)
        Hf = torch.exp(-(2*torch.pi**2) * ((sigma_h**2)*(fy**2) + (sigma_w**2)*(fx**2)))
        
        # Broadcast Hf up to Xf's ndim (prepend singleton dims)
        while Hf.ndim < Xf.ndim:
            Hf = Hf.unsqueeze(0)
        
        Yf   = Xf * Hf
        ypad = torch.fft.irfft2(Yf, s=(H, W))
            
        # Crop back
        y = ypad[slices]
        
        return y
    
    def gaussian_fft_filter_numpy(self, x, sigma_h, sigma_w=None, pad_mult=5):
    
        # x: (N, C, H, W), float32
        if sigma_w is None: sigma_w = sigma_h
        
        # Pad to reduce circular artifacts: ~3σ on each spatial edge
        pad_vec = np.zeros(x.ndim, int)
        pad_vec[-2:] = np.ceil(np.array((sigma_h, sigma_w)) * pad_mult + 0.5).astype(int)
        old_shape = np.array(x.shape, int)
        pad_shape = old_shape + pad_vec
        slices = tuple(slice(c, c + s) for c, s in zip(pad_vec, old_shape))
        
        xpad = np.zeros(tuple(pad_shape)) + x.mean()
                
        xpad[slices] = x
                    
        # Xf is from rfft2(xpad): shape (..., H, W_r)
        Xf = np.fft.rfft2(xpad)              # (..., H, W_r), W_r = W//2 + 1
        
        H  = xpad.shape[-2]
        W  = xpad.shape[-1]
        fy = np.fft.fftfreq(H,  d=1.0)[:, None]     # (H, 1)
        fx = np.fft.rfftfreq(W, d=1.0)[None, :]     # (1, W_r)
        
        # Gaussian frequency response on the rFFT grid: (H, W_r)
        Hf = np.exp(-(2*torch.pi**2) * ((sigma_h**2)*(fy**2) + (sigma_w**2)*(fx**2)))
        
        # Broadcast Hf up to Xf's ndim (prepend singleton dims)
        while Hf.ndim < Xf.ndim:
            Hf = Hf.unsqueeze(0)
        
        Yf   = Xf * Hf
        ypad = np.fft.irfft2(Yf, s=(H, W))
            
        # Crop back
        y = ypad[slices]
                
        return y
        
    
    def black_blobs(self, key, frequency, magnitude):
        self.tk.b('augment')

        if frequency:
            ag = self.section_augment_info(key, frequency, magnitude)
            source_fov = np.ceil(ag['max_fov']).astype(int)
            small_fov = np.array([1, 128, 128])
            scale_factor = np.ceil(source_fov / small_fov)+1
            big_fov = (small_fov * scale_factor).astype(int)
            blob_field_big = np.zeros(tuple(big_fov[1:]))

            for zi in range(len(ag['r_hit_full'])):
                sigma = [ag['r_sigs'][zi, 0], ag['r_sigs'][zi, 1] ] 
                blob_field = np.random.rand(small_fov[1], small_fov[2])
                blob_field= self.gaussian_fft_filter_numpy( blob_field, sigma[0], sigma[1], pad_mult=0)
                blob_field_big[:] = self.upsample_array(blob_field, scale_factor[1:])
                blob_field_big[:] = self.gaussian_fft_filter_numpy( blob_field_big, scale_factor[1], scale_factor[2], pad_mult=0)
                blob_mask = np.expand_dims((blob_field_big > .5).astype(np.float32), 0)
    
                for ai in range(self.num_v):
                    hit_z = ag['r_hits'][ai][zi] 
                    if hit_z >= 0:
                        blob_scaled = self.downsample_3d_kernel(blob_mask, ag['vox_sizes'][ai], method='mean')
                        blob_pull = self.get_center(arr=blob_scaled, target_shape=ag['shapes'][ai,1:])
                        blob_tensor = torch.from_numpy(blob_pull).to(self.pulled[key][ai])
                        self.pulled[key][ai][0, 0, hit_z, :, :] = self.pulled[key][ai][0, 0, hit_z, :, :] - blob_tensor   
             
                for ai in range(len(self.pulled[key])):
                    self.pulled[key][ai][torch.where(self.pulled[key][ai]<0)] = 0
        self.tk.e('augment')
        
        
    def black_corner(self, key, frequency):
        self.tk.b('augment')

        if frequency:
            print('augmentation broken')
            ag = self.section_augment_info(key, frequency, 1)
            source_fov = np.ceil(ag['max_fov']).astype(int)

            for zi in range(len(ag['r_hit_full'])):
                mids = np.random.rand(2) * source_fov[1:]
                edges = (np.random.rand(2) < .5).astype(float) * source_fov[1:]
                y_rand = np.sort(np.floor([mids[0], edges[0]]).astype(int))
                x_rand = np.sort(np.floor([mids[1], edges[1]]).astype(int))
               
              
                for ai in range(self.num_v):
                    hit_z = ag['r_hits'][ai][zi] 
                    if hit_z >= 0:
                        max_shape = (source_fov / ag['vox_sizes'][ai]).astype(int)
                        targ_shape = self.pulled[key][ai].shape[-3:]
                        yrs = y_rand / ag['vox_sizes'][ai][1]
                        xrs = x_rand / ag['vox_sizes'][ai][2]
                        off_y = (max_shape[1] - targ_shape[1]) // 2
                        off_x = (max_shape[2] - targ_shape[2]) // 2
                        yrs = (yrs - off_y).astype(int)
                        xrs = (xrs - off_x).astype(int)
                        y0, y1 = np.clip(yrs, 0, targ_shape[1])
                        x0, x1 = np.clip(xrs, 0, targ_shape[2])
                        self.pulled[key][ai][0, 0, hit_z, y0:y1, x0:x1] = 0  
             
                for ai in range(len(self.pulled[key])):
                    self.pulled[key][ai][torch.where(self.pulled[key][ai]<0)] = 0
        self.tk.e('augment')
        
    def random_rotate_z(self):
        self.tk.b('augment')
        angle_deg = np.random.uniform(-180, 180)
        for k in self.pulled:
            if self.requests[k]['request_type']=='points':
                angle_rad = angle_deg / 180 * np.pi * -1
                rot_mat = np.array([
                    [np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad), np.cos(angle_rad)]
                ]) 
                for mi, m in enumerate(self.pulled[k]):
                    rad = (self.requests['raw_pts']['pull_shapes'][mi][-2:]-1 )/2
                    yx = self.pulled[k][mi][:,[-2,-1]] - rad
                    yx = np.dot(yx, rot_mat) + rad
                    self.pulled[k][mi][:,[-2,-1]] = yx
            else:
                
                for mi, m in enumerate(self.pulled[k]):
                    rotate_dim = (m.ndim - 2, m.ndim -1)
                    arr = nd_rotate(
                        m.detach().cpu().numpy(),
                        angle=angle_deg,
                        axes=rotate_dim,  # rotate in the XY plane
                        reshape=False,
                        order=1,      # bilinear interpolation
                        mode='constant',
                        cval = 0
                    ).astype('float32')
                    self.pulled[k][mi][:] = torch.from_numpy(arr).to(m.device)
        self.tk.e('augment')
        
    def crop_pulled(self):
        
        for k in self.requests:
            if self.requests[k]['request_type']=='points':
                for mi, m in enumerate(self.pulled[k]):
                    self.pulled_c[k][mi] =  self.get_center_pts(m, 
                                                                 self.requests[k]['pull_shapes'][mi],
                                                                 self.requests[k]['target_shapes'][mi])
            else:
                for mi, m in enumerate(self.pulled[k]):
                    self.pulled_c[k][mi] =  self.get_center(m, self.requests[k]['target_shapes'][mi])
        
        
    def destig(self, key, frequency, magnitude):
        """
        Add assymetric gaussian blur to a subset of sections 
        Apply the same destig to corresponding plane in multiscale set of arrays
        """
        self.tk.b('augment')
        if frequency:
            ag = self.section_augment_info(key, frequency, magnitude)
            
            for zi in range(len(ag['r_hit_full'])):
                for ai in range(len(ag['vox_sizes'])):
                    if ag['r_hits'][ai][zi] >= 0:
                        sigma = ag['r_sigs'][zi, :] / ag['vox_sizes'][ai][1:]
                        
                        self.pulled[key][ai][0, 0, ag['r_hits'][ai][zi], :, :] \
                            = self.gaussian_fft_filter_tensor(
                            self.pulled[key][ai][0, 0, ag['r_hits'][ai][zi], :, :],
                            sigma[0], sigma[1])
                      
                        # arr =  self.pulled[key][ai][0, 0, ag['r_hits'][ai][zi], :, :].detach().numpy()
                        # arr2 = sk.filters.gaussian(
                        #     arr, 
                        #     sigma=sigma,
                        #     mode='reflect')
                        # self.pulled[key][ai][0, 0, ag['r_hits'][ai][zi], :, :] = torch.from_numpy(arr2)
        self.tk.e('augment')
                
    def jitter(self, key, frequency, magnitude):
        if frequency:
            ag = self.section_augment_info(key, frequency, magnitude, mag_type = 'max')
            
            jitters = []
            jitter_buf = []
            for ai in range(len(ag['vox_sizes'])):
                is_shift = np.where(ag['r_hits'][ai]>0)[0]
                shift_planes = ag['r_hits'][ai][is_shift]
                shifts_p = np.round(ag['r_sigs'][is_shift,:] / ag['vox_sizes'][ai,1:])
                shifts_p = shifts_p * (np.round(np.random.rand(shifts_p.shape[0], 2)) * 2 -1)
                num_planes = self.requests[key]['pull_shapes'][ai][0]
                jitters.append(np.zeros((num_planes,2),int))
                jitters[-1][shift_planes,:] = shifts_p
                jitter_buf.append(np.array([jitters[-1].min(0), jitters[-1].max(0)]))  
            
            self.jit['shifts'] = jitters.copy()
            self.jit['do'] = 1
            self.jit['buf'] = jitter_buf.copy()
            self.jit['shift_planes'] = shift_planes 
            
            for ri, k in enumerate(self.requests.keys()):
                request = self.requests[k]
                for vi in range(self.num_v):
                    if request['request_type']=='points':
                        for sp in self.jit['shift_planes']:
                            is_p = np.where(self.pulled[k][vi][:,0] == sp)
                            self.pulled[k][vi][is_p,1]  +=  self.jit['shifts'][vi][sp][0]
                            self.pulled[k][vi][is_p,2]  +=  self.jit['shifts'][vi][sp][1]
                        print('WARNING, jitter points not tested')
                    else:
                        for sp in self.jit['shift_planes']:
                            self.pulled[k][vi][0,0,sp,:] = torch.roll(self.pulled[k][vi][0,0,sp,:] , self.jit['shifts'][vi][sp][0], dims=-1)
                            self.pulled[k][vi][0,0,sp,:] = torch.roll(self.pulled[k][vi][0,0,sp,:] , self.jit['shifts'][vi][sp][1], dims=-2)
                  
        else:
            self.jit['do'] = 0         
        
        
       

    # def jitter_pull(self):
    
    #     jit_buf = []
    #     buf_shape = []
    #     pull_corner = []
    #     pull_win = []
    #     buf_block = []
    #     first_key = next(iter(self.requests)) ## 'WARNING. Using default key for jitter shape'
    #     for vi in range(self.num_v):
    #         jit_buf.append(self.jit['buf'][vi])
    #         bigger = np.concatenate((np.zeros(1), jit_buf[-1][1,:] - jit_buf[-1][0,:])).astype(int)
    #         buf_shape.append(self.requests[first_key]['pull_shapes'][vi] + bigger)
    #         pull_corner.append(np.round(self.center_voxes[vi] - buf_shape[-1] / 2))
    #         pull_win.append(np.array([pull_corner[-1], pull_corner[-1] +  buf_shape[-1]]).astype(int))
    #         buf_block.append(torch.empty(tuple(buf_shape[vi])))
    
    #     for ri, k in enumerate(self.requests.keys()):
    #         request = self.requests[k]
    #         if request['is_input']:
    #             for vi in range(self.num_v):
    #                 if request['request_type']=='points':
    #                     pts = self.data['raw_pts'].copy() / request['vox_sizes'][vi]
    #                     is_in = np.where(self.pts_in_win(pts, pull_win ))[0] ## select in fov space
    #                     pts_in_fov = pts[is_in,:] - (pull_corner )
    #                     self.pulled[k][vi] =  pts_in_fov 
    #                     print('WARNING, points were not jittered')
    #                 else:
    #                     buf_block[vi][:] = 0
    #                     source_win = np.array([[0, 0, 0], self.zm.datasets[request['source_key']].shapes[vi]])
    #                     transfer = vt.block_to_window(pull_win[vi], source_win)
    #                     if transfer['has_data']:
    #                         vt.array_to_tensor(buf_block[vi], transfer['win'], 
    #                                    self.zm.datasets[request['source_key']].arr[vi], transfer['ch'])
    #                         for sp in self.jit['shift_planes']:
    #                             buf_block[vi][sp,:] = torch.roll(buf_block[vi][sp,:], self.jit['shifts'][vi][sp][0], dims=0)
    #                             buf_block[vi][sp,:] = torch.roll(buf_block[vi][sp,:], self.jit['shifts'][vi][sp][1], dims=1)
    #                         self.pulled[k][vi][:] = buf_block[vi][:,
    #                             -jit_buf[vi][0,0]:-jit_buf[vi][0,0] + request['pull_shapes'][vi][1], 
    #                             -jit_buf[vi][0,1]:-jit_buf[vi][0,1] + request['pull_shapes'][vi][2]] 
                                
    #                     else:
    #                             print('data window empty')
                    
    
    
    ## Model manipulations
    
    def wack_weights():
        print('fix wack_weights')
        # if 0:
        #     with torch.no_grad():
        #             print("Before:", model.v[0].conv0_0.conv1.weight.view(-1)[0])
        #             #model.v[0].apply(manual_reinit)
        # elif 0:
        #     with torch.no_grad():
        #         for p in model.v[0].parameters():
        #             p.add_(.1 * torch.randn_like(p))
        #     print("After:", model.v[0].conv0_0.conv1.weight.view(-1)[0])
        # else:
        #     optimizer = torch.optim.AdamW(lr=0.7e-4, params=model.parameters(), weight_decay=0.7e-4)
    
        # tr['wack_weights'] = 1
    
    
    def manual_reinit(m):
        print('fix manual_reinit')
        # if isinstance(m, nn.Conv3d):
        #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     if m.bias is not None:
        #         nn.init.zeros_(m.bias)
        # elif isinstance(m, nn.BatchNorm3d):
        #     nn.init.ones_(m.weight)
        #     nn.init.zeros_(m.bias)
        # elif isinstance(m, nn.Linear):
        #     nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #     if m.bias is not None:
        #         nn.init.zeros_(m.bias)
        # elif isinstance(m, nn.Module):
        #     # This ensures nested blocks like conVGGBlock are reached
        #     for child in m.children():
        #         manual_reinit(child)


    def make_loss_masks(self):
        
        self.loss_masks = [None] * self.num_v
        
        if self.tr['loss_mask_type'] == 'hard':
            for vi in range(self.num_v):
                loss_masks = self.cropping_mask(self.tr['v_shapes'][vi], self.tr['v_crop'][vi])
                self.loss_masks[vi] = torch.from_numpy(loss_masks, device=self.tr['device'])
        elif self.tr['loss_mask_type'] == 'soft':
            for vi in range(self.num_v):
                self.loss_masks[vi] = torch.from_numpy(self.cvv.masks[vi]).to(self.tr['device'])
                            
    def masked_bce_with_logits(self, logits, target, mask=None):
        """
        logits, target, mask: broadcastable to (B, 1, D, H, W) or (B, C, D, H, W)
        mask should be float {0,1}. Booleans are fine; they’ll be cast.
        """
        
        if mask is None:
            mask = torch.ones_like(logits, dtype=logits.dtype, device=logits.device)
        else:
            mask = mask.to(dtype=logits.dtype, device=logits.device)
            while mask.ndim < logits.ndim:
                mask = mask.unsqueeze(0)
        
        B, C = logits.shape[:2]
        dims = tuple(range(2, target.ndim))  # (2,3,4) for 5D tensors
        pos = (target.float() * mask).sum(dim=dims) / mask.sum(dim=dims).clamp_min(1.0)
        pos = pos.clamp(1e-6, 1 - 1e-6)      
        pw  = ((1 - pos) / pos).to(dtype=logits.dtype, device=logits.device)
        pw = pw.view(B, C, *([1] * (logits.ndim - 2)))   # -> (B, C, 1, 1, 1)
        
        per_voxel = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction='none', pos_weight=pw)
        
        masked = per_voxel * mask
        loss = masked.sum() / mask.sum().clamp_min(1.0)
        return loss
        

    def cropping_mask(self, t, crop):
        # t: (B, C, D, H, W). crop is [b,c,cz,cy,cx] in your code; use spatial entries.
        _, _, D, H, W = t.shape
        cz, cy, cx = crop[-3], crop[-2], crop[-1]
        m = torch.zeros((t.shape[0], 1, D, H, W), device=t.device, dtype=t.dtype)
        m[..., cz:D-cz, cy:H-cy, cx:W-cx] = 1
        return m

    
    def check_valid(self, modules):
        
        self.cvv = CheckValid(modules)
        self.cvv.check_shapes(self.tr)  
        self.cvv.recommendation = self.cvv.recommend()
                
    def get_losses(self, preds, target_key='aff'):
        #Choose valid region
        self.tk.b('loss')
        tr = self.tr
        if tr['true_crop_for_loss']:
            use_preds = []
            use_target = []
            use_masks = []
            for vi in  range(len(tr['use_mips'])):
                use_preds.append(crop_tensor(preds[vi], tr['crop_v'][vi]))
                use_target.append(crop_tensor(self.pulled_b['aff'][vi], tr['crop_v'][vi]))
                if self.loss_masks[vi] is not None:
                    use_masks.append(crop_tensor(self.loss_masks[vi], tr['crop_v'][vi]))
                else: 
                    use_masks.append(None)
        else:
            use_preds = preds
            use_target = self.pulled_b[target_key]
            use_masks = self.loss_masks

        loss_vals = []
        for vi, pred in enumerate(use_preds):
            if isinstance(pred, list):
                supervivised_loss = []
                for pi, pr in enumerate(pred):
                    supervivised_loss.append(
                        self.masked_bce_with_logits(logits=pr[:, 0:1 , :, :, :], 
                                                     target=use_target[vi], 
                                                     mask=use_masks[vi],))
                loss_vals.append( sum(l * w for l,w in zip(supervivised_loss, tr['scale_losses']))) 
            else:    
                      loss_vals.append(
                          self.masked_bce_with_logits(logits=pred[:, 0:1 , :, :, :], 
                                                     target=use_target[vi], 
                                                     mask=use_masks[vi],))
        loss_val= sum(l * w for l,w in zip(loss_vals, tr['v_loss_scale'])) 
        
        self.tk.e('loss')
        return loss_val, loss_vals
    
    
def crop_tensor(raw_tensor, ct):
    
    if isinstance(raw_tensor, list):
        cropped_tensor = []
        for rt in raw_tensor:
            if rt.ndim < len(ct):
                ct = ct[-raw_tensor.ndim:]
            end = torch.tensor(rt.shape) - torch.tensor(ct)
            cropped_t = rt[tuple(slice(s, e) for s, e in zip(ct, end))]       
            cropped_tensor.append(cropped_t)
    else:
        if raw_tensor.ndim < len(ct):
            ct = ct[-raw_tensor.ndim:]
        end = torch.tensor(raw_tensor.shape) - torch.tensor(ct)
        cropped_tensor = raw_tensor[tuple(slice(s, e) for s, e in zip(ct, end))]     

    return cropped_tensor

class CheckValid():
    """
    Use artificial boundary around dummy volumes to calculate validity masks
    for the desired volume shapes that will be run through the inputed lmodule
    list
    """
    
    def __init__(self, modules, fill_with=1):
        self.modules_in = modules
                
        ## make dummy network for calculating the validity of voxels    
        self.test_model = torch.nn.ModuleList()
        for v in self.modules_in: ## duplicate model
            self.test_model.append(copy.deepcopy(v))
        with torch.no_grad(): ## Set all weights
            for m in self.test_model:
                for param in m.parameters():
                   param.fill_(fill_with)
        
        
        
    def check_shapes(self, tr, border_val=1):    
        
        self.param = {'crop_validity': np.array([.68, .95, .999])}
        self.maps = []
        self.masks = []
        for mi, m in enumerate(self.test_model):
           
            self.maps.append({})
            ## plot validit from edges
            self.change_conv3d_padding_mode(m, 'replicate') ## change padding mode
            t_shape = np.array(tr['v_shapes'][mi])
            self.maps[mi]['test_shape'] = t_shape.copy()
            pad_shape = t_shape + 2
            border_tensor = torch.ones(1, 1, pad_shape[0], pad_shape[1], pad_shape[2]) * border_val
            border_tensor[0, 0, 1:1+t_shape[0], 1:1+t_shape[1], 1:1+t_shape[2]] = 0           
            border_tensor = border_tensor.repeat(1, tr['input_channels'][mi], 1, 1, 1)
            border_tensor = border_tensor.to(tr['device'])
            border_out, feats = self.test_model[mi](border_tensor)
            border_map = border_out[-1][0, :].detach().cpu().squeeze().numpy()  # if deep supervision
            border_map = border_map[1:-1,1:-1,1:-1]
            self.maps[mi]['min_border_val'] = border_map.min()
            self.maps[mi]['max_border_val'] = border_map.max()
            border_map = border_map - border_map.min()
            if border_map.max():
                border_map = border_map / border_map.max()            
            border_map = 1 - border_map            
            border_samp_yx = border_map[t_shape[0]//2,:,:]
            border_samp_zx = border_map[:, t_shape[1]//2,:]

            rf_trace = []
            bs = np.array(border_map.shape) // 2
            rf_trace.append(border_map[0:bs[0], bs[1], bs[2]])
            rf_trace.append(border_map[bs[0], 0:bs[1], bs[2]])
            rf_trace.append(border_map[bs[0], bs[1], 0:bs[2]])

            has_frac = self.param['crop_validity'] 
            crops = []
            rf_min = []
            for rf in rf_trace:
                if rf.shape[0]:
                    rf = rf / rf.max()
                    rf_min.append(rf.min())
                    rf_cumsum = np.cumsum(rf) / rf.sum()
                    crops.append([int(np.where(rf_cumsum >= f)[0][0])
                                  for f in has_frac])
                else:
                    crops.append([0, 0, 0])
            
            self.masks.append(border_map.copy())
            self.maps[mi]['border_section_yx'] = border_samp_yx.copy()
            self.maps[mi]['border_section_zx'] = border_samp_zx.copy()
            self.maps[mi]['rf_mins'] = rf_min.copy()
            self.maps[mi]['rf'] = rf_trace.copy()
            self.maps[mi]['crop'] = crops.copy()

    def show_results(self):
       
            fig_test = dsp.figure(num_ax=9, fig_name='test_validity')
            for mi, m in enumerate(self.maps):
                fig_test.ax[mi].imshow(m['border_section_zx'], cmap='grey',vmax=1)
                fig_test.ax[mi].set_title(f'v{mi} validity map')
                fig_test.ax[mi+3].plot(m['rf'][0])
                fig_test.ax[mi+3].set_title(f'v{mi} z rf')
                fig_test.ax[mi+6].plot(m['rf'][1])
                fig_test.ax[mi+6].set_title(f'v{mi} y rf')
            fig_test.update()
            
        
            
    def recommend(self):
            
         for mi, m in enumerate(self.maps):
             print(
             f"Module_{mi}: tested shape {m['test_shape']}\n"
             f"Validity map values ranged from {m['min_border_val']:0.2f} to {m['max_border_val']:0.2f}\n"
             f"In yx, crop {m['crop'][1]} for validity fraction {self.param['crop_validity']}\n"
             f"In  z, crop {m['crop'][0]} for validity fraction {self.param['crop_validity']}\n"
             f" \n"
             )

    def change_conv3d_padding_mode(self, module, new_padding_mode='replicate'):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Conv3d):
                # Rebuild the Conv3d layer with new padding mode
                new_conv = torch.nn.Conv3d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=new_padding_mode
                )
                # Copy weights and bias
                new_conv.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_conv.bias.data = child.bias.data.clone()

                # Replace in parent module
                setattr(module, name, new_conv)
            else:
                # Recurse into submodules
                self.change_conv3d_padding_mode(child, new_padding_mode)
                
                
    