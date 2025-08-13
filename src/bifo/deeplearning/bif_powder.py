# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 09:58:28 2025

@author: jlmorgan
"""
import numpy as np
from dataclasses import dataclass
import bifo.tools.voxTools as vt
import torch
from scipy.ndimage import rotate as nd_rotate
import copy
import skimage as sk
from  skimage.transform import AffineTransform as affine 
import torchvision
import bifo.tools.display as dsp
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


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
        if self.tr['augment_rotate']:
             request['buf_fac'] = 2 ** (1/2)
        else:
            request['buf_fac'] = 1
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
        
        
        if request_type=='zarr':
            pin_memory = 1
        else:
            pin_memory = 0
            
        self.pulled_b[request_key] = []
        self.pulled[request_key] = []
        self.pulled_c[request_key] = []
        for m in range(len(request['use_mips'])):
            targ_shape = request['target_shapes'][m]
            buf_shape =  targ_shape * np.array([1, request['buf_fac'], request['buf_fac']])
            fix_shape =  np.round(buf_shape) 
            request['pull_shapes'].append(fix_shape.copy().astype(int))
            request['pull_fovs'].append( np.astype(fix_shape * request['vox_sizes'][m],int))
            
            if pin_memory:
                self.pulled_b[request_key].append(torch.empty(
                    (request['batch_size'], channels, int(targ_shape[0]), int(targ_shape[1]), int(targ_shape[2])), dtype=torch.float32, pin_memory=True))
                self.pulled[request_key].append(torch.empty(
                    (1, channels, int(fix_shape[0]), int(fix_shape[1]), int(fix_shape[2])), dtype=torch.float32, pin_memory=True))
                self.pulled_c[request_key].append(torch.empty(
                    (1, channels, int(targ_shape[0]), int(targ_shape[1]), int(targ_shape[2])), dtype=torch.float32, pin_memory=True))
                
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
        
        if self.jit['do']:
            self.jitter_pull()
        else:
            for ri, k in enumerate(self.requests.keys()):
                request = self.requests[k]
                if request['is_input']:
                    for vi in range(self.num_v):
                        pull_corner = np.round(self.center_voxes[vi] - request['pull_shapes'][vi] / 2)
                        pull_win = np.array([pull_corner, pull_corner +  request['pull_shapes'][vi]]).astype(int)
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
        
        
    def batchify(self, batch_num=0, device='cpu'):
        
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
    
    

    def affinities(self, key_in, key_out='aff', merge=True, shift = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]):
        """
        Assumes second dim is a channel of shape 1 that is 
        available for stacking affinities
        treats affinities as channels and adds a 4th merged affinity channel
        """
        
        for mi, m in enumerate(self.pulled[key_in]):
            affd = torch.empty((1,4, m.shape[2], m.shape[3], m.shape[4]))
            ndim = m.ndim
            m_stable = m.clone() 
            for d in range(3):
                dif = m_stable -  torch.roll(m, shifts=shift[mi][d], dims=-d)
                affd[0,d,:] = dif == 0 ## add affinity to channel
                
                ## reflect
                affd[0,d,:] = affd[0,d,:] * torch.roll(affd[0,d,:], shifts=-shift[mi][d], dims=-d)
                affd[0,d,:] = affd[0,d,:]  * (m > 0)
                
                ## negate invalid
                low1 = np.array((1,d,0, 0, 0),int)
                high1 = np.array((1,d,m.shape[2], m.shape[3], m.shape[4]),int)
                low2 = low1.copy()
                high2 = high1.copy()
                low2[-d] = high1[-d]
                high2[-d] = low1[-d]
                slice1 = tuple(slice(c, s) for c, s in zip(low1, high2))
                slice2 = tuple(slice(c, s) for c, s in zip(low2, high1))
                affd[slice1] = 1
                affd[slice2] = 1
            affd[:,3,:,:,:] = affd[:,0:3,:,:,:].prod(1) 
            self.pulled[key_out][mi][:] = affd
            
            
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
                    
    def section_augment_info(self, key, frequency, magnitude):
    
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
           
            aug_seg['r_sigs'] = np.abs(np.random.randn(max_z, 2)) * magnitude
            aug_seg['r_hit_full'] = np.where(np.random.rand(max_z) <= frequency)[0]
            aug_seg['r_hits'] = []
            for r in range(len(z_offset)):
                aug_seg['r_hits'].append(aug_seg['r_hit_full'] - z_offset[r])
                aug_seg['r_hits'][-1][np.where(aug_seg['r_hits'][-1] >= z_num[r])[0]] = -1   
            
            return aug_seg
           
        
    def gaussian_fft_filter(self, x, sigma_h, sigma_w=None, pad_mult=5, device='cpu'):
        import torch.fft as tfft
    
        # x: (N, C, H, W), float32
        if sigma_w is None: sigma_w = sigma_h
        dev, dt = device, torch.float32
        old_mean = x.mean()
        
    
        # Pad to reduce circular artifacts: ~3σ on each spatial edge
        pad_vec = np.zeros(x.ndim, int)
        pad_vec[-2:] = np.ceil(np.array((sigma_h, sigma_w)) * pad_mult + 0.5).astype(int)
        old_shape = np.array(x.shape, int)
        pad_shape = old_shape + pad_vec
        slices = tuple(slice(c, c + s) for c, s in zip(pad_vec, old_shape))
        
        xpad = torch.zeros(tuple(pad_shape),dtype=torch.float32) + x.mean()
                
        if torch.is_tensor(x):
            xpad[slices] = x
        else:
            xpad[slices] = torch.from_numpy(x)
                
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
        if torch.is_tensor(x):
            y = ypad[slices]
        else:
            y = ypad[slices].detach().numpy()
        
        return y
    
    def blackout(self, key, frequency, magnitude):
    

        if frequency:
            ag = self.section_augment_info(key, frequency, magnitude)
            max_fov = np.ceil(ag['max_fov']).astype(int)
            blob_field = np.zeros((max_fov[1], max_fov[2]),np.float32)
            
            for zi in range(len(ag['r_hit_full'])):
                sigma = [ag['r_sigs'][zi, 0], ag['r_sigs'][zi, 1] ] 
                blob_field = np.random.rand(max_fov[1]+int(sigma[0]*6), max_fov[2]+int(sigma[1]*6))
                #blob_field = self.gaussian_fft_filter(blob_field, sigma[0], sigma[1], pad_mult=0)
                blob_field = sk.filters.gaussian(
                    blob_field, 
                    sigma=sigma,
                    mode='reflect')
                
                blob_mask = (blob_field > .5).astype(np.float32)
                    
                for ai in range(len(ag['vox_sizes'])):
                    hit_z = ag['r_hits'][ai][zi] 
                    if hit_z >= 0:
                        blob_scaled = sk.transform.downscale_local_mean(blob_mask, 
                                factors= tuple((ag['vox_sizes'][ai,1], ag['vox_sizes'][ai,2])))
                        blob_pull = self.get_center(arr=blob_scaled, target_shape=ag['shapes'][ai,1:])
                        # self.pulled[key][ai][0, 0, ag['r_hits'][ai][zi], :, :] = \
                        #      self.pulled[key][ai][0, 0, ag['r_hits'][ai][zi], :, :] - torch.from_numpy(blob_pull)
                        blob_tensor = torch.from_numpy(blob_pull)
                        self.pulled[key][ai][0, 0, hit_z, :, :] = self.pulled[key][ai][0, 0, hit_z, :, :] - blob_tensor   
             
                for ai in range(len(self.pulled[key])):
                    self.pulled[key][ai][np.where(self.pulled[key][ai]<0)] = 0
    
    def random_rotate_z(self):
            
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
                    self.pulled[k][mi][:] = torch.from_numpy( 
                        nd_rotate(
                        m,
                        angle=angle_deg,
                        axes=rotate_dim,  # rotate in the XY plane
                        reshape=False,
                        order=1,      # bilinear interpolation
                        mode='constant',
                        cval = 0
                    ))
             
        
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
       
        if frequency:
            ag = self.section_augment_info(key, frequency, magnitude)
            
            for zi in range(len(ag['r_hit_full'])):
                for ai in range(len(ag['vox_sizes'])):
                    if ag['r_hits'][ai][zi] >= 0:
                        sigma = ag['r_sigs'][zi, :] / ag['vox_sizes'][ai][1:]
                        
                        self.pulled[key][ai][0, 0, ag['r_hits'][ai][zi], :, :] \
                            = self.gaussian_fft_filter(
                            self.pulled[key][ai][0, 0, ag['r_hits'][ai][zi], :, :],
                            sigma[0], sigma[1])
                      
                        # arr =  self.pulled[key][ai][0, 0, ag['r_hits'][ai][zi], :, :].detach().numpy()
                        # arr2 = sk.filters.gaussian(
                        #     arr, 
                        #     sigma=sigma,
                        #     mode='reflect')
                        # self.pulled[key][ai][0, 0, ag['r_hits'][ai][zi], :, :] = torch.from_numpy(arr2)
                    
                
    def jitter(self, key, frequency, magnitude):
        if frequency:
            ag = self.section_augment_info(key, frequency, magnitude)
            
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
        else:
            self.jit['do'] = 0
       

    def jitter_pull(self):
         print('doesnt jitter yet')
         print("should {self.jit['buf']")
         for ri, k in enumerate(self.requests.keys()):
             r = self.requests[k]
             if r['is_input']:
                 for mi, m in enumerate(r['use_mips']):
                     self.pulled[k][mi][:] = 0
                     pull_corner = self.center_voxes[mi] - r['pull_shapes'][mi] // 2
                     pull_win = np.array([pull_corner, pull_corner +  r['pull_shapes'][mi]]).astype(int)
                     source_win = np.array([[0, 0, 0], self.zm.datasets[r['source_key']].shapes[mi]])
                     transfer = vt.block_to_window(source_win,pull_win)
                     if transfer['has_data']:
                         vt.array_to_tensor(self.pulled[k][mi], transfer['ch'], 
                                    self.zm.datasets[r['source_key']].arr[mi], transfer['win'])


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












