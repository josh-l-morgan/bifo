# jmTrainTools module

"""
Created on Mon Jul  7 18:48:41 2025

@author: jlmorgan
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import gunpowder as gp
import copy
import bifo.tools.display as dsp



def cropTensor(raw_tensor,ct):
    
    if isinstance(raw_tensor, list):

        cropped_tensor = []
        for rt in raw_tensor:
           
            end = torch.tensor(rt.shape) - torch.tensor(ct)
            cropped_t = rt[tuple(slice(s, e) for s, e in zip(ct, end))]       
            #cropped_t = raw_tensor[ct[0]:end[0],ct[1]:end[1],ct[2]:end[2],ct[3]:end[3],ct[4]:end[4]]
            cropped_tensor.append(cropped_t)
        
    else:
        end = torch.tensor(raw_tensor.shape) - torch.tensor(ct)
        cropped_tensor = raw_tensor[tuple(slice(s, e) for s, e in zip(ct, end))]     
        #cropped_tensor = raw_tensor[ct[0]:-ct[0],ct[1]:-ct[1],ct[2]:-ct[2],ct[3]:-ct[3],ct[4]:-ct[4]]

    return cropped_tensor


def center_window(s1,s2):
    s1 = np.array(s1)
    s2 = np.array(s2)
    
    lower = (s1 - s2) // 2
    cap = lower + s2
    return np.array([lower, cap])


class showTrainingProgress:
   
    def __init__(self, xs = 256):
               
        fig_num = plt.get_fignums()
        if fig_num:
            self.fig = plt.figure(fig_num[-1])
        else:
            self.fig = plt.figure()
        self.fig.clear()  
        
        #Set up plots
        self.fig.set_size_inches((12,8))
        self.sp1 = self.fig.add_subplot(1,3,1)
        self.sp2 = self.fig.add_subplot(1,3,2)
        self.sp3 = self.fig.add_subplot(1,3,3)
        
        self.im1 = self.sp1.imshow(np.zeros((xs,xs)), cmap='gray', vmin=0, vmax=1)
        self.im2 = self.sp2.imshow(np.zeros((xs,xs)), cmap='gray', vmin=0, vmax=1)
        self.im3 = self.sp3.imshow(np.zeros((xs,xs)), cmap='gray', vmin=0, vmax=1)
        
        self.sp1.set_title("Input")
        self.sp2.set_title("Target")
        self.sp3.set_title("Prediction")
        
        self.fig.show()
    
        
    def update(self, input_tensor, target_tensor, pred, showPlane = None):
    
        midpoint = round(input_tensor.shape[2]/2)
        if showPlane is None:
            showPlane = [midpoint]
        elif showPlane == 'all':
            showPlane = np.arange(input_tensor.size(2))
            showPlane = np.concatenate((showPlane[::-1], showPlane[1:midpoint]))
        
        for i in showPlane:        
            
            # Ensure everything is on CPU and converted to numpy
            i_input = input_tensor[0, 0, i].detach().cpu().numpy()
            i_target = target_tensor[0, 0, i].detach().cpu().numpy()
            i_pred = pred[0, 0, i].detach().cpu().numpy()        
              
            # Normalize
            i_pred = i_pred - i_pred.min()
            i_pred = i_pred / (i_pred.max()+ 1e-8)
            
            # Display
            self.im1.set_data(i_input/i_input.max())
            self.im2.set_data(i_target)
            self.im3.set_data(i_pred)                     
            
            title1 = f"Target section {i}"
            self.sp1.set_title(title1)

            # self.fig.canvas.blit(self.sp1.bbox)
            # self.fig.canvas.blit(self.sp2.bbox)
            # self.fig.canvas.blit(self.sp3.bbox)
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
            plt.pause(0.01)   
            
        i_cat = np.concatenate((i_input*255/i_input.max(), i_target*200, i_pred*255),1)
        i_cat = np.uint8(i_cat)
        i_col = np.dstack((i_pred*255, i_input*255/i_input.max(), i_target*200))
        i_col = np.uint8(i_col)
    
        return i_cat, i_col
        
         
class showTrainingProgressPts:
   
    def __init__(self, tr):
               
        fig_num = plt.get_fignums()
        if fig_num:
            self.fig = plt.figure(fig_num[-1])
        else:
            self.fig = plt.figure()
        self.fig.clear()  
        
        #Set up plots
        v_shapes = tr['v_shapes']
        self.v_num = len(v_shapes)
        
        self.fig.set_size_inches((9,9))
        self.ax = []
        num_ax = self.v_num  * 2
        for a in range(num_ax):
            self.ax.append(self.fig.add_subplot(2, self.v_num ,a+1))
        
        for vi in range(self.v_num):
            dummy = np.zeros((v_shapes[vi][1], v_shapes[vi][2]))
            self.ax[vi].im = self.ax[vi].imshow(dummy, cmap='gray', vmin=0, vmax=1)
            self.ax[vi+self.v_num].im = self.ax[vi+self.v_num].imshow(dummy, cmap='gray', vmin=0, vmax=1)
            self.ax[vi].scat = self.ax[vi].scatter(0,0)
            self.ax[vi+self.v_num].scat = self.ax[vi+self.v_num].scatter(0,0)
        
        for a in range(self.v_num ):
            self.ax[a].set_title("v{a}")
                
        self.fig.show()
    
        
    def update(self, input_t, pred_t, pts=None, show_plane=None):
      
        shift_z = []
        for k in range(self.v_num):
            shift_z.append((input_t[k].shape[2]-input_t[-1].shape[2])//2)
        
        midpoints = [round(input_t[k].shape[2]/2) for k in range(len(input_t))]
        if show_plane is None:
            showPlane = [midpoints[-1]]
        elif show_plane == 'all':
            showPlane = np.arange(input_t[-1].size(2))
            showPlane = np.concatenate((showPlane[::-1], showPlane[1:midpoints[-1]]))
                
        for i in showPlane:        
            
            # Ensure everything is on CPU and converted to numpy
            i_input = []
            i_pred = []
            for k in range(len(input_t)):
                i_input.append(input_t[k][0, 0, i+shift_z[k]].detach().cpu().numpy())
                if isinstance(pred_t[k], list):
                    i_pred.append(pred_t[k][-1][0, 0, i+shift_z[k]].detach().cpu().numpy())
                else:
                    i_pred.append(pred_t[k][0, 0, i+shift_z[k]].detach().cpu().numpy())
            
            # Normalize
            for i,inp in enumerate(i_input):
                i_input[i] = inp/inp.max()
                
            for i, pred in enumerate(i_pred):
                pred = pred - pred.min()
                pred = pred / (pred.max() + 1e-8)
                i_pred[i] = pred
            
            # Display
            for a in range(self.v_num):
                self.ax[a].im.set_data(i_input[a])
                self.ax[a+self.v_num].im.set_data(i_pred[a])
                self.ax[a].set_title(f'v{a}')
                self.ax[a].relim()
                self.ax[a].autoscale_view()
            
            
            for vi in range(self.v_num):
                self.ax[vi].scat.remove()
                self.ax[vi + self.v_num].scat.remove() 
                self.ax[vi].scat = self.ax[vi].scatter(
                    pts[vi][0][:,-1], pts[vi][0][:,-2], 10, [1, 0, 0, .5])
                self.ax[vi + self.v_num].scat = self.ax[vi + self.v_num].scatter(
                    pts[vi][0][:,-1], pts[vi][0][:,-2], 10, [1, 0, 0, .5])
            
            self.fig.canvas.flush_events()
            self.fig.canvas.draw_idle()
            plt.pause(0.01)
               
            
        # i_cat = np.concatenate((i_input_2*255/i_input_2.max(), i_target*200, i_pred*255),1)
        # i_cat = np.uint8(i_cat)
        # i_col = np.dstack((i_pred*255, i_input_2*255/i_input_2.max(), i_target*200))
        # i_col = np.uint8(i_col)
  

               
class showTrainingProgress3V(dsp.figure):
   
    def __init__(self, tr):
        
        super().__init__(num_ax = 9, fig_name='training')   
       
        #Set up plots
        self.fig.set_size_inches((9,9))
      
        
        for a in self.ax:
            a.im = a.imshow(np.zeros((256, 256)), cmap='gray', vmin=0, vmax=1)
        
        self.ax[0].set_title("v0")
        self.ax[1].set_title("v1")
        self.ax[2].set_title("v2")
        
    def show_dat(self, input_t, pred_t, lab_t=None, show_plane=None):
      
        shift_z = []
        for k in range(3):
            shift_z.append((input_t[k].shape[2]-input_t[-1].shape[2])//2)
        
        midpoints = [round(input_t[k].shape[2]/2) for k in range(len(input_t))]
        if show_plane is None:
            showPlane = [midpoints[-1]]
        elif show_plane == 'all':
            showPlane = np.arange(input_t[-1].size(2))
            showPlane = np.concatenate((showPlane[::-1], showPlane[1:midpoints[-1]]))
        else:
            showPlane = [show_plane]
        
        for i in showPlane:        
            
            # Ensure everything is on CPU and converted to numpy
            i_input = []
            i_pred = []
            for k in range(len(input_t)):
                i_input.append(input_t[k][0, 0, i+shift_z[k]].detach().cpu().numpy())
                if isinstance(pred_t[k], list):
                    i_pred.append(pred_t[k][-1][0, 0, i+shift_z[k]].detach().cpu().numpy())
                else:
                    i_pred.append(pred_t[k][0, 0, i+shift_z[k]].detach().cpu().numpy())
            
            if lab_t is None:
                i_target = [pr * 0 for pr in i_pred]
            else:
                i_target = []
                for k in range(len(input_t)):
                    i_target.append(lab_t[k][0, -1, i+shift_z[k]].detach().cpu().numpy())
                        
            # Normalize
            for i,inp in enumerate(i_input):
                i_input[i] = inp/inp.max()
                
            for i, pred in enumerate(i_pred):
                pred = pred - pred.min()
                pred = pred / (pred.max() + 1e-8)
                i_pred[i] = pred
            
            # Display
            for a in range(3):
                self.ax[a].im.set_data(i_input[a])
                self.ax[a+6].im.set_data(i_pred[a])
                self.ax[a+3].im.set_data(i_target[a])
                self.ax[a].set_title(f'v{a}')

            self.update()
            
            
        # i_cat = np.concatenate((i_input_2*255/i_input_2.max(), i_target*200, i_pred*255),1)
        # i_cat = np.uint8(i_cat)
        # i_col = np.dstack((i_pred*255, i_input_2*255/i_input_2.max(), i_target*200))
        # i_col = np.uint8(i_col)
    
        # return i_cat, i_col    

class show_prev_v3(dsp.figure):
    def __init__(self):
        super().__init__(num_ax = 8, fig_name='prev')
        for a in self.ax:
            a.img = a.imshow(np.zeros((64, 64)), cmap='grey', vmax=1)
        self.num_v = 3
        self.num_c = 4

    def show_dat(self, prev):
        
        for vi in range(self.num_v-1):
            for c in range(self.num_c):
                mid_point = int(prev[vi].shape[2] // 2)
                im = prev[vi][0, c, mid_point, :, :].detach().numpy()
                self.ax[c+ vi * self.num_c].img.set_data(im / 2 + .1)
        self.update()

                
class showInferenceProgressV3:
   
    def __init__(self, xs = 256):
               
        self.fig = plt.figure(num='train')    
        self.fig.clear()  
        
        #Set up plots
        self.fig.set_size_inches((12,8))
        self.ax = []
        num_ax = 6
        for a in range(num_ax):
            self.ax.append(self.fig.add_subplot(2,3,a+1))
        
        for a in range(num_ax):
            self.ax[a].im = self.ax[a].imshow(np.zeros((xs,xs)), cmap='gray', vmin=0, vmax=1)
        
        self.ax[0].set_title("EM input for mip of v")
        self.ax[1].set_title("Previous output from v-1")
        self.ax[2].set_title("v prediction")
        self.ax[3].set_title("v prediction")
        self.ax[4].set_title("v prediction")
        self.ax[5].set_title("v prediction")
        
        self.fig.show()
    
        
    def update(self, input_t=None, pred_t=None, lab_t=None, 
               prev=None, show_plane=None):
      
             
            
        midpoints = round(input_t.shape[0]/2)
        if show_plane is None:
            showPlane = [midpoints]
        elif show_plane == 'all':
            showPlane = np.arange(input_t[0].size(2))
            showPlane = np.concatenate((showPlane[::-1], showPlane[1:midpoints]))
        else:
            showPlane = show_plane
        
        for i in showPlane:        
            
            # Ensure everything is on CPU and converted to numpy
            i_input = input_t[i,:,:]
            if prev is None:
                i_prev = [i_input * 0] 
            else:
                i_prev = []
                for k in range(prev.shape[0]):
                    i_prev.append(prev[k, i, :, :])
            
            i_pred = pred_t[i,:,:]
           
            # Normalize
            i_input[np.isnan(i_input)] = 0 
            i_input = i_input/i_input.max()
            i_pred = i_pred - i_pred.min()
            i_pred = i_pred/(i_pred.max() + 1e-8)
                
            for k, pred in enumerate(i_prev):
                pred = pred - pred .min()
                pred = pred / (pred.max()+ 1e-8)
                i_prev[k] = pred
            
            # Display
            self.ax[0].im.set_data(i_input)
            self.ax[1].im.set_data(i_pred)
            for k in range(len(i_prev)):
                self.ax[k+2].im.set_data(i_prev[k])
                
            for a in self.ax:
                h, w = a.im.get_array().shape
                a.im.set_extent((0, w, h, 0))
                      
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
            plt.pause(0.01)   
      
def pickTargetOrLastFileID(checkpoint_dir,start_checkpoint_itteration):
    """
    Find either the last file in a folder or the file identified by the second input
    The file itteration is defined by the last digit in the file name
    Enter 'new' to return an empty '' path
    If no second input is give, the maximum file ID will be returned

    Parameters
    ----------
    checkpoint_dir : string
        DESCRIPTION.
    start_checkpoint_itteration : int
        if -1, start from beginning.

    Returns
    -------
    cp_path : TYPE
        DESCRIPTION.
    cp_start_itt : TYPE
        DESCRIPTION.

    """
   
    
    # Identify checkpoint path
    cp_start_itt = 0
    if start_checkpoint_itteration == 'new':
        cp_use_idx = -1 # use impossible
    else:
        
        if start_checkpoint_itteration == 'None': #default to max
            target_idx = -1
        else:
            target_idx = start_checkpoint_itteration
            
        
        available_checkpoints = os.listdir(checkpoint_dir)
        numbers = [re.findall(r'\d+', s) for s in available_checkpoints]
        numbers = [int(p[-1]) for p in numbers]

                
        cp_target_idx = [i for i, x in enumerate(numbers) if x == target_idx]
        
        # Find max
        if not numbers:
            cp_max_idx = -1
        else:
            cp_max_idx = [i for i, x in enumerate(numbers) if x == np.max(numbers)][0]
        
        # Decide
        if not cp_target_idx:
            cp_use_idx = cp_max_idx # use max
        else:
            cp_use_idx = cp_target_idx[0] # use target
            
    if cp_use_idx > -1 :
        cp_path = checkpoint_dir + available_checkpoints[cp_use_idx]
        cp_start_itt = numbers[cp_use_idx]
    else:
        cp_path = ''
        cp_start_itt = 0
        
    return cp_path, cp_start_itt

class CastArray(gp.BatchFilter):
    def __init__(self, array_key, dtype):
        self.array_key = array_key
        self.dtype = dtype

    def process(self, batch, request):
        data = batch[self.array_key].data
        batch[self.array_key].data = data.astype(self.dtype)
        batch[self.array_key].spec.dtype = self.dtype  
        return batch

class gpVaryBrightContrast(gp.BatchFilter):
    def __init__(self, array_key, b_var, c_var):
        self.array_key = array_key
        self.b_var = b_var
        self.c_var = c_var

    def process(self, batch, request):
        data = batch[self.array_key].data
        c_change = np.random * self.c_var
        if np.random > .5:
            c_change = 1/c_change
        data = data * c_change
        
        batch[self.array_key].data = data.astype(self.dtype)
        batch[self.array_key].spec.dtype = self.dtype  
        return batch

# def tr_RF(tr):
#     """
#     Computes receptive field size for a sequential stack of conv layers.
    
#     layers: list of dicts, each with keys: 'kernel_size', 'stride', 'padding', 'dilation'

#     Returns:
#         total_rf: receptive field size
#         total_stride: effective stride
#         total_padding: total padding
#     """
#     rf = 1
#     stride = 1
#     dilation = 1

#     for layer in layers:
#         k = layer['kernel_size']
#         s = layer['stride']
#         p = layer['padding']
#         d = layer['dilation']

#         rf = rf + ((k - 1) * d) * stride
#         stride *= s
#         dilation *= d

#     return rf, stride



def change_conv3d_padding_mode(module, new_padding_mode='replicate'):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv3d):
            # Rebuild the Conv3d layer with new padding mode
            new_conv = nn.Conv3d(
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
            change_conv3d_padding_mode(child, new_padding_mode)




# class uProp():

#     def __init__(self):
#         ds = [
#             [1, 2, 2],
#             [1, 8, 8]
#         ]
#         ksd = [
#             [[1, 3, 3], [1, 3, 3]],
#             [[1, 5, 5], [1, 7, 7]],
#             [[3, 3, 3], [3, 3, 3]]
#         ]
#         ksu = [
#             [[1, 3, 3], [1, 3, 3]],
#             [[1, 7, 7], [1, 5, 5]]
#         ]
        
#         #Estimate receptive field
#         rf_width = 513
#         rf_center = (rf_width-1) / 2 + 1
#         xy_rf = np.zeros([rf_width])
#         xy_rf[int(rf_center)] = 1
#         xy_dsF = 1;
#         z_rf = np.zeros([rf_width])
#         z_rf[int(rf_center)] = 1
#         z_dsF = 1;
#         for v in range(len(ksd)):
#             for c in range(len(ksd[v])):
#                 xy_look = ksd[v][c][1] * xy_dsF;
#                 xy_f = np.ones([xy_look])
#                 xy_rf = np.convolve(xy_rf,xy_f,mode='same')
#                 z_look = ksd[v][c][0] * z_dsF;
#                 z_f = np.ones([z_look])
#                 z_rf = np.convolve(z_rf,z_f,mode='same')
#                 #ax.plot(rf)
#             if v < len(ds):
#                 xy_dsF = xy_dsF * ds[v][1]
#                 z_dsF = z_dsF * ds[v][0]
#         xy_max_rf = xy_rf.max()
#         z_max_rf = z_rf.max()
     
#         self.downsample_factors = ds
#         self.kernel_size_down = ksd
#         self.kernel_size_up = ksu
#         self.xy_receptiveField = xy_rf
#         self.z_receptiveField = xy_rf
#         self.xy_half_max_rad =  rf_center - np.where(xy_rf>xy_max_rf/2)[0][0]       
#         self.xy_full_reach =  rf_center - np.where(xy_rf>0)[0][0]       
#         self.z_half_max_rad =  rf_center - np.where(z_rf>z_max_rf/2)[0][0]       
#         self.z_full_reach =  rf_center - np.where(z_rf>0)[0][0]  


class CheckValid():
    def __init__(self, modules, fill_with=1):
        self.modules_in = modules
                
        ## make dummy network for calculating the validity of voxels    
        self.test_model = nn.ModuleList()
        for v in self.modules_in: ## duplicate model
            self.test_model.append(copy.deepcopy(v))
        with torch.no_grad(): ## Set all weights
            for m in self.test_model:
                for param in m.parameters():
                   param.fill_(fill_with)
        
        
        
    def check_shapes(self, tr, border_val=1):    
        
        num_m = len(self.test_model)
        min_border_val = np.zeros(num_m)
        max_border_val = np.zeros(num_m)
       
        self.param = {'crop_validity': np.array([.68, .95, .999])}
        self.maps = []
        self.masks
        for mi, m in enumerate(self.test_model):
           
            self.maps.append({})
            ## plot validit from edges
            change_conv3d_padding_mode(m, 'replicate') ## change padding mode
            t_shape = np.array(tr['v_shapes'][mi])
            self.maps[mi]['test_shape'] = t_shape.copy()
            pad_shape = t_shape + 2
            border_tensor = torch.ones(1, 1, pad_shape[0], pad_shape[1], pad_shape[2]) * border_val
            border_tensor[0, 0, 1:1+t_shape[0], 1:1+t_shape[1], 1:1+t_shape[2]] = 0           
            border_tensor = border_tensor.repeat(1, tr['input_channels'][mi], 1, 1, 1)
            border_tensor = border_tensor.to(tr['device'])
            border_out = self.test_model[mi](border_tensor)
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

           
            # ## plot from single voxel at corner
            # change_conv3d_padding_mode(m, 'zeros')
            # rf_tensor = torch.zeros(1, 1, t_shape[0], t_shape[1], t_shape[2])
            # rf_tensor[0, 0, 0, 0, 0] = 1 #/(10**9)
            # rf_tensor = rf_tensor.repeat(1, tr['input_channels'][mi], 1, 1, 1)
            # rf_out = self.test_model[mi](rf_tensor)
            # rf_map = rf_out[-1][0, :].detach().numpy().squeeze()  # if deep supervision
            
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

                
    # def check_mask(self, mask, tr):
    #     rf_maps = [''] * len(self.test_model)
    #     for mi, m in enumerate(self.test_model):
    #         ## plot from single voxel at corner
    #         change_conv3d_padding_mode(m, 'zeros')
    #         t_shape = mask[mi].shape
    #         rf_tensor = torch.zeros(1, 1, t_shape[0], t_shape[1], t_shape[2])
    #         rf_tensor[0, 0, :, :, :] = mask[mi] #/(10**9)
    #         rf_tensor = rf_tensor.repeat(1, tr['input_channels'][mi], 1, 1, 1)
    #         rf_out = test_model[mi](rf_tensor)
    #         rf_map[mi] = rf_out[-1][0, :].detach().numpy().squeeze()  # if deep supervision
    #     return rf_map
 
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

        
        
        
        