# jmTrainTools module

"""
Created on Mon Jul  7 18:48:41 2025

@author: jlmorgan
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import gunpowder as gp



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
        
                
class showTrainingProgress3V:
   
    def __init__(self, xs = 256):
               
        fig_num = plt.get_fignums()
        if fig_num:
            self.fig = plt.figure(fig_num[-1])
        else:
            self.fig = plt.figure()
        self.fig.clear()  
        
        #Set up plots
        self.fig.set_size_inches((12,8))
        self.ax = []
        num_ax = 8
        for a in range(num_ax):
            self.ax.append(self.fig.add_subplot(2,4,a+1))
        
        for a in range(num_ax):
            self.ax[a].im = self.ax[a].imshow(np.zeros((xs,xs)), cmap='gray', vmin=0, vmax=1)
        
        self.ax[3].set_title("Input")
        self.ax[4].set_title("Target")
        self.ax[5].set_title("Prediction")
        
        self.fig.show()
    
        
    def update(self, input_t, pred_t, lab_t=None, show_plane=None):
      
        shift_z = []
        for k in range(3):
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
            
            if lab_t is None:
                i_target = [pr * 0 for pr in i_pred]
            else:
                i_target = []
                for k in range(len(input_t)):
                    i_target.append(lab_t[k][0, 0, i+shift_z[k]].detach().cpu().numpy())
                        
            # Normalize
            for i,inp in enumerate(i_input):
                i_input[i] = inp/inp.max()
                
            for i, pred in enumerate(i_pred):
                pred = pred - pred .min()
                pred = pred / (pred.max()+ 1e-8)
                i_pred[i] = pred
            
            # Display
            for a in range(3):
                self.ax[a].im.set_data(i_input[a])
                self.ax[a+4].im.set_data(i_pred[a])
                self.ax[a].set_title(f'v{a}')

            self.ax[3].im.set_data(i_target[-1])
            self.ax[3].set_title('labels')
          
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
            plt.pause(0.01)   
            
        # i_cat = np.concatenate((i_input_2*255/i_input_2.max(), i_target*200, i_pred*255),1)
        # i_cat = np.uint8(i_cat)
        # i_col = np.dstack((i_pred*255, i_input_2*255/i_input_2.max(), i_target*200))
        # i_col = np.uint8(i_col)
    
        # return i_cat, i_col    


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

def tr_RF(tr):
    """
    Computes receptive field size for a sequential stack of conv layers.
    
    layers: list of dicts, each with keys: 'kernel_size', 'stride', 'padding', 'dilation'

    Returns:
        total_rf: receptive field size
        total_stride: effective stride
        total_padding: total padding
    """
    rf = 1
    stride = 1
    dilation = 1

    for layer in layers:
        k = layer['kernel_size']
        s = layer['stride']
        p = layer['padding']
        d = layer['dilation']

        rf = rf + ((k - 1) * d) * stride
        stride *= s
        dilation *= d

    return rf, stride
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
