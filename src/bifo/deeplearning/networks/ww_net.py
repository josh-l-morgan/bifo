# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:14:07 2025

@author: jlmorgan
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



def upsample_tensor(tensor, scale_factors = [1, 2, 2]):
    """
    Upsample last dimensions of tensor.

    Args:
        tensor (torch.Tensor): tensor.
        scale_factors (tuple/list):  iterable of ints, e.g., (1, 2, 3).

    Returns:
        torch.Tensor: Upsampled tensor.
    """
    
    # Apply repeat_interleave along each axis in reverse order
    ndim = tensor.ndim
    scale_factors = np.array(scale_factors).astype(int).flatten()
    nscale = len(scale_factors)
    for d in range(nscale):
        tensor = tensor.repeat_interleave(scale_factors[d], dim= ndim-nscale+d)  # 
   
    return tensor

class conVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,con_shape):
        super().__init__()
        padding = np.array(con_shape)//2
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, con_shape, padding=padding)
        self.bn1 = nn.BatchNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, con_shape, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


def crop_center(x, shape):
    """Crop x (B, C, D, H, W) to shape=(D,H,W) centered."""
    _, _, d, h, w = x.shape
    td, th, tw = np.array(shape).astype(int)
    d1 = (d - td) // 2
    h1 = (h - th) // 2
    w1 = (w - tw) // 2
    return x[:, :, d1:d1+td, h1:h1+th, w1:w1+tw]


class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, deep_supervision=True, 
                 nb_filter=[16, 32, 64, 128], is_iso=(0,0,1,1), **kwargs):
        super().__init__()

        con_shape = []
        pool_shape = [(1, 1, 1)] ## blank pool spacer for c0
        for c in is_iso:
            if c:
                con_shape.append((3, 3, 3))
                pool_shape.append((2,2,2))
            else:
               con_shape.append((1, 3, 3))
               pool_shape.append((1,2,2))


        self.deep_supervision = deep_supervision

        self.pool1 = nn.MaxPool3d(kernel_size=pool_shape[1], stride=pool_shape[1], padding=(0, 0, 0))
        self.pool2 = nn.MaxPool3d(kernel_size=pool_shape[2], stride=pool_shape[2], padding=(0, 0, 0))
        self.pool3 = nn.MaxPool3d(kernel_size=pool_shape[3], stride=pool_shape[3], padding=(0, 0, 0))
      
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        def upsample_like(src, target):
            return F.interpolate(src, size=target.shape[2:], mode="trilinear", align_corners=True)

        self.upsample_like = upsample_like

        # Encoder
        self.conv0_0 = conVGGBlock(input_channels, nb_filter[0], nb_filter[0],con_shape[0])
        self.conv1_0 = conVGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],con_shape[1])
        self.conv2_0 = conVGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],con_shape[2])
        self.conv3_0 = conVGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],con_shape[3])

        # Decoder (Nested)
        self.conv0_1 = conVGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0],con_shape[0])
        self.conv1_1 = conVGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1],con_shape[1])
        self.conv2_1 = conVGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2],con_shape[2])

        self.conv0_2 = conVGGBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], nb_filter[0],con_shape[0])
        self.conv1_2 = conVGGBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], nb_filter[1],con_shape[2])

        self.conv0_3 = conVGGBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], nb_filter[0],con_shape[0])


        if self.deep_supervision:
            self.final1 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        input = input.float()
        up = self.upsample_like

        x0_0 = self.conv0_0(input)
        
        x1_0 = self.conv1_0(self.pool1(x0_0))
        x2_0 = self.conv2_0(self.pool2(x1_0))
        x3_0 = self.conv3_0(self.pool3(x2_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, up(x1_0, x0_0)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, up(x2_0, x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, up(x1_1, x0_0)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, up(x3_0, x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, up(x2_1, x1_0)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, up(x1_2, x0_0)], 1))

       
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            return [output1, output2, output3]

        else:
            output = self.final(x0_3)
            return output


class threeScaleNestedUNet(nn.Module):
    def __init__(self, tr ):
        super().__init__()

        nb_filters = tr['nb_filters']
        v_shapes = np.array(tr['v_shapes'])
        vox_sizes = np.array(tr['vox_sizes'])
        input_channels = np.array(tr['input_channels']).astype(int)
        use_supervision = tr['use_supervision']
        
        is_iso = tr['is_iso']
        output_channels = np.array(tr['output_channels'])
        
        num_drops = len(vox_sizes)-1 
        ds = [np.zeros(3)] * num_drops
        pull = [np.zeros(3)] * num_drops
        for d in range(len(vox_sizes)-1):
            ds[d] = vox_sizes[d] / vox_sizes[d+1]
            pull[d] = v_shapes[d+1] / ds[d]
        
        # build multiple u nets
        self.v = nn.ModuleList()
        for u in tr['use_u']:
            self.v.append(NestedUNet(num_classes = output_channels[u], 
                                 input_channels=input_channels[u], nb_filter=nb_filters[u], 
                                 deep_supervision=use_supervision[u], is_iso=is_iso[u]))
            
        self.p = {
           'nb_filters': nb_filters,
           'v_shapes': v_shapes,
           'vox_sizes': vox_sizes,
           'ds': ds,
           'pull': pull,
           'output_channels': output_channels,
           'input_channels': input_channels
           }
        

    def forward(self, input_tensor, use_v=None):
        """
        raw: Tensor of shape (B, C, D, H, W)
        """
        if use_v is None:
            
            in_tensor = input_tensor[0]
            num_tensor = len(input_tensor)
            out = []
            if num_tensor>1:
                out_core = [] ## lists for cutting center out of output for next phase
                out_core_u = [] ## list for upsampled centers
                
            for t in range(num_tensor):
                out.append(self.v[t](in_tensor)) ## run model on input tensor
                if (num_tensor > 1) & (t < (num_tensor-1)):
                    out_core.append(crop_center(out[t], self.p['pull'][t]))
                    out_core_u.append(upsample_tensor(out_core[t], self.p['ds'][t]))
                    in_tensor = torch.cat([input_tensor[t+1], out_core_u[t]], dim=1)
    
            return out  # full supervision or final output
        
        else:   
            out =  self.v[use_v](input_tensor[0]) ## run model on input tensor
            return out
        






