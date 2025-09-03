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


class AnisoVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, (1, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, (1, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class IsoVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
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
                 nb_filter=[16, 32, 64, 128], **kwargs):
        super().__init__()


        self.deep_supervision = deep_supervision

        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
      
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        def upsample_like(src, target):
            return F.interpolate(src, size=target.shape[2:], mode="trilinear", align_corners=True)

        self.upsample_like = upsample_like

        # Encoder
        self.conv0_0 = AnisoVGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = AnisoVGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = AnisoVGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = IsoVGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        # Decoder (Nested)
        self.conv0_1 = AnisoVGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = AnisoVGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = AnisoVGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_2 = AnisoVGGBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = AnisoVGGBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = AnisoVGGBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], nb_filter[0])


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
        output_channels = np.array(tr['output_channels'])
        ds_0 = vox_sizes[0] / vox_sizes[1]
        ds_1 = vox_sizes[1] / vox_sizes[2]
        pull_0 = v_shapes[1] / ds_0
        pull_1 = v_shapes[2] / ds_1
        
       

        # Stage v0: Large FOV, small channels
        self.v0 = NestedUNet(num_classes = output_channels[0], 
                             input_channels=1, nb_filter=nb_filters[0], 
                             deep_supervision=False)

        # Stage v1: Mid FOV, mid channels (raw + v1_out)
        self.v1 = NestedUNet(num_classes = output_channels[1], 
                             input_channels=output_channels[0]+1, 
                             nb_filter=nb_filters[1], deep_supervision=False)

        # Stage v2: Small FOV, large channels (raw + v2_out)
        self.v2 = NestedUNet(num_classes = output_channels[2],  
                             input_channels=output_channels[1]+1, 
                             nb_filter=nb_filters[2], deep_supervision=True)
        
        self.p = {
           'nb_filters': nb_filters,
           'v_shapes': v_shapes,
           'vox_sizes': vox_sizes,
           'ds_0': ds_0,
           'ds_1': ds_1,
           'pull_0': pull_0,
           'pull_1': pull_1,
           'output_channels': output_channels
           }
        

    def forward(self, input_tensor):
        """
        raw: Tensor of shape (B, 1, D, H, W)
        """
       
        ### --- Stage v1 ---
        out0 = self.v0(input_tensor[0])
        out0_core = crop_center(out0, self.p['pull_0'])
        out0_core_u = upsample_tensor(out0_core, self.p['ds_0'])
        
        ### --- Stage v2 ---        
        x_v1 = torch.cat([input_tensor[1], out0_core_u], dim=1)
        out1 = self.v1(x_v1)
        out1_core = crop_center(out1, self.p['pull_1'])
        out1_core_u = upsample_tensor(out1_core, self.p['ds_1'])

        ### --- Stage v3 ---
        x_v2 = torch.cat([input_tensor[2], out1_core_u], dim=1)
        out2 = self.v2(x_v2)

        return [out0, out1, out2]  # full supervision or final output
    
    
