# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:14:07 2025

@author: jlmorgan
"""
import torch
from torch import nn
import torch.nn.functional as F

def crop_center(x, shape):
    """Crop x (B, C, D, H, W) to shape=(D,H,W) centered."""
    _, _, d, h, w = x.shape
    td, th, tw = shape
    d1 = (d - td) // 2
    h1 = (h - th) // 2
    w1 = (w - tw) // 2
    return x[:, :, d1:d1+td, h1:h1+th, w1:w1+tw]


class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, deep_supervision=True, **kwargs):
        super().__init__()

        nb_filter = [4, 8, 16]


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


class TripleNestedUNetCascade(nn.Module):
    def __init__(self):
        super().__init__()

        # Stage v1: Large FOV, small channels
        self.v1 = NestedUNet(input_channels=1, nb_filter=[4, 8, 16], deep_supervision=True)

        # Stage v2: Mid FOV, mid channels (raw + v1_out)
        self.v2 = NestedUNet(input_channels=2, nb_filter=[8, 16, 32], deep_supervision=True)

        # Stage v3: Small FOV, large channels (raw + v2_out)
        self.v3 = NestedUNet(input_channels=2, nb_filter=[256, 512, 1024], deep_supervision=True)

    def forward(self, raw):
        """
        raw: Tensor of shape (B, 1, D, H, W), should be at least 512³
        """

        ### --- Stage v1 ---
        raw_v1 = crop_center(raw, (raw.shape[2], 512, 512))  # ensure 512x512
        out1 = self.v1(raw_v1)
        out1_core = crop_center(out1[-1], (256, 256, 256))

        ### --- Stage v2 ---
        raw_v2 = crop_center(raw, (raw.shape[2], 256, 256))
        x_v2 = torch.cat([raw_v2, out1_core], dim=1)
        out2 = self.v2(x_v2)
        out2_core = crop_center(out2[-1], (64, 64, 64))

        ### --- Stage v3 ---
        raw_v3 = crop_center(raw, (raw.shape[2], 64, 64))
        x_v3 = torch.cat([raw_v3, out2_core], dim=1)
        out3 = self.v3(x_v3)

        return out1, out2, out3  # full supervision or final output
    
model = TripleNestedUNetCascade()
raw = torch.randn(1, 1, 25, 512, 512)  # full raw input
target = torch.randint(0, 2, (1, 1, 25, 64, 64)).float()

out1, out2, out3 = model(raw)
loss_fn = nn.BCEWithLogitsLoss()

loss = (
    0.1 * loss_fn(out1[-1], crop_center(target, out1[-1].shape[2:])) +
    0.3 * loss_fn(out2[-1], crop_center(target, out2[-1].shape[2:])) +
    1.0 * loss_fn(out3[-1], target)
)
loss.backward()

