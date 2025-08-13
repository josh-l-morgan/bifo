import torch
from torch import nn
import torch.nn.functional as F


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


class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, deep_supervision=True, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]


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


