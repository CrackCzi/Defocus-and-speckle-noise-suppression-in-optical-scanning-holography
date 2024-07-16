import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSamp(nn.Module):
    """DownSampling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpSamp(nn.Module):
    """UpSampling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = DownSamp(64, 128)
        self.down2 = DownSamp(128, 256)
        self.down3 = DownSamp(256, 512)
        self.down4 = DownSamp(512, 1024)
        self.up1 = UpSamp(1024, 512)
        self.convU1 = DoubleConv(1024, 512)
        self.up2 = UpSamp(512, 256)
        self.convU2 = DoubleConv(512, 256)
        self.up3 = UpSamp(256, 128)
        self.convU3 = DoubleConv(256, 128)
        self.up4 = UpSamp(128, 64)
        self.convU4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.inc(x)
        d1 = self.down1(c1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        merge1 = torch.cat([u1, d3], dim=1)
        cu1 = self.convU1(merge1)
        u2 = self.up2(cu1)
        merge2 = torch.cat([u2, d2], dim=1)
        cu2 = self.convU2(merge2)
        u3 = self.up3(cu2)
        merge3 = torch.cat([u3, d1], dim=1)
        cu3 = self.convU3(merge3)
        u4 = self.up4(cu3)
        merge4 = torch.cat([u4, c1], dim=1)
        cu4 = self.convU4(merge4)
        x = self.outc(cu4)
        return x
    

###############################
#COPIED FROM THE OFFICIAL REPO#
###############################
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        """
         @brief Initialize Charbonnier Loss.
         @param eps Epsilon to use for
        """
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        """
         @brief Computes Loss between two tensors.
         @param x Tensor with shape [ batch_size image_size ]
         @param y Tensor with shape [ batch_size image_size ]
         @return A tensor with shape [ batch_size num_features ] where each element is the Loss between x and y
        """
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

###############################
#COPIED FROM THE OFFICIAL REPO#
###############################