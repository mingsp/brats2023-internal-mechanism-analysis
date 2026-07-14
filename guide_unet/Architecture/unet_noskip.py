import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import DoubleConv, Down, OutConv


class UpNoSkip(nn.Module):
    """U-Net decoder block that preserves the same call signature as `Up`
    but intentionally ignores skip features.

    Keeping the signature `forward(x_deep, x_skip)` lets downstream analysis
    reuse the same hooks and data flow while ensuring the actual computation
    depends only on the decoder main path.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bool(bilinear)
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x_deep, x_skip):
        x = self.up(x_deep)
        if x_skip is not None:
            diff_y = x_skip.size()[2] - x.size()[2]
            diff_x = x_skip.size()[3] - x.size()[3]
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(x)


class UNetNoSkip(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = UpNoSkip(1024 // factor, 512 // factor, bilinear)
        self.up2 = UpNoSkip(512 // factor, 256 // factor, bilinear)
        self.up3 = UpNoSkip(256 // factor, 128 // factor, bilinear)
        self.up4 = UpNoSkip(128 // factor, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        encoder = [x2, x3, x4, x5]

        d1 = self.up1(x5, x4)
        bridge_x = d1
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        decoder = [d1, d2, d3]
        logits = self.outc(d4)
        return logits, bridge_x, encoder, decoder


def get_model(n_channels=4, n_classes=4, bilinear=False):
    return UNetNoSkip(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
