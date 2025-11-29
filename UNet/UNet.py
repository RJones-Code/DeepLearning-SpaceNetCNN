import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters

        self.down1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(f, f*2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(f*2, f*4)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(f*4, f*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f*8, f*16)

        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)
        self.dec4 = DoubleConv(f*16, f*8)

        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.dec3 = DoubleConv(f*8, f*4)

        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.dec2 = DoubleConv(f*4, f*2)

        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.dec1 = DoubleConv(f*2, f)

        self.out = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        c1 = self.down1(x); p1 = self.pool1(c1)
        c2 = self.down2(p1); p2 = self.pool2(c2)
        c3 = self.down3(p2); p3 = self.pool3(c3)
        c4 = self.down4(p3); p4 = self.pool4(c4)
        bn = self.bottleneck(p4)

        u4 = self.up4(bn); u4 = torch.cat([u4, c4], dim=1); u4 = self.dec4(u4)
        u3 = self.up3(u4); u3 = torch.cat([u3, c3], dim=1); u3 = self.dec3(u3)
        u2 = self.up2(u3); u2 = torch.cat([u2, c2], dim=1); u2 = self.dec2(u2)
        u1 = self.up1(u2); u1 = torch.cat([u1, c1], dim=1); u1 = self.dec1(u1)

        return self.out(u1)
