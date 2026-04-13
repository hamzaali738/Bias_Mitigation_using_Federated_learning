import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128, 256]):
        """
        Standard U-Net Architecture.
        in_channels: 3 for an RGB image.
        out_channels: 3 for an RGB noise map.
        features: List of channel dimensions for each level of the encoder.
        """
        super(UNet, self).__init__()
        
        # --- Encoder (Down-sampling Path) ---
        self.down1 = DoubleConv(in_channels, features[0]) # 3 -> 64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 224x224 -> 112x112

        self.down2 = DoubleConv(features[0], features[1]) # 64 -> 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 112x112 -> 56x56

        self.down3 = DoubleConv(features[1], features[2]) # 128 -> 256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 56x56 -> 28x28
        
        self.down4 = DoubleConv(features[2], features[3]) # 256 -> 512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[3], features[3] * 2) # 512 -> 1024

        # --- Decoder (Up-sampling Path) ---
        # Up-conv 1: 14x14 -> 28x28
        self.upconv1 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.up1 = DoubleConv(features[3] * 2, features[3]) # (512 from skip + 512 from upconv) -> 512

        # Up-conv 2: 28x28 -> 56x56
        self.upconv2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.up2 = DoubleConv(features[2] * 2, features[2]) # (256 from skip + 256 from upconv) -> 256

        # Up-conv 3: 56x56 -> 112x112
        self.upconv3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up3 = DoubleConv(features[1] * 2, features[1]) # (128 from skip + 128 from upconv) -> 128

        # Up-conv 4: 112x112 -> 224x224
        self.upconv4 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.up4 = DoubleConv(features[0] * 2, features[0]) # (64 from skip + 64 from upconv) -> 64

        # --- Final Output Layer ---
        # 1x1 convolution to map from 64 features to 3 output channels (noise map)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        # --- Bottleneck ---
        b = self.bottleneck(p4)

        # --- Decoder (with Skip Connections) ---
        u1 = self.upconv1(b)
        # Concatenate skip connection (d4) with up-sampled tensor (u1)
        skip1 = torch.cat([u1, d4], dim=1) # dim=1 is the channel dimension
        c1 = self.up1(skip1)
        
        u2 = self.upconv2(c1)
        skip2 = torch.cat([u2, d3], dim=1)
        c2 = self.up2(skip2)
        
        u3 = self.upconv3(c2)
        skip3 = torch.cat([u3, d2], dim=1)
        c3 = self.up3(skip3)
        
        u4 = self.upconv4(c3)
        skip4 = torch.cat([u4, d1], dim=1)
        c4 = self.up4(skip4)
            
        # --- Final Output ---
        return self.final_conv(c4)

# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256]):
#         """
#         U-Net Architecture modified for 32x32 images.
#         Has 3 down-sampling stages instead of 4.
#         """
#         super(UNet, self).__init__()
        
#         # --- Encoder (Down-sampling Path) ---
#         # 32x32 -> 32x32
#         self.down1 = DoubleConv(in_channels, features[0]) 
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16

#         # 16x16 -> 16x16
#         self.down2 = DoubleConv(features[0], features[1])
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8

#         # 8x8 -> 8x8
#         self.down3 = DoubleConv(features[1], features[2])
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4

#         # --- Bottleneck ---
#         # 4x4 -> 4x4
#         self.bottleneck = DoubleConv(features[2], features[2] * 2)

#         # --- Decoder (Up-sampling Path) ---
#         # 4x4 -> 8x8
#         self.upconv1 = nn.ConvTranspose2d(features[2] * 2, features[2], kernel_size=2, stride=2)
#         self.up1 = DoubleConv(features[2] * 2, features[2]) # (256 from skip + 256 from upconv)

#         # 8x8 -> 16x16
#         self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
#         self.up2 = DoubleConv(features[1] * 2, features[1]) # (128 from skip + 128 from upconv)

#         # 16x16 -> 32x32
#         self.upconv3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
#         self.up3 = DoubleConv(features[0] * 2, features[0]) # (64 from skip + 64 from upconv)

#         # --- Final Output Layer ---
#         # 32x32 -> 32x32
#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

#     def forward(self, x):
#         # --- Encoder ---
#         d1 = self.down1(x)   # 32x32
#         p1 = self.pool1(d1)  # 16x16
        
#         d2 = self.down2(p1)  # 16x16
#         p2 = self.pool2(d2)  # 8x8
        
#         d3 = self.down3(p2)  # 8x8
#         p3 = self.pool3(d3)  # 4x4

#         # --- Bottleneck ---
#         b = self.bottleneck(p3) # 4x4

#         # --- Decoder (with Skip Connections) ---
#         u1 = self.upconv1(b)        # 8x8
#         skip1 = torch.cat([u1, d3], dim=1)
#         c1 = self.up1(skip1)        # 8x8
        
#         u2 = self.upconv2(c1)       # 16x16
#         skip2 = torch.cat([u2, d2], dim=1)
#         c2 = self.up2(skip2)        # 16x16
        
#         u3 = self.upconv3(c2)       # 32x32
#         skip3 = torch.cat([u3, d1], dim=1)
#         c3 = self.up3(skip3)        # 32x32
        
#         # --- Final Output ---
#         return self.final_conv(c3) # 32x32