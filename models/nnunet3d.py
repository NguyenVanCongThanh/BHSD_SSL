"""
nnU-Net 3D Architecture for Volumetric Medical Image Segmentation (BHSD).
Implemented following the nnU-Net design principles (Isensee et al., Nature Methods 2021):
- 5 resolution stages (perfectly downsamples 32x160x160 -> 2x10x10 at bottleneck).
- Double Conv3D blocks with InstanceNorm3d and LeakyReLU (negative slope 0.01).
- Strided Conv3D / MaxPool3d for downsampling and Transposed Conv3D for upsampling.
- Deep supervision-friendly modular layout.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODELS


class nnUNetBlock(nn.Module):
    """Basic two-convolution block with InstanceNorm3d and LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@MODELS.register("nnunet3d")
class nnUNet3D(nn.Module):
    """
    Standard 5-stage nnU-Net 3D for Volumetric CT Segmentation.

    Downsampling factor: 16x spatially (2^4 across 4 pooling layers)
    For BHSD (32, 160, 160):
      - Stage 1 (enc1): 32 -> (32, 160, 160)
      - Stage 2 (enc2): 64 -> (16, 80, 80)
      - Stage 3 (enc3): 128 -> (8, 40, 40)
      - Stage 4 (enc4): 256 -> (4, 20, 20)
      - Stage 5 (bottleneck): 320 -> (2, 10, 10)
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_filters: int = 32,
        has_dropout: bool = False,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Filter capacity configuration across 5 stages
        f = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, min(base_filters * 10, 320)]
        # e.g., base=32 -> [32, 64, 128, 256, 320]

        # Encoder stages
        self.enc1 = nnUNetBlock(in_channels, f[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = nnUNetBlock(f[0], f[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = nnUNetBlock(f[1], f[2])
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = nnUNetBlock(f[2], f[3])
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck (Stage 5)
        self.bottleneck = nnUNetBlock(f[3], f[4])
        self.dropout = nn.Dropout3d(p=0.2) if has_dropout else nn.Identity()

        # Decoder stages (Transposed Conv + Concat Skip + Block)
        self.up4 = nn.ConvTranspose3d(f[4], f[3], kernel_size=2, stride=2)
        self.dec4 = nnUNetBlock(f[3] * 2, f[3])

        self.up3 = nn.ConvTranspose3d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = nnUNetBlock(f[2] * 2, f[2])

        self.up2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = nnUNetBlock(f[1] * 2, f[1])

        self.up1 = nn.ConvTranspose3d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = nnUNetBlock(f[0] * 2, f[0])

        # Final segmentation head (1x1x1 Conv)
        self.final_conv = nn.Conv3d(f[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization suitable for LeakyReLU."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        # Bottleneck
        b = self.bottleneck(self.pool4(x4))
        b = self.dropout(b)

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        logits = self.final_conv(d1)
        return logits
