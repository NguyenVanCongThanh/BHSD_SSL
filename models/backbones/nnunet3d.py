"""
nnU-Net 3D Architecture for Volumetric Medical Image Segmentation (BHSD).
Implemented following the nnU-Net design principles (Isensee et al., Nature Methods 2021):
- 5 resolution stages (perfectly downsamples 32x160x160 -> 2x10x10 at bottleneck).
- Double Conv3D blocks with InstanceNorm3d and LeakyReLU (negative slope 0.01).
- Strided Conv3D / MaxPool3d for downsampling and Transposed Conv3D for upsampling.
- Deep supervision-friendly modular layout.
"""

from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODELS


class nnUNetBlock(nn.Module):
    """Basic two-convolution block with InstanceNorm3d and LeakyReLU supporting anisotropic kernel."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]] = 3):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@MODELS.register("nnunet3d")
class nnUNet3D(nn.Module):
    """
    Standard 5-stage, 6-stage, or 7-stage nnU-Net 3D for Volumetric CT Segmentation.

    When num_stages == 7:
      Matches 100% with nnU-Net v2 PlainConvUNet from nnUNetPlans_Option1_CTClip and nnUNetPlans_spacing05:
      - 7 feature channels: [32, 64, 128, 256, 320, 320, 320]
      - 7 kernel sizes: [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
      - 6 downsample strides: (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)
        (Z is only downsampled 4x total, preserving thin intracranial hemorrhage structures like SAH/SDH).
      - Bottleneck at stage 7 with 320 features and 64x spatial downsampling.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_filters: int = 32,
        num_stages: int = 7,
        has_dropout: bool = False,
        deep_supervision: bool = False,
        anisotropic_kernels: bool = True,
        use_strided_conv: bool = True,
        **kwargs
    ):
        super().__init__()
        assert num_stages in (5, 6, 7), f"num_stages must be 5, 6, or 7, got {num_stages}"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.deep_supervision = deep_supervision
        self.anisotropic_kernels = anisotropic_kernels
        self.use_strided_conv = use_strided_conv

        # Filter capacity configuration across stages
        if num_stages == 7:
            f = [
                base_filters,                   # 32
                base_filters * 2,               # 64
                base_filters * 4,               # 128
                base_filters * 8,               # 256
                min(base_filters * 10, 320),    # 320
                min(base_filters * 10, 320),    # 320
                min(base_filters * 10, 320)     # 320 (bottleneck)
            ]
        elif num_stages == 6:
            f = [
                base_filters,
                base_filters * 2,
                base_filters * 4,
                base_filters * 8,
                min(base_filters * 10, 320),
                min(base_filters * 10, 320)
            ]
        else:
            f = [
                base_filters,
                base_filters * 2,
                base_filters * 4,
                base_filters * 8,
                min(base_filters * 10, 320)
            ]

        # Kernel configurations: (1, 3, 3) for enc1, enc2 if anisotropic; else (3, 3, 3)
        k1 = (1, 3, 3) if anisotropic_kernels else (3, 3, 3)
        k2 = (1, 3, 3) if anisotropic_kernels else (3, 3, 3)
        k3 = (3, 3, 3)
        k4 = (3, 3, 3)
        k5 = (3, 3, 3)
        k6 = (3, 3, 3)
        k_btn = (3, 3, 3)

        def _make_downsample(in_c: int, stride: Tuple[int, int, int]):
            if use_strided_conv:
                return nn.Conv3d(in_c, in_c, kernel_size=stride, stride=stride, bias=False)
            else:
                return nn.MaxPool3d(kernel_size=stride, stride=stride)

        # Encoder stages & Strides (bám sát nnUNet v2 Plans)
        self.enc1 = nnUNetBlock(in_channels, f[0], kernel_size=k1)
        self.pool1 = _make_downsample(f[0], stride=(1, 2, 2) if anisotropic_kernels else (2, 2, 2))

        self.enc2 = nnUNetBlock(f[0], f[1], kernel_size=k2)
        self.pool2 = _make_downsample(f[1], stride=(1, 2, 2) if anisotropic_kernels else (2, 2, 2))

        self.enc3 = nnUNetBlock(f[1], f[2], kernel_size=k3)
        self.pool3 = _make_downsample(f[2], stride=(2, 2, 2))

        self.enc4 = nnUNetBlock(f[2], f[3], kernel_size=k4)
        self.pool4 = _make_downsample(f[3], stride=(2, 2, 2))

        if self.num_stages >= 6:
            self.enc5 = nnUNetBlock(f[3], f[4], kernel_size=k5)
            # In 7-stage nnUNet, pool5 stride is (1, 2, 2)
            self.pool5 = _make_downsample(f[4], stride=(1, 2, 2))

        if self.num_stages == 7:
            self.enc6 = nnUNetBlock(f[4], f[5], kernel_size=k6)
            self.pool6 = _make_downsample(f[5], stride=(1, 2, 2))
            self.bottleneck = nnUNetBlock(f[5], f[6], kernel_size=k_btn)

            # Decoder 6
            self.up6 = nn.ConvTranspose3d(f[6], f[5], kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.dec6 = nnUNetBlock(f[5] * 2, f[5], kernel_size=k6)
            # Decoder 5
            self.up5 = nn.ConvTranspose3d(f[5], f[4], kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.dec5 = nnUNetBlock(f[4] * 2, f[4], kernel_size=k5)
        elif self.num_stages == 6:
            self.bottleneck = nnUNetBlock(f[4], f[5], kernel_size=k_btn)
            self.up5 = nn.ConvTranspose3d(f[5], f[4], kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.dec5 = nnUNetBlock(f[4] * 2, f[4], kernel_size=k5)
        else:
            self.bottleneck = nnUNetBlock(f[3], f[4], kernel_size=k_btn)

        self.dropout = nn.Dropout3d(p=0.2) if has_dropout else nn.Identity()

        # Shared Decoders (Stages 4, 3, 2, 1)
        dec4_in = f[4]
        self.up4 = nn.ConvTranspose3d(dec4_in, f[3], kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec4 = nnUNetBlock(f[3] * 2, f[3], kernel_size=k4)

        self.up3 = nn.ConvTranspose3d(f[3], f[2], kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec3 = nnUNetBlock(f[2] * 2, f[2], kernel_size=k3)

        stride2 = (1, 2, 2) if anisotropic_kernels else (2, 2, 2)
        self.up2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=stride2, stride=stride2)
        self.dec2 = nnUNetBlock(f[1] * 2, f[1], kernel_size=k2)

        stride1 = (1, 2, 2) if anisotropic_kernels else (2, 2, 2)
        self.up1 = nn.ConvTranspose3d(f[1], f[0], kernel_size=stride1, stride=stride1)
        self.dec1 = nnUNetBlock(f[0] * 2, f[0], kernel_size=k1)

        # Final segmentation head (1x1x1 Conv)
        self.final_conv = nn.Conv3d(f[0], num_classes, kernel_size=1)

        # Auxiliary heads for deep supervision
        if self.deep_supervision:
            self.ds_conv2 = nn.Conv3d(f[1], num_classes, kernel_size=1)
            self.ds_conv3 = nn.Conv3d(f[2], num_classes, kernel_size=1)
            self.ds_conv4 = nn.Conv3d(f[3], num_classes, kernel_size=1)
            if self.num_stages >= 6:
                self.ds_conv5 = nn.Conv3d(f[4], num_classes, kernel_size=1)
            if self.num_stages == 7:
                self.ds_conv6 = nn.Conv3d(f[5], num_classes, kernel_size=1)

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

    def forward(self, x: torch.Tensor, return_all_scales: bool = False):
        # Encoder forward pass
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        if self.num_stages == 7:
            x5 = self.enc5(self.pool4(x4))
            x6 = self.enc6(self.pool5(x5))
            b = self.bottleneck(self.pool6(x6))
            b = self.dropout(b)

            d6 = self.up6(b)
            d6 = torch.cat([d6, x6], dim=1)
            d6 = self.dec6(d6)

            d5 = self.up5(d6)
            d5 = torch.cat([d5, x5], dim=1)
            d5 = self.dec5(d5)

            d4 = self.up4(d5)
            d4 = torch.cat([d4, x4], dim=1)
            d4 = self.dec4(d4)
        elif self.num_stages == 6:
            x5 = self.enc5(self.pool4(x4))
            b = self.bottleneck(self.pool5(x5))
            b = self.dropout(b)

            d5 = self.up5(b)
            d5 = torch.cat([d5, x5], dim=1)
            d5 = self.dec5(d5)

            d4 = self.up4(d5)
            d4 = torch.cat([d4, x4], dim=1)
            d4 = self.dec4(d4)
        else:
            b = self.bottleneck(self.pool4(x4))
            b = self.dropout(b)

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

        if self.deep_supervision and (self.training or return_all_scales):
            aux = [
                self.ds_conv2(d2),  # Scale 0.5
                self.ds_conv3(d3),  # Scale 0.25
                self.ds_conv4(d4)   # Scale 0.125
            ]
            if self.num_stages >= 6:
                aux.append(self.ds_conv5(d5))  # Scale 0.0625
            if self.num_stages == 7:
                aux.append(self.ds_conv6(d6))  # Scale 0.03125
            return {
                "seg": logits,
                "aux": aux
            }
        return logits
