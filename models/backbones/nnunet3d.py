"""
nnU-Net 3D Architecture for Volumetric Medical Image Segmentation (BHSD).
Implemented following the nnU-Net design principles (Isensee et al., Nature Methods 2021):
- 5 resolution stages (perfectly downsamples 32x160x160 -> 2x10x10 at bottleneck).
- Double Conv3D blocks with InstanceNorm3d and LeakyReLU (negative slope 0.01).
- Strided Conv3D / MaxPool3d for downsampling and Transposed Conv3D for upsampling.
- Deep supervision-friendly modular layout.
"""

from typing import List, Tuple, Optional, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODELS

DEFAULT_MAX_FILTERS = 320
DEFAULT_DROPOUT_PROB = 0.2
DEFAULT_STAGE_MULTIPLIERS: Tuple[int, ...] = (1, 2, 4, 8, 10, 10, 10)
DEFAULT_ANISOTROPIC_KERNELS: Tuple[Tuple[int, int, int], ...] = (
    (1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)
)
DEFAULT_ISOTROPIC_KERNELS: Tuple[Tuple[int, int, int], ...] = ((3, 3, 3),) * 7
DEFAULT_ANISOTROPIC_POOL_STRIDES: Tuple[Tuple[int, int, int], ...] = (
    (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)
)
DEFAULT_ISOTROPIC_POOL_STRIDES: Tuple[Tuple[int, int, int], ...] = (
    (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)
)


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
    """
    CAPABILITIES = {"segmentation", "deep_supervision"}

    """
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
        dropout_prob: float = DEFAULT_DROPOUT_PROB,
        max_filters: int = DEFAULT_MAX_FILTERS,
        stage_multipliers: Optional[Sequence[int]] = None,
        pool_strides: Optional[Sequence[Tuple[int, int, int]]] = None,
        stage_kernels: Optional[Sequence[Tuple[int, int, int]]] = None,
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
        multipliers = tuple(stage_multipliers) if stage_multipliers is not None else DEFAULT_STAGE_MULTIPLIERS
        if len(multipliers) < num_stages:
            raise ValueError(f"stage_multipliers needs at least {num_stages} entries, got {len(multipliers)}")
        f = [
            (base_filters * int(m) if idx < 4 else min(base_filters * int(m), max_filters))
            for idx, m in enumerate(multipliers[:num_stages])
        ]

        # Kernel configurations: (1, 3, 3) for enc1, enc2 if anisotropic; else (3, 3, 3)
        if stage_kernels is not None:
            kernels = [tuple(k) for k in stage_kernels]
        else:
            kernels = list(DEFAULT_ANISOTROPIC_KERNELS if anisotropic_kernels else DEFAULT_ISOTROPIC_KERNELS)
        if len(kernels) < num_stages:
            raise ValueError(f"stage_kernels needs at least {num_stages} entries, got {len(kernels)}")
        k1, k2, k3, k4 = kernels[0], kernels[1], kernels[2], kernels[3]
        k5 = kernels[4] if num_stages >= 5 else (3, 3, 3)
        k6 = kernels[5] if num_stages >= 6 else (3, 3, 3)
        k_btn = kernels[num_stages - 1]

        if pool_strides is not None:
            strides = [tuple(s) for s in pool_strides]
        else:
            strides = list(DEFAULT_ANISOTROPIC_POOL_STRIDES if anisotropic_kernels else DEFAULT_ISOTROPIC_POOL_STRIDES)
        if len(strides) < num_stages - 1:
            raise ValueError(f"pool_strides needs at least {num_stages - 1} entries, got {len(strides)}")
        stride1, stride2, stride3, stride4 = strides[0], strides[1], strides[2], strides[3]
        stride5 = strides[4] if num_stages >= 6 else (1, 2, 2)
        stride6 = strides[5] if num_stages >= 7 else (1, 2, 2)

        def _make_downsample(in_c: int, stride: Tuple[int, int, int]):
            if use_strided_conv:
                return nn.Conv3d(in_c, in_c, kernel_size=stride, stride=stride, bias=False)
            else:
                return nn.MaxPool3d(kernel_size=stride, stride=stride)

        # Encoder stages & Strides (bám sát nnUNet v2 Plans)
        self.enc1 = nnUNetBlock(in_channels, f[0], kernel_size=k1)
        self.pool1 = _make_downsample(f[0], stride=stride1)

        self.enc2 = nnUNetBlock(f[0], f[1], kernel_size=k2)
        self.pool2 = _make_downsample(f[1], stride=stride2)

        self.enc3 = nnUNetBlock(f[1], f[2], kernel_size=k3)
        self.pool3 = _make_downsample(f[2], stride=stride3)

        self.enc4 = nnUNetBlock(f[2], f[3], kernel_size=k4)
        self.pool4 = _make_downsample(f[3], stride=stride4)

        if self.num_stages >= 6:
            self.enc5 = nnUNetBlock(f[3], f[4], kernel_size=k5)
            # In 7-stage nnUNet, pool5 stride is (1, 2, 2)
            self.pool5 = _make_downsample(f[4], stride=stride5)

        if self.num_stages == 7:
            self.enc6 = nnUNetBlock(f[4], f[5], kernel_size=k6)
            self.pool6 = _make_downsample(f[5], stride=stride6)
            self.bottleneck = nnUNetBlock(f[5], f[6], kernel_size=k_btn)

            # Decoder 6
            self.up6 = nn.ConvTranspose3d(f[6], f[5], kernel_size=stride6, stride=stride6)
            self.dec6 = nnUNetBlock(f[5] * 2, f[5], kernel_size=k6)
            # Decoder 5
            self.up5 = nn.ConvTranspose3d(f[5], f[4], kernel_size=stride5, stride=stride5)
            self.dec5 = nnUNetBlock(f[4] * 2, f[4], kernel_size=k5)
        elif self.num_stages == 6:
            self.bottleneck = nnUNetBlock(f[4], f[5], kernel_size=k_btn)
            self.up5 = nn.ConvTranspose3d(f[5], f[4], kernel_size=stride5, stride=stride5)
            self.dec5 = nnUNetBlock(f[4] * 2, f[4], kernel_size=k5)
        else:
            self.bottleneck = nnUNetBlock(f[3], f[4], kernel_size=k_btn)

        self.dropout = nn.Dropout3d(p=dropout_prob) if has_dropout else nn.Identity()

        # Shared Decoders (Stages 4, 3, 2, 1)
        dec4_in = f[4]
        self.up4 = nn.ConvTranspose3d(dec4_in, f[3], kernel_size=stride4, stride=stride4)
        self.dec4 = nnUNetBlock(f[3] * 2, f[3], kernel_size=k4)

        self.up3 = nn.ConvTranspose3d(f[3], f[2], kernel_size=stride3, stride=stride3)
        self.dec3 = nnUNetBlock(f[2] * 2, f[2], kernel_size=k3)

        self.up2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=stride2, stride=stride2)
        self.dec2 = nnUNetBlock(f[1] * 2, f[1], kernel_size=k2)

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
