"""
7-Stage Anisotropic Residual V-Net Architecture for Volumetric Medical Image Segmentation.
Designed for 1:1 ablation comparison against 7-Stage nnU-Net 3D and SWDL variants.
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import nn

from core.registry import MODELS
from models.blocks import VNetResidualBlock

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


@MODELS.register("vnet_7stage")
@MODELS.register("resvnet3d")
class VNet7Stage(nn.Module):
    """
    7-Stage Anisotropic Residual V-Net for Volumetric CT Segmentation.
    Designed for controlled 1:1 ablation comparison against 7-Stage nnU-Net 3D and SWDL variants:
    - 7 resolution stages with matching stride schedule.
    - Anisotropic kernels (1, 3, 3) at early stages, (3, 3, 3) at deeper stages.
    - Long skip connections: Concatenation ('concat') or Element-wise Addition ('add').
    - Intra-block connections: Residual addition with 1x1 projection shortcuts.
    - Multi-scale deep supervision (1 main + 5 auxiliary heads).
    """
    CAPABILITIES = {"segmentation", "deep_supervision"}

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_filters: int = 32,
        num_stages: int = 7,
        has_dropout: bool = False,
        deep_supervision: bool = True,
        anisotropic_kernels: bool = True,
        use_strided_conv: bool = True,
        skip_mode: str = "concat",
        activation: str = "leaky_relu",
        dropout_prob: float = DEFAULT_DROPOUT_PROB,
        stage_multipliers: Optional[Sequence[int]] = None,
        pool_strides: Optional[Sequence[Tuple[int, int, int]]] = None,
        stage_kernels: Optional[Sequence[Tuple[int, int, int]]] = None,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.deep_supervision = deep_supervision
        self.anisotropic_kernels = anisotropic_kernels
        self.use_strided_conv = use_strided_conv
        self.skip_mode = str(skip_mode).lower()
        self.activation = str(activation).lower()

        if self.skip_mode not in ("concat", "add"):
            raise ValueError(f"skip_mode must be 'concat' or 'add', got {self.skip_mode}")

        multipliers = tuple(stage_multipliers) if stage_multipliers is not None else DEFAULT_STAGE_MULTIPLIERS
        if len(multipliers) != 7:
            raise ValueError(f"stage_multipliers must have length 7 for VNet7Stage, got {len(multipliers)}")
        f = [base_filters * int(m) for m in multipliers]

        if stage_kernels is not None:
            kernels = [tuple(k) for k in stage_kernels]
        else:
            kernels = list(DEFAULT_ANISOTROPIC_KERNELS if anisotropic_kernels else DEFAULT_ISOTROPIC_KERNELS)
        if len(kernels) != 7:
            raise ValueError(f"stage_kernels must have length 7 for VNet7Stage, got {len(kernels)}")
        k1, k2, k3, k4, k5, k6, k_btn = kernels

        if pool_strides is not None:
            strides = [tuple(s) for s in pool_strides]
        else:
            strides = list(DEFAULT_ANISOTROPIC_POOL_STRIDES if anisotropic_kernels else DEFAULT_ISOTROPIC_POOL_STRIDES)
        if len(strides) != 6:
            raise ValueError(f"pool_strides must have length 6 for VNet7Stage, got {len(strides)}")
        stride1, stride2, stride3, stride4, stride5, stride6 = strides

        def _make_downsample(channels: int, stride: Tuple[int, int, int]):
            if use_strided_conv:
                return nn.Conv3d(channels, channels, kernel_size=stride, stride=stride, padding=0, bias=False)
            else:
                return nn.MaxPool3d(kernel_size=stride, stride=stride)

        # Encoder stages & Strides
        self.enc1 = VNetResidualBlock(in_channels, f[0], kernel_size=k1, activation=self.activation)
        self.pool1 = _make_downsample(f[0], stride=stride1)

        self.enc2 = VNetResidualBlock(f[0], f[1], kernel_size=k2, activation=self.activation)
        self.pool2 = _make_downsample(f[1], stride=stride2)

        self.enc3 = VNetResidualBlock(f[1], f[2], kernel_size=k3, activation=self.activation)
        self.pool3 = _make_downsample(f[2], stride=stride3)

        self.enc4 = VNetResidualBlock(f[2], f[3], kernel_size=k4, activation=self.activation)
        self.pool4 = _make_downsample(f[3], stride=stride4)

        self.enc5 = VNetResidualBlock(f[3], f[4], kernel_size=k5, activation=self.activation)
        self.pool5 = _make_downsample(f[4], stride=stride5)

        self.enc6 = VNetResidualBlock(f[4], f[5], kernel_size=k6, activation=self.activation)
        self.pool6 = _make_downsample(f[5], stride=stride6)

        self.bottleneck = VNetResidualBlock(f[5], f[6], kernel_size=k_btn, activation=self.activation)

        # Decoder stages (skip_mode = 'concat' -> in_channels = f * 2; skip_mode = 'add' -> in_channels = f)
        dec_in6 = f[5] * 2 if self.skip_mode == "concat" else f[5]
        self.up6 = nn.ConvTranspose3d(f[6], f[5], kernel_size=stride6, stride=stride6)
        self.dec6 = VNetResidualBlock(dec_in6, f[5], kernel_size=k6, activation=self.activation)

        dec_in5 = f[4] * 2 if self.skip_mode == "concat" else f[4]
        self.up5 = nn.ConvTranspose3d(f[5], f[4], kernel_size=stride5, stride=stride5)
        self.dec5 = VNetResidualBlock(dec_in5, f[4], kernel_size=k5, activation=self.activation)

        self.dropout = nn.Dropout3d(p=dropout_prob) if has_dropout else nn.Identity()

        dec4_in = f[4]
        dec_in4 = f[3] * 2 if self.skip_mode == "concat" else f[3]
        self.up4 = nn.ConvTranspose3d(dec4_in, f[3], kernel_size=stride4, stride=stride4)
        self.dec4 = VNetResidualBlock(dec_in4, f[3], kernel_size=k4, activation=self.activation)

        dec_in3 = f[2] * 2 if self.skip_mode == "concat" else f[2]
        self.up3 = nn.ConvTranspose3d(f[3], f[2], kernel_size=stride3, stride=stride3)
        self.dec3 = VNetResidualBlock(dec_in3, f[2], kernel_size=k3, activation=self.activation)

        dec_in2 = f[1] * 2 if self.skip_mode == "concat" else f[1]
        self.up2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=stride2, stride=stride2)
        self.dec2 = VNetResidualBlock(dec_in2, f[1], kernel_size=k2, activation=self.activation)

        dec_in1 = f[0] * 2 if self.skip_mode == "concat" else f[0]
        self.up1 = nn.ConvTranspose3d(f[1], f[0], kernel_size=stride1, stride=stride1)
        self.dec1 = VNetResidualBlock(dec_in1, f[0], kernel_size=k1, activation=self.activation)

        # Final segmentation head (1x1x1 Conv)
        self.final_conv = nn.Conv3d(f[0], num_classes, kernel_size=1)

        # Auxiliary heads for deep supervision
        if self.deep_supervision:
            self.ds_conv2 = nn.Conv3d(f[1], num_classes, kernel_size=1)
            self.ds_conv3 = nn.Conv3d(f[2], num_classes, kernel_size=1)
            self.ds_conv4 = nn.Conv3d(f[3], num_classes, kernel_size=1)
            self.ds_conv5 = nn.Conv3d(f[4], num_classes, kernel_size=1)
            self.ds_conv6 = nn.Conv3d(f[5], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        nonlinearity = "relu" if self.activation == "relu" else "leaky_relu"
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_out", nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _combine_skip(self, up_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        if self.skip_mode == "add":
            return up_feat + skip_feat
        return torch.cat([up_feat, skip_feat], dim=1)

    def forward(self, x: torch.Tensor, return_all_scales: bool = False):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))
        x6 = self.enc6(self.pool5(x5))
        b = self.bottleneck(self.pool6(x6))
        b = self.dropout(b)

        d6 = self.up6(b)
        d6 = self._combine_skip(d6, x6)
        d6 = self.dec6(d6)

        d5 = self.up5(d6)
        d5 = self._combine_skip(d5, x5)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = self._combine_skip(d4, x4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._combine_skip(d3, x3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._combine_skip(d2, x2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._combine_skip(d1, x1)
        d1 = self.dec1(d1)

        logits = self.final_conv(d1)

        if self.deep_supervision and (self.training or return_all_scales):
            aux = [
                self.ds_conv2(d2),  # Scale 0.5
                self.ds_conv3(d3),  # Scale 0.25
                self.ds_conv4(d4),  # Scale 0.125
                self.ds_conv5(d5),  # Scale 0.0625
                self.ds_conv6(d6),  # Scale 0.03125
            ]
            return {
                "seg": logits,
                "aux": aux
            }
        return logits

