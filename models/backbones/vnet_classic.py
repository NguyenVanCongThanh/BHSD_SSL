"""
Classic 5-Stage V-Net Architecture (Milletari et al., 3DV 2016).
Provides modular VNetEncoder and VNetDecoder, used for 5-stage baselines and PICK multi-decoder models.
"""

from typing import List, Tuple
import torch
from torch import nn

from models.blocks import (
    ConvBlock,
    ResidualConvBlock,
    DownsamplingConvBlock,
    UpsamplingDeconvBlock,
)

LEGACY_DROPOUT_PROB = 0.5
CLASSIC_ANISOTROPIC_KERNELS: Tuple[Tuple[int, int, int], ...] = (
    (1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)
)
CLASSIC_ISOTROPIC_KERNELS: Tuple[Tuple[int, int, int], ...] = ((3, 3, 3),) * 5
CLASSIC_ANISOTROPIC_STRIDES: Tuple[Tuple[int, int, int], ...] = (
    (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)
)
CLASSIC_ISOTROPIC_STRIDES: Tuple[Tuple[int, int, int], ...] = ((2, 2, 2),) * 4


def _init_vnet_weights(module: nn.Module, activation: str) -> None:
    """Initialize 3D Conv/Norm layers using Kaiming normal aligned with activation."""
    nonlinearity = "relu" if str(activation).lower() == "relu" else "leaky_relu"
    for m in module.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_out", nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class VNetEncoder(nn.Module):
    """V-Net Feature Extractor (5-Stage Encoder)."""
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: int = 16,
        normalization: str = "instancenorm",
        has_dropout: bool = False,
        has_residual: bool = False,
        dropout_prob: float = LEGACY_DROPOUT_PROB,
        activation: str = "leaky_relu",
        anisotropic_kernels: bool = True,
        overlap_downsample: bool = False,
        outer_convs: int = 1,
    ):
        super(VNetEncoder, self).__init__()
        if int(outer_convs) < 1:
            raise ValueError(f"outer_convs must be >= 1, got {outer_convs}")
        self.has_dropout = has_dropout
        self.anisotropic_kernels = bool(anisotropic_kernels)
        self.overlap_downsample = bool(overlap_downsample)
        self.outer_convs = int(outer_convs)
        conv_block = ConvBlock if not has_residual else ResidualConvBlock
        k1, k2, k3, k4, k5 = CLASSIC_ANISOTROPIC_KERNELS if self.anisotropic_kernels else CLASSIC_ISOTROPIC_KERNELS
        s1, s2, s3, s4 = CLASSIC_ANISOTROPIC_STRIDES if self.anisotropic_kernels else CLASSIC_ISOTROPIC_STRIDES
        dw_k = (k1, k2, k3, k4) if self.overlap_downsample else (None, None, None, None)
        dw_pad = None if self.overlap_downsample else 0

        self.block_one = conv_block(self.outer_convs, n_channels, n_filters, kernel_size=k1, normalization=normalization, activation=activation)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, stride=s1, padding=dw_pad, kernel_size=dw_k[0], normalization=normalization, activation=activation)

        self.block_two = conv_block(2, n_filters * 2, n_filters * 2, kernel_size=k2, normalization=normalization, activation=activation)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, stride=s2, padding=dw_pad, kernel_size=dw_k[1], normalization=normalization, activation=activation)

        self.block_three = conv_block(3, n_filters * 4, n_filters * 4, kernel_size=k3, normalization=normalization, activation=activation)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, stride=s3, padding=dw_pad, kernel_size=dw_k[2], normalization=normalization, activation=activation)

        self.block_four = conv_block(3, n_filters * 8, n_filters * 8, kernel_size=k4, normalization=normalization, activation=activation)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, stride=s4, padding=dw_pad, kernel_size=dw_k[3], normalization=normalization, activation=activation)

        self.block_five = conv_block(3, n_filters * 16, n_filters * 16, kernel_size=k5, normalization=normalization, activation=activation)
        self.dropout = nn.Dropout3d(p=dropout_prob, inplace=False)
        _init_vnet_weights(self, activation)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.block_one(x)
        x2 = self.block_two(self.block_one_dw(x1))
        x3 = self.block_three(self.block_two_dw(x2))
        x4 = self.block_four(self.block_three_dw(x3))
        x5 = self.block_five(self.block_four_dw(x4))
        if self.has_dropout:
            x5 = self.dropout(x5)
        return [x1, x2, x3, x4, x5]


class VNetDecoder(nn.Module):
    """V-Net Feature Reconstructor (5-Stage Decoder)."""
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: int = 16,
        normalization: str = "instancenorm",
        has_dropout: bool = False,
        has_residual: bool = False,
        dropout_prob: float = LEGACY_DROPOUT_PROB,
        skip_mode: str = "concat",
        activation: str = "leaky_relu",
        anisotropic_kernels: bool = True,
        deep_supervision: bool = False,
        outer_convs: int = 1,
    ):
        super(VNetDecoder, self).__init__()
        if int(outer_convs) < 1:
            raise ValueError(f"outer_convs must be >= 1, got {outer_convs}")
        self.has_dropout = has_dropout
        self.anisotropic_kernels = bool(anisotropic_kernels)
        self.deep_supervision = bool(deep_supervision)
        self.outer_convs = int(outer_convs)
        self.skip_mode = str(skip_mode).lower()
        if self.skip_mode not in ("concat", "add"):
            raise ValueError(f"skip_mode must be 'concat' or 'add', got {self.skip_mode}")
        skip_mult = 2 if self.skip_mode == "concat" else 1
        conv_block = ConvBlock if not has_residual else ResidualConvBlock
        upsampling = UpsamplingDeconvBlock
        k1, k2, k3, k4, _ = CLASSIC_ANISOTROPIC_KERNELS if self.anisotropic_kernels else CLASSIC_ISOTROPIC_KERNELS
        s1, s2, s3, s4 = CLASSIC_ANISOTROPIC_STRIDES if self.anisotropic_kernels else CLASSIC_ISOTROPIC_STRIDES

        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, stride=s4, normalization=normalization, activation=activation)
        self.block_six = conv_block(3, n_filters * 8 * skip_mult, n_filters * 8, kernel_size=k4, normalization=normalization, activation=activation)
        self.block_six_up = upsampling(n_filters * 8, n_filters * 4, stride=s3, normalization=normalization, activation=activation)

        self.block_seven = conv_block(3, n_filters * 4 * skip_mult, n_filters * 4, kernel_size=k3, normalization=normalization, activation=activation)
        self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, stride=s2, normalization=normalization, activation=activation)

        self.block_eight = conv_block(2, n_filters * 2 * skip_mult, n_filters * 2, kernel_size=k2, normalization=normalization, activation=activation)
        self.block_eight_up = upsampling(n_filters * 2, n_filters, stride=s1, normalization=normalization, activation=activation)

        self.block_nine = conv_block(self.outer_convs, n_filters * skip_mult, n_filters, kernel_size=k1, normalization=normalization, activation=activation)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        if self.deep_supervision:
            self.ds_conv2 = nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)
            self.ds_conv3 = nn.Conv3d(n_filters * 4, n_classes, 1, padding=0)
            self.ds_conv4 = nn.Conv3d(n_filters * 8, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=dropout_prob, inplace=False)
        _init_vnet_weights(self, activation)

    def _combine_skip(self, up_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        if self.skip_mode == "concat":
            return torch.cat([up_feat, skip_feat], dim=1)
        return up_feat + skip_feat

    def forward(self, features: List[torch.Tensor], return_aux: bool = False):
        x1, x2, x3, x4, x5 = features
        x6 = self.block_six(self._combine_skip(self.block_five_up(x5), x4))
        x7 = self.block_seven(self._combine_skip(self.block_six_up(x6), x3))
        x8 = self.block_eight(self._combine_skip(self.block_seven_up(x7), x2))
        x8_up = self._combine_skip(self.block_eight_up(x8), x1)
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        if return_aux and self.deep_supervision:
            aux = [
                self.ds_conv2(x8),
                self.ds_conv3(x7),
                self.ds_conv4(x6),
            ]
            return out_seg, x8, aux
        return out_seg, x8


# Backward-compatible aliases for historical imports
Encoder = VNetEncoder
Decoder = VNetDecoder

