"""
V-Net 3D Volumetric Segmentation Architecture and Modular Encoder/Decoder Blocks.
Reference: Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", 3DV 2016.
"""

from typing import List, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F

from core.registry import MODELS


class ConvBlock(nn.Module):
    def __init__(
        self,
        n_stages: int,
        n_filters_in: int,
        n_filters_out: int,
        kernel_size: int = 3,
        padding: int = 1,
        normalization: str = 'none'
    ):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, kernel_size=kernel_size, padding=padding))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError(f"Unknown normalization: {normalization}")
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        n_stages: int,
        n_filters_in: int,
        n_filters_out: int,
        normalization: str = 'none'
    ):
        super(ResidualConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError(f"Unknown normalization: {normalization}")
            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + x)


class DownsamplingConvBlock(nn.Module):
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        stride: int = 2,
        padding: int = 0,
        normalization: str = 'none'
    ):
        super(DownsamplingConvBlock, self).__init__()
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride, padding=padding))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unknown normalization: {normalization}")
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        stride: int = 2,
        padding: int = 0,
        normalization: str = 'none'
    ):
        super(UpsamplingDeconvBlock, self).__init__()
        ops = []
        ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride, padding=padding))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unknown normalization: {normalization}")
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsampling(nn.Module):
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        stride: int = 2,
        normalization: str = 'none'
    ):
        super(Upsampling, self).__init__()
        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unknown normalization: {normalization}")
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VNetEncoder(nn.Module):
    """V-Net Feature Extractor (Encoder)."""
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: int = 16,
        normalization: str = 'batchnorm',
        has_dropout: bool = False,
        has_residual: bool = False
    ):
        super(VNetEncoder, self).__init__()
        self.has_dropout = has_dropout
        conv_block = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = conv_block(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = conv_block(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = conv_block(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = conv_block(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = conv_block(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

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
    """V-Net Feature Reconstructor (Decoder)."""
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: int = 16,
        normalization: str = 'batchnorm',
        has_dropout: bool = False,
        has_residual: bool = False
    ):
        super(VNetDecoder, self).__init__()
        self.has_dropout = has_dropout
        conv_block = ConvBlock if not has_residual else ResidualConvBlock
        upsampling = UpsamplingDeconvBlock

        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_six = conv_block(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = upsampling(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = conv_block(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = conv_block(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = upsampling(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = conv_block(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2, x3, x4, x5 = features
        x6 = self.block_six(self.block_five_up(x5) + x4)
        x7 = self.block_seven(self.block_six_up(x6) + x3)
        x8 = self.block_eight(self.block_seven_up(x7) + x2)
        x8_up = self.block_eight_up(x8) + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, x8


# For exact checkpoint compatibility, retain Encoder and Decoder class names
Encoder = VNetEncoder
Decoder = VNetDecoder


@MODELS.register("vnet_single")
class VNet(nn.Module):
    """Standard Single-Head VNet for supervised baseline and general segmentation."""
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        n_filters: int = 16,
        normalization: str = 'batchnorm',
        has_dropout: bool = True,
        has_residual: bool = False
    ):
        super(VNet, self).__init__()
        self.encoder = VNetEncoder(
            n_channels=in_channels,
            n_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual
        )
        self.decoder = VNetDecoder(
            n_channels=in_channels,
            n_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual
        )

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        out_seg, feat_de = self.decoder(features)
        return out_seg
