"""
3D Spatial Sampling Building Blocks.
Downsampling and upsampling modules (strided conv, transposed conv, trilinear, nearest).
"""

from typing import Optional
import torch
from torch import nn

from .norm_act import get_activation, get_norm3d


class DownsamplingConvBlock(nn.Module):
    """
    Strided Conv3D downsampling block with optional normalization and activation.
    Preserves exact Sequential indexing: [Conv3d, (Norm if != 'none'), Act].
    """
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        stride: int = 2,
        padding: int = 0,
        normalization: str = "none",
        activation: str = "relu",
        inplace: bool = False,
    ):
        super(DownsamplingConvBlock, self).__init__()
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride, padding=padding))
        norm_layer = get_norm3d(normalization, n_filters_out, affine=True)
        if norm_layer is not None:
            ops.append(norm_layer)
        ops.append(get_activation(activation, inplace=inplace))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    """
    Strided Transposed Conv3D upsampling block with optional normalization and activation.
    Preserves exact Sequential indexing: [ConvTranspose3d, (Norm if != 'none'), Act].
    """
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        stride: int = 2,
        padding: int = 0,
        normalization: str = "none",
        activation: str = "relu",
        inplace: bool = False,
    ):
        super(UpsamplingDeconvBlock, self).__init__()
        ops = []
        ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride, padding=padding))
        norm_layer = get_norm3d(normalization, n_filters_out, affine=True)
        if norm_layer is not None:
            ops.append(norm_layer)
        ops.append(get_activation(activation, inplace=inplace))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsampling(nn.Module):
    """
    Trilinear interpolation followed by 3x3x3 Conv3D upsampling block.
    Preserves exact Sequential indexing: [Upsample, Conv3d, (Norm if != 'none'), Act].
    """
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        stride: int = 2,
        normalization: str = "none",
        activation: str = "relu",
        inplace: bool = False,
    ):
        super(Upsampling, self).__init__()
        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        norm_layer = get_norm3d(normalization, n_filters_out, affine=True)
        if norm_layer is not None:
            ops.append(norm_layer)
        ops.append(get_activation(activation, inplace=inplace))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsamplingFunction(nn.Module):
    """
    Multi-mode upsampling function used in dual-decoder networks like LeFeD:
    mode 0: ConvTranspose3d
    mode 1: Trilinear Upsample + Conv3d
    mode 2: Nearest Upsample + Conv3d
    """
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        stride: int = 2,
        normalization: str = "groupnorm",
        mode_upsampling: int = 1,
        inplace: bool = True,
    ):
        super(UpsamplingFunction, self).__init__()
        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride, padding=0))
        elif mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        else:
            raise ValueError(f"Unsupported mode_upsampling: {mode_upsampling}")

        norm_layer = get_norm3d(normalization, n_filters_out, affine=True)
        if norm_layer is not None:
            ops.append(norm_layer)
        ops.append(nn.ReLU(inplace=inplace))

        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


__all__ = [
    "DownsamplingConvBlock",
    "UpsamplingDeconvBlock",
    "Upsampling",
    "UpsamplingFunction",
]

