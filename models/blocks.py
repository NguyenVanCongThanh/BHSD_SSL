"""
Shared 3D Convolutional, Residual, Normalization, and Sampling Blocks.
Reusable building blocks for 3D medical volumetric segmentation architectures (V-Net, LeFeD, U-Net).
"""

from typing import List, Optional, Tuple, Union
import torch
from torch import nn


def get_norm3d(
    normalization: str,
    num_channels: int,
    num_groups: Optional[int] = None,
    affine: bool = True,
) -> Optional[nn.Module]:
    """Factory function for 3D normalization layers."""
    norm = str(normalization).lower()
    if norm in ("none", "", "identity"):
        return None
    elif norm == "batchnorm":
        return nn.BatchNorm3d(num_channels, affine=affine)
    elif norm == "groupnorm":
        groups = min(16, num_channels) if num_groups is None else num_groups
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels, affine=affine)
    elif norm == "instancenorm":
        return nn.InstanceNorm3d(num_channels, affine=affine)
    else:
        raise ValueError(f"Unknown or unsupported 3D normalization: {normalization}")


def get_activation(
    activation: str,
    inplace: bool = False,
    negative_slope: float = 0.01,
) -> nn.Module:
    """Factory function for activation layers."""
    act = str(activation).lower()
    if act == "relu":
        return nn.ReLU(inplace=inplace)
    elif act == "prelu":
        return nn.PReLU()
    elif act in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif act in ("none", "", "identity"):
        return nn.Identity()
    else:
        raise ValueError(f"Unknown or unsupported activation: {activation}")


class ConvBlock(nn.Module):
    """
    Standard sequential Conv3D block with configurable stages, normalization, and activation.
    Preserves exact Sequential indexing: [Conv3d, (Norm if != 'none'), Act, Conv3d, ...].
    """
    def __init__(
        self,
        n_stages: int,
        n_filters_in: int,
        n_filters_out: int,
        kernel_size: int = 3,
        padding: int = 1,
        normalization: str = "none",
        activation: str = "relu",
        inplace: bool = False,
    ):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, kernel_size=kernel_size, padding=padding))
            norm_layer = get_norm3d(normalization, n_filters_out, affine=True)
            if norm_layer is not None:
                ops.append(norm_layer)
            ops.append(get_activation(activation, inplace=inplace))

        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    """
    Residual Conv3D block adding the input identity to the output of multi-stage convolutions.
    Preserves exact Sequential indexing: [Conv3d, (Norm if != 'none'), Act, ..., Conv3d, (Norm if != 'none')].
    """
    def __init__(
        self,
        n_stages: int,
        n_filters_in: int,
        n_filters_out: int,
        normalization: str = "none",
        activation: str = "relu",
        inplace: bool = False,
    ):
        super(ResidualConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, 3, padding=1))
            norm_layer = get_norm3d(normalization, n_filters_out, affine=True)
            if norm_layer is not None:
                ops.append(norm_layer)
            if i != n_stages - 1:
                ops.append(get_activation(activation, inplace=inplace))

        self.conv = nn.Sequential(*ops)
        self.relu = get_activation(activation, inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + x)


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


class VNetResidualBlock(nn.Module):
    """
    Two-convolution residual block with InstanceNorm3d and LeakyReLU/ReLU for V-Net.
    Implements intra-block residual addition: y = Act(Conv2(Norm(Conv1(x))) + Shortcut(x)).
    When in_channels != out_channels, shortcut projects via 1x1x1 Conv3D + InstanceNorm3d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = get_activation(activation, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = get_activation(activation, inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act1(self.norm1(self.conv1(x)))
        res = self.norm2(self.conv2(res))
        res = res + self.shortcut(x)
        return self.act2(res)

