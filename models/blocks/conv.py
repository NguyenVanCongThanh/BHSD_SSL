"""
3D Convolutional Building Blocks.
Standard and residual 3D convolutional blocks with sequential indexing compatibility.
"""

from typing import Optional
import torch
from torch import nn

from .norm_act import get_activation, get_norm3d


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


__all__ = ["ConvBlock", "ResidualConvBlock"]

