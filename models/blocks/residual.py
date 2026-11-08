"""
3D Architecture-specific Residual Blocks.
V-Net style intra-block residual additions with InstanceNorm3d and projection shortcuts.
"""

from typing import Tuple, Union
import torch
from torch import nn

from .norm_act import get_activation


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


__all__ = ["VNetResidualBlock"]

