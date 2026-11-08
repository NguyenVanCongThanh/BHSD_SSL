"""
3D Reusable Neural Building Blocks Package.
Exposes normalization/activation factories, convolutional blocks, sampling modules,
and residual units for 3D medical image segmentation.
"""

from .norm_act import get_activation, get_norm3d
from .conv import ConvBlock, ResidualConvBlock
from .sampling import (
    DownsamplingConvBlock,
    Upsampling,
    UpsamplingDeconvBlock,
    UpsamplingFunction,
)
from .residual import VNetResidualBlock

__all__ = [
    "get_norm3d",
    "get_activation",
    "ConvBlock",
    "ResidualConvBlock",
    "DownsamplingConvBlock",
    "UpsamplingDeconvBlock",
    "Upsampling",
    "UpsamplingFunction",
    "VNetResidualBlock",
]

