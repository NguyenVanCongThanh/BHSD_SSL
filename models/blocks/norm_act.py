"""
3D Normalization and Activation Layer Factories.
Provides configurable factory functions for standard 3D medical image segmentation models.
"""

from typing import Optional
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


__all__ = ["get_norm3d", "get_activation"]

