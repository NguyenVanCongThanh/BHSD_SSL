"""
Net factory module (kept for backward compatibility, integrated with core.registry).
"""

from typing import Optional
import torch.nn as nn
from core.registry import MODELS
import models  # Ensure all models are registered


def net_factory(net_type: str = "VNet", in_chns: int = 1, class_num: int = 2, mode: str = "train", tsne: int = 0) -> nn.Module:
    """
    Instantiate model architecture according to net_type and parameters.
    Supports VNet, unet3d, unetr, etc.
    """
    key = net_type.lower()
    has_dropout = (mode == "train")

    if key in ("vnet", "pick_vnet"):
        net = MODELS.build("vnet", in_channels=in_chns, num_classes=class_num, has_dropout=has_dropout)
    elif key == "vnet_single":
        net = MODELS.build("vnet_single", in_channels=in_chns, num_classes=class_num, has_dropout=has_dropout)
    elif key in ("unet3d", "unet_3d"):
        net = MODELS.build("unet3d", in_channels=in_chns, num_classes=class_num, has_dropout=has_dropout)
    elif key == "unetr":
        net = MODELS.build("unetr", in_channels=in_chns, num_classes=class_num)
    elif MODELS.contains(key):
        net = MODELS.build(key, in_channels=in_chns, num_classes=class_num)
    else:
        raise ValueError(f"Unknown net_type: {net_type}. Available: {MODELS.list_keys()}")

    return net
