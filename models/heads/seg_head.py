"""
Segmentation Prediction Head.
"""

import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """Simple 1x1x1 convolutional segmentation head."""
    def __init__(self, in_channels: int, num_classes: int):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
