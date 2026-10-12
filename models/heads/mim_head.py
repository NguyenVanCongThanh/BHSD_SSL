"""
Masked Image Modeling (MIM) Image Reconstruction Head.
"""

import torch
import torch.nn as nn


class MIMReconstructionHead(nn.Module):
    """Reconstruction head for Masked Image Modeling, outputting image channels."""
    def __init__(self, in_channels: int, out_channels: int = 1):
        super(MIMReconstructionHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
