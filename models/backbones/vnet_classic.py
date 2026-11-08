"""
Classic 5-Stage V-Net Architecture (Milletari et al., 3DV 2016).
Provides modular VNetEncoder and VNetDecoder, used for 5-stage baselines and PICK multi-decoder models.
"""

from typing import List, Tuple
import torch
from torch import nn

from models.blocks import (
    ConvBlock,
    ResidualConvBlock,
    DownsamplingConvBlock,
    UpsamplingDeconvBlock,
)

LEGACY_DROPOUT_PROB = 0.5


class VNetEncoder(nn.Module):
    """V-Net Feature Extractor (5-Stage Encoder)."""
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: int = 16,
        normalization: str = "batchnorm",
        has_dropout: bool = False,
        has_residual: bool = False,
        dropout_prob: float = LEGACY_DROPOUT_PROB,
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
        self.dropout = nn.Dropout3d(p=dropout_prob, inplace=False)

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
    """V-Net Feature Reconstructor (5-Stage Decoder)."""
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: int = 16,
        normalization: str = "batchnorm",
        has_dropout: bool = False,
        has_residual: bool = False,
        dropout_prob: float = LEGACY_DROPOUT_PROB,
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
        self.dropout = nn.Dropout3d(p=dropout_prob, inplace=False)

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


# Backward-compatible aliases for historical imports
Encoder = VNetEncoder
Decoder = VNetDecoder

