"""
3D U-Net Architecture for Volumetric Medical Image Segmentation.
Reference: Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation", MICCAI 2016.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from torch.nn import (
    Module, Sequential, Conv3d, ConvTranspose3d,
    BatchNorm3d, MaxPool3d, Dropout3d, ReLU
)

from core.registry import MODELS


class Conv3DBlock(Module):
    def __init__(self, in_feat: int, out_feat: int, kernel: int = 3, stride: int = 1, padding: int = 1, residual: Optional[str] = None):
        super(Conv3DBlock, self).__init__()
        self.conv1 = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU(inplace=True)
        )
        self.residual = residual
        if self.residual is not None:
            self.residual_upsampler = Conv3d(in_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.conv2(self.conv1(x))
        if self.residual is not None:
            out = out + self.residual_upsampler(res)
        return out


class Deconv3DBlock(Module):
    def __init__(self, in_feat: int, out_feat: int, kernel: int = 3, stride: int = 2, padding: int = 1):
        super(Deconv3DBlock, self).__init__()
        self.deconv = Sequential(
            ConvTranspose3d(
                in_feat, out_feat,
                kernel_size=kernel, stride=stride, padding=padding, output_padding=1, bias=True
            ),
            ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


@MODELS.register("unet3d")
class UNet3D(Module):
    """
    3D U-Net segmentation network with configurable feature channels and skip connections.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        feat_channels: Optional[List[int]] = None,
        residual: Optional[str] = 'conv',
        has_dropout: bool = True
    ):
        super(UNet3D, self).__init__()
        if feat_channels is None:
            feat_channels = [32, 64, 128, 256, 512]

        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv3DBlock(in_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3DBlock(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3DBlock(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3DBlock(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3DBlock(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3DBlock(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3DBlock(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3DBlock(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3DBlock(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3DBlock(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3DBlock(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3DBlock(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3DBlock(feat_channels[1], feat_channels[0])

        self.has_dropout = has_dropout
        self.dropout = Dropout3d(p=0.5) if has_dropout else nn.Identity()

        self.out_conv = Conv3d(feat_channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.conv_blk1(x)
        x2 = self.conv_blk2(self.pool1(x1))
        x3 = self.conv_blk3(self.pool2(x2))
        x4 = self.conv_blk4(self.pool3(x3))
        base = self.conv_blk5(self.pool4(x4))

        # Decoder with skip connections
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dropout(self.dec_conv_blk3(d3))

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dropout(self.dec_conv_blk2(d2))

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        seg = self.out_conv(d_high1)
        return seg
