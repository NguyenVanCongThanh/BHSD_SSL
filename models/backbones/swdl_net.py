"""
SWDL Architecture (Stratum-Wise Difference Learning Network with Deep Laplacian Pyramid).
Reference: Wang et al., "SWDL: Stratum-Wise Difference Learning with Deep Laplacian Pyramid
for Semi-Supervised 3D Intracranial Hemorrhage Segmentation", BSPC 2026.

Integrates:
- Depthwise Gaussian Blurring & Laplacian Pyramid 3D Upsampling for edge-preserving feature recovery.
- Multi-stratum dual feedback decoding with precise channel alignment for subtle intracranial hemorrhage lesions.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODELS
from models.blocks import (
    ConvBlock,
    ResidualConvBlock,
    DownsamplingConvBlock,
)
from models.blocks.norm_act import get_norm3d


class DepthwiseGaussianBlur2D(nn.Module):
    """Slice-wise Depthwise Gaussian Smoothing for volumetric tensors."""
    def __init__(self, channels: int = 1, kernel_size: int = 5, sigma: float = 1.0):
        super(DepthwiseGaussianBlur2D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        self.channels = channels
        self.register_buffer('kernel', self._create_gaussian_kernel(kernel_size, sigma))

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(-self.padding, self.padding + 1, dtype=torch.float32)
        y = torch.arange(-self.padding, self.padding + 1, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(grid_x.pow(2) + grid_y.pow(2)) / (2.0 * (sigma ** 2)))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(self.channels, 1, 1, 1)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width = x.shape
        if channels != self.channels:
            kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma).to(x.device, dtype=x.dtype)
        else:
            kernel = self.kernel.to(x.device, dtype=x.dtype)

        slices = []
        for d in range(depth):
            sl = x[:, :, d, :, :]
            sl_blurred = F.conv2d(sl, kernel, padding=self.padding, groups=channels)
            slices.append(sl_blurred.unsqueeze(2))
        return torch.cat(slices, dim=2)


class LaplacianUpsample3D(nn.Module):
    """Laplacian Pyramid 3D Upsampling layer for stratum-wise multi-scale reconstruction."""
    def __init__(self, scale_factor: float = 2.0, mu: float = 1.5):
        super(LaplacianUpsample3D, self).__init__()
        self.scale_factor = scale_factor
        self.mu = mu

    def get_levels(self, height: int, width: int, depth: int) -> int:
        max_dim = max(height, width, depth)
        if max_dim <= 8:
            return 0
        elif max_dim <= 32:
            return 1
        elif max_dim <= 64:
            return 2
        elif max_dim <= 128:
            return 3
        elif max_dim <= 256:
            return 4
        else:
            return 5

    def get_kernel_size(self, min_dim: int) -> int:
        if min_dim >= 256:
            return 5
        elif min_dim == 128:
            return 4
        else:
            return 3

    def build_gaussian_pyramid(self, x: torch.Tensor, levels: int) -> List[torch.Tensor]:
        pyramid = [x]
        curr = x
        for _ in range(levels):
            min_dim = min(curr.shape[2], curr.shape[3], curr.shape[4])
            k_size = self.get_kernel_size(min_dim)
            blur = DepthwiseGaussianBlur2D(channels=curr.shape[1], kernel_size=k_size, sigma=1.0)
            curr = blur(curr)
            curr = F.interpolate(curr, scale_factor=0.5, mode='trilinear', align_corners=True)
            pyramid.append(curr)
        return pyramid

    def build_laplacian_pyramid(self, gaussian_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        pyramid = []
        for i in range(len(gaussian_pyramid) - 1):
            min_dim = min(gaussian_pyramid[i + 1].shape[3], gaussian_pyramid[i + 1].shape[4])
            k_size = self.get_kernel_size(min_dim)
            blur = DepthwiseGaussianBlur2D(channels=gaussian_pyramid[i + 1].shape[1], kernel_size=k_size, sigma=1.0)
            upsampled = F.interpolate(gaussian_pyramid[i + 1], size=gaussian_pyramid[i].shape[2:], mode='trilinear', align_corners=True)
            upsampled = blur(upsampled)
            laplacian = gaussian_pyramid[i] - upsampled
            pyramid.append(laplacian)
        pyramid.append(gaussian_pyramid[-1])
        return pyramid

    def reconstruct_image(self, laplacian_pyramid: List[torch.Tensor], target_size: Tuple[int, int, int], mu: float) -> torch.Tensor:
        image = laplacian_pyramid[-1]
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            min_dim = min(laplacian_pyramid[i].shape[2], laplacian_pyramid[i].shape[3], laplacian_pyramid[i].shape[4])
            k_size = self.get_kernel_size(min_dim)
            blur = DepthwiseGaussianBlur2D(channels=laplacian_pyramid[i].shape[1], kernel_size=k_size, sigma=1.0)
            upsampled = F.interpolate(image, size=laplacian_pyramid[i].shape[2:], mode='trilinear', align_corners=True)
            upsampled = blur(upsampled)
            image = upsampled + mu * laplacian_pyramid[i]
        image = F.interpolate(image, size=target_size, mode='trilinear', align_corners=True)
        return image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, depth, height, width = x.shape
        levels = self.get_levels(height, width, depth)
        if levels == 0:
            return F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
        target_size = (int(depth * self.scale_factor), int(height * self.scale_factor), int(width * self.scale_factor))
        g_pyr = self.build_gaussian_pyramid(x, levels)
        l_pyr = self.build_laplacian_pyramid(g_pyr)
        return self.reconstruct_image(l_pyr, target_size, self.mu)


class SWDLUpsamplingFunction(nn.Module):
    """
    Upsampling module for SWDL:
    - Mode 0 (Standard): ConvTranspose3d upsampling.
    - Mode 1 (Laplacian): LaplacianUpsample3D followed by Conv3D for precise edge reconstruction.
    """
    def __init__(self, n_filters_in: int, n_filters_out: int, stride: int = 2, normalization: str = 'groupnorm', mode_upsampling: int = 1):
        super(SWDLUpsamplingFunction, self).__init__()
        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride, padding=0))
        elif mode_upsampling == 1:
            ops.append(LaplacianUpsample3D(scale_factor=float(stride), mu=1.5))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        else:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        norm_layer = get_norm3d(normalization, n_filters_out, affine=True)
        if norm_layer is not None:
            ops.append(norm_layer)
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SWDLEncoder(nn.Module):
    """SWDL Feature Encoder with multi-level feedback receptors matching decoder levels."""
    def __init__(self, in_channels: int = 1, n_filters: int = 16, normalization: str = 'groupnorm', has_dropout: bool = False, has_residual: bool = False):
        super(SWDLEncoder, self).__init__()
        self.has_dropout = has_dropout
        block_cls = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = block_cls(1, in_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = block_cls(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = block_cls(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = block_cls(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = block_cls(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.dropout = nn.Dropout3d(p=0.3, inplace=False)

    def forward(self, x: torch.Tensor, en: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        if en is not None and len(en) == 5:
            x1 = self.block_one(x) + en[4]
            x1_dw = self.block_one_dw(x1)

            x2 = self.block_two(x1_dw) + en[3]
            x2_dw = self.block_two_dw(x2)

            x3 = self.block_three(x2_dw) + en[2]
            x3_dw = self.block_three_dw(x3)

            x4 = self.block_four(x3_dw) + en[1]
            x4_dw = self.block_four_dw(x4)

            x5 = self.block_five(x4_dw) + en[0]
            if self.has_dropout:
                x5 = self.dropout(x5)
        else:
            x1 = self.block_one(x)
            x1_dw = self.block_one_dw(x1)

            x2 = self.block_two(x1_dw)
            x2_dw = self.block_two_dw(x2)

            x3 = self.block_three(x2_dw)
            x3_dw = self.block_three_dw(x3)

            x4 = self.block_four(x3_dw)
            x4_dw = self.block_four_dw(x4)

            x5 = self.block_five(x4_dw)
            if self.has_dropout:
                x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]


class SWDLDecoder(nn.Module):
    """Decoder incorporating SWDLUpsamplingFunction for stratum-wise reconstruction."""
    def __init__(self, num_classes: int = 6, n_filters: int = 16, normalization: str = 'groupnorm', has_dropout: bool = False, has_residual: bool = False, up_type: int = 0):
        super(SWDLDecoder, self).__init__()
        self.up_type = up_type
        self.has_dropout = has_dropout
        block_cls = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = SWDLUpsamplingFunction(n_filters * 16, n_filters * 8, stride=2, normalization=normalization, mode_upsampling=up_type)
        self.block_six = block_cls(3, n_filters * 8, n_filters * 8, normalization=normalization)

        self.block_six_up = SWDLUpsamplingFunction(n_filters * 8, n_filters * 4, stride=2, normalization=normalization, mode_upsampling=up_type)
        self.block_seven = block_cls(3, n_filters * 4, n_filters * 4, normalization=normalization)

        self.block_seven_up = SWDLUpsamplingFunction(n_filters * 4, n_filters * 2, stride=2, normalization=normalization, mode_upsampling=up_type)
        self.block_eight = block_cls(2, n_filters * 2, n_filters * 2, normalization=normalization)

        self.block_eight_up = SWDLUpsamplingFunction(n_filters * 2, n_filters, stride=2, normalization=normalization, mode_upsampling=up_type)
        self.block_nine = block_cls(1, n_filters, n_filters, normalization=normalization)

        self.out_conv = nn.Conv3d(n_filters, num_classes, kernel_size=1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features: List[torch.Tensor], f1: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x1, x2, x3, x4, x5 = features[0], features[1], features[2], features[3], features[4]

        if f1 is None:
            x5_up_ori = self.block_five_up(x5)
            x6 = self.block_six(x5_up_ori)

            x6_up_ori = self.block_six_up(x6)
            x7 = self.block_seven(x6_up_ori)

            x7_up_ori = self.block_seven_up(x7)
            x8 = self.block_eight(x7_up_ori)

            x8_up_ori = self.block_eight_up(x8)
            x9 = self.block_nine(x8_up_ori)
        else:
            m5, m4, m3, m2, m1 = f1[0], f1[1], f1[2], f1[3], f1[4]
            w5 = torch.sigmoid(m5).detach()
            w4 = torch.sigmoid(m4).detach()
            w3 = torch.sigmoid(m3).detach()
            w2 = torch.sigmoid(m2).detach()
            w1 = torch.sigmoid(m1).detach()

            x5_mod = x5 + x5 * w5
            x5_up_ori = self.block_five_up(x5_mod)
            x5_up = x5_up_ori + x4 * w4
            x6 = self.block_six(x5_up)

            x6_up_ori = self.block_six_up(x6)
            x6_up = x6_up_ori + x3 * w3
            x7 = self.block_seven(x6_up)

            x7_up_ori = self.block_seven_up(x7)
            x7_up = x7_up_ori + x2 * w2
            x8 = self.block_eight(x7_up)

            x8_up_ori = self.block_eight_up(x8)
            x8_up = x8_up_ori + x1 * w1
            x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, [x5, x5_up_ori, x6_up_ori, x7_up_ori, x8_up_ori]


class SWDLSideConv(nn.Module):
    """Side projection heads generating multi-scale deep supervision outputs."""
    def __init__(self, num_classes: int = 6, n_filters: int = 16):
        super(SWDLSideConv, self).__init__()
        self.side5 = nn.Conv3d(n_filters * 16, num_classes, kernel_size=1)
        self.side4 = nn.Conv3d(n_filters * 8, num_classes, kernel_size=1)
        self.side3 = nn.Conv3d(n_filters * 4, num_classes, kernel_size=1)
        self.side2 = nn.Conv3d(n_filters * 2, num_classes, kernel_size=1)
        self.side1 = nn.Conv3d(n_filters, num_classes, kernel_size=1)

    def forward(self, stage_feat: List[torch.Tensor], target_shape: Tuple[int, int, int]) -> List[torch.Tensor]:
        x5, x5_up, x6_up, x7_up, x8_up = stage_feat[0], stage_feat[1], stage_feat[2], stage_feat[3], stage_feat[4]
        out5 = F.interpolate(self.side5(x5), size=target_shape, mode='trilinear', align_corners=False)
        out4 = F.interpolate(self.side4(x5_up), size=target_shape, mode='trilinear', align_corners=False)
        out3 = F.interpolate(self.side3(x6_up), size=target_shape, mode='trilinear', align_corners=False)
        out2 = F.interpolate(self.side2(x7_up), size=target_shape, mode='trilinear', align_corners=False)
        out1 = F.interpolate(self.side1(x8_up), size=target_shape, mode='trilinear', align_corners=False)
        return [out5, out4, out3, out2, out1]


@MODELS.register("swdl_net")
@MODELS.register("swdl")
class SWDLNet(nn.Module):
    """
    SWDL Network (BSPC 2026):
    Dual Decoders (Standard vs Laplacian Pyramid) communicating via stratum-wise difference feedback.
    """
    CAPABILITIES = {"segmentation", "dual_feedback", "swdl_laplacian"}

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        n_filters: int = 16,
        normalization: str = 'groupnorm',
        has_dropout: bool = False,
        has_residual: bool = False
    ):
        super(SWDLNet, self).__init__()
        self.num_classes = num_classes

        self.encoder = SWDLEncoder(
            in_channels=in_channels,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual
        )
        self.decoder1 = SWDLDecoder(
            num_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
            up_type=0
        )
        self.decoder2 = SWDLDecoder(
            num_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
            up_type=1
        )
        self.sideconv = SWDLSideConv(num_classes=num_classes, n_filters=n_filters)

    def forward(
        self,
        x: torch.Tensor,
        en: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]], List[torch.Tensor]]:
        features = self.encoder(x, en)
        out_seg1, stage_feat1 = self.decoder1(features)

        if not self.training and en is None:
            return out_seg1

        out_seg2, stage_feat2 = self.decoder2(features, stage_feat1)
        deep_outs = self.sideconv(stage_feat1, target_shape=x.shape[2:])
        return out_seg1, out_seg2, [stage_feat2, stage_feat1], deep_outs
