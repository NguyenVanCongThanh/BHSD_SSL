"""
LeFeD Architecture (Learning Feedback Dual-branch Net) for 3D Semi-Supervised Medical Image Segmentation.
Reference: "Learning Feedback for Semi-Supervised 3D Medical Image Segmentation".
Adapted for BHSD 3D Brain Hemorrhage Segmentation with anisotropic V-Net alignment.
"""

from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODELS
from models.backbones.vnet_classic import (
    CLASSIC_ANISOTROPIC_KERNELS,
    CLASSIC_ANISOTROPIC_STRIDES,
    CLASSIC_ISOTROPIC_KERNELS,
    CLASSIC_ISOTROPIC_STRIDES,
    _init_vnet_weights,
)
from models.blocks import (
    ConvBlock,
    ResidualConvBlock,
    DownsamplingConvBlock,
    UpsamplingFunction,
)

DEFAULT_ENCODER_DROPOUT_PROB = 0.3
DEFAULT_DECODER_DROPOUT_PROB = 0.5


class LeFeDEncoder(nn.Module):
    """Encoder capable of receiving multi-level feedback signals (en)."""
    def __init__(
        self,
        in_channels: int = 1,
        n_filters: int = 16,
        normalization: str = 'groupnorm',
        has_dropout: bool = False,
        has_residual: bool = False,
        anisotropic_kernels: bool = True,
        activation: str = 'leaky_relu',
        dropout_prob: Optional[float] = None,
    ):
        super(LeFeDEncoder, self).__init__()
        self.has_dropout = has_dropout
        self.anisotropic_kernels = bool(anisotropic_kernels)
        block_cls = ConvBlock if not has_residual else ResidualConvBlock
        k1, k2, k3, k4, k5 = (
            CLASSIC_ANISOTROPIC_KERNELS if self.anisotropic_kernels else CLASSIC_ISOTROPIC_KERNELS
        )
        s1, s2, s3, s4 = (
            CLASSIC_ANISOTROPIC_STRIDES if self.anisotropic_kernels else CLASSIC_ISOTROPIC_STRIDES
        )

        self.block_one = block_cls(1, in_channels, n_filters, kernel_size=k1, normalization=normalization, activation=activation)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, stride=s1, normalization=normalization, activation=activation)

        self.block_two = block_cls(2, n_filters * 2, n_filters * 2, kernel_size=k2, normalization=normalization, activation=activation)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, stride=s2, normalization=normalization, activation=activation)

        self.block_three = block_cls(3, n_filters * 4, n_filters * 4, kernel_size=k3, normalization=normalization, activation=activation)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, stride=s3, normalization=normalization, activation=activation)

        self.block_four = block_cls(3, n_filters * 8, n_filters * 8, kernel_size=k4, normalization=normalization, activation=activation)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, stride=s4, normalization=normalization, activation=activation)

        self.block_five = block_cls(3, n_filters * 16, n_filters * 16, kernel_size=k5, normalization=normalization, activation=activation)
        enc_drop = DEFAULT_ENCODER_DROPOUT_PROB if dropout_prob is None else float(dropout_prob)
        self.dropout = nn.Dropout3d(p=enc_drop, inplace=False)
        _init_vnet_weights(self, activation)

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


class LeFeDDecoder(nn.Module):
    """Decoder supporting feature-guided gating from complementary decoder."""
    def __init__(
        self,
        num_classes: int = 6,
        n_filters: int = 16,
        normalization: str = 'groupnorm',
        has_dropout: bool = False,
        has_residual: bool = False,
        up_type: int = 0,
        anisotropic_kernels: bool = True,
        activation: str = 'leaky_relu',
        skip_mode: str = 'add',
        dropout_prob: Optional[float] = None,
    ):
        super(LeFeDDecoder, self).__init__()
        self.has_dropout = has_dropout
        self.anisotropic_kernels = bool(anisotropic_kernels)
        self.skip_mode = str(skip_mode).lower()
        if self.skip_mode not in ('concat', 'add'):
            raise ValueError(f"skip_mode must be 'concat' or 'add', got {self.skip_mode}")
        skip_mult = 2 if self.skip_mode == 'concat' else 1
        block_cls = ConvBlock if not has_residual else ResidualConvBlock
        k1, k2, k3, k4, _ = (
            CLASSIC_ANISOTROPIC_KERNELS if self.anisotropic_kernels else CLASSIC_ISOTROPIC_KERNELS
        )
        s1, s2, s3, s4 = (
            CLASSIC_ANISOTROPIC_STRIDES if self.anisotropic_kernels else CLASSIC_ISOTROPIC_STRIDES
        )

        self.block_five_up = UpsamplingFunction(
            n_filters * 16, n_filters * 8, stride=s4, normalization=normalization,
            mode_upsampling=up_type, activation=activation, kernel_size=k4,
        )
        self.block_six = block_cls(3, n_filters * 8 * skip_mult, n_filters * 8, kernel_size=k4, normalization=normalization, activation=activation)

        self.block_six_up = UpsamplingFunction(
            n_filters * 8, n_filters * 4, stride=s3, normalization=normalization,
            mode_upsampling=up_type, activation=activation, kernel_size=k3,
        )
        self.block_seven = block_cls(3, n_filters * 4 * skip_mult, n_filters * 4, kernel_size=k3, normalization=normalization, activation=activation)

        self.block_seven_up = UpsamplingFunction(
            n_filters * 4, n_filters * 2, stride=s2, normalization=normalization,
            mode_upsampling=up_type, activation=activation, kernel_size=k2,
        )
        self.block_eight = block_cls(2, n_filters * 2 * skip_mult, n_filters * 2, kernel_size=k2, normalization=normalization, activation=activation)

        self.block_eight_up = UpsamplingFunction(
            n_filters * 2, n_filters, stride=s1, normalization=normalization,
            mode_upsampling=up_type, activation=activation, kernel_size=k1,
        )
        self.block_nine = block_cls(1, n_filters * skip_mult, n_filters, kernel_size=k1, normalization=normalization, activation=activation)

        self.out_conv = nn.Conv3d(n_filters, num_classes, 1, padding=0)
        dec_drop = DEFAULT_DECODER_DROPOUT_PROB if dropout_prob is None else float(dropout_prob)
        self.dropout = nn.Dropout3d(p=dec_drop, inplace=False)
        _init_vnet_weights(self, activation)

    def _combine_skip(self, up_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        if self.skip_mode == 'concat':
            return torch.cat([up_feat, skip_feat], dim=1)
        return up_feat + skip_feat

    def forward(self, features: List[torch.Tensor], f1: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x1, x2, x3, x4, x5 = features

        if f1 is None:
            x5_up_ori = self.block_five_up(x5)
            x5_up = self._combine_skip(x5_up_ori, x4)

            x6 = self.block_six(x5_up)
            x6_up_ori = self.block_six_up(x6)
            x6_up = self._combine_skip(x6_up_ori, x3)

            x7 = self.block_seven(x6_up)
            x7_up_ori = self.block_seven_up(x7)
            x7_up = self._combine_skip(x7_up_ori, x2)

            x8 = self.block_eight(x7_up)
            x8_up_ori = self.block_eight_up(x8)
            x8_up = self._combine_skip(x8_up_ori, x1)

            x9 = self.block_nine(x8_up)
            if self.has_dropout:
                x9 = self.dropout(x9)
            out_seg = self.out_conv(x9)
        else:
            m5, m4, m3, m2, m1 = f1
            w5, w4, w3, w2, w1 = [torch.sigmoid(m).detach() for m in [m5, m4, m3, m2, m1]]

            x5 = x5 + x5 * w5
            x5_up_ori = self.block_five_up(x5)
            x5_up = self._combine_skip(x5_up_ori, x4 * w4)

            x6 = self.block_six(x5_up)
            x6_up_ori = self.block_six_up(x6)
            x6_up = self._combine_skip(x6_up_ori, x3 * w3)

            x7 = self.block_seven(x6_up)
            x7_up_ori = self.block_seven_up(x7)
            x7_up = self._combine_skip(x7_up_ori, x2 * w2)

            x8 = self.block_eight(x7_up)
            x8_up_ori = self.block_eight_up(x8)
            x8_up = self._combine_skip(x8_up_ori, x1 * w1)

            x9 = self.block_nine(x8_up)
            if self.has_dropout:
                x9 = self.dropout(x9)
            out_seg = self.out_conv(x9)

        return out_seg, [x5, x5_up_ori, x6_up_ori, x7_up_ori, x8_up_ori]


class SideConv(nn.Module):
    """
    Deep Supervision multi-scale projection head.
    Returns logits ordered from highest resolution to lowest resolution: [out1, out2, out3, out4, out5],
    aligned with high-to-low ds_weights (e.g. [0.5, 0.25, 0.125, 0.0625, 0.03125]).
    """
    def __init__(self, num_classes: int = 6, n_filters: int = 16, upsample: bool = False):
        super(SideConv, self).__init__()
        self.upsample = bool(upsample)
        self.side5 = nn.Conv3d(n_filters * 16, num_classes, 1, padding=0)
        self.side4 = nn.Conv3d(n_filters * 8, num_classes, 1, padding=0)
        self.side3 = nn.Conv3d(n_filters * 4, num_classes, 1, padding=0)
        self.side2 = nn.Conv3d(n_filters * 2, num_classes, 1, padding=0)
        self.side1 = nn.Conv3d(n_filters, num_classes, 1, padding=0)

    def forward(self, stage_feat: List[torch.Tensor]) -> List[torch.Tensor]:
        x5, x5_up, x6_up, x7_up, x8_up = stage_feat
        out1 = self.side1(x8_up)
        out2 = self.side2(x7_up)
        out3 = self.side3(x6_up)
        out4 = self.side4(x5_up)
        out5 = self.side5(x5)

        if self.upsample:
            target_size = x8_up.shape[2:]
            out2 = F.interpolate(out2, size=target_size, mode='trilinear', align_corners=True)
            out3 = F.interpolate(out3, size=target_size, mode='trilinear', align_corners=True)
            out4 = F.interpolate(out4, size=target_size, mode='trilinear', align_corners=True)
            out5 = F.interpolate(out5, size=target_size, mode='trilinear', align_corners=True)

        return [out1, out2, out3, out4, out5]


@MODELS.register("lefed_net")
class LeFeD_Net(nn.Module):
    """
    Complete LeFeD Network:
    - Shared Feedback Encoder (supports anisotropic CT strides/kernels aligned with VNetClassic).
    - Decoder 1 (mode 0 deconvolution) + Decoder 2 (mode 1 trilinear upsample + conv).
    - SideConv for Deep Supervision (ordered finest -> coarsest).
    """
    CAPABILITIES = {"segmentation", "dual_feedback"}

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        n_filters: int = 16,
        normalization: str = 'groupnorm',
        has_dropout: bool = True,
        has_residual: bool = False,
        anisotropic_kernels: bool = True,
        activation: str = 'leaky_relu',
        skip_mode: str = 'add',
        dropout_prob: Optional[float] = None,
        eval_ensemble: bool = False,
    ):
        super(LeFeD_Net, self).__init__()
        self.num_classes = num_classes
        self.eval_ensemble = bool(eval_ensemble)

        self.encoder = LeFeDEncoder(
            in_channels=in_channels,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
            anisotropic_kernels=anisotropic_kernels,
            activation=activation,
            dropout_prob=dropout_prob,
        )
        self.decoder1 = LeFeDDecoder(
            num_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
            up_type=0,
            anisotropic_kernels=anisotropic_kernels,
            activation=activation,
            skip_mode=skip_mode,
            dropout_prob=dropout_prob,
        )
        self.decoder2 = LeFeDDecoder(
            num_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
            up_type=1,
            anisotropic_kernels=anisotropic_kernels,
            activation=activation,
            skip_mode=skip_mode,
            dropout_prob=dropout_prob,
        )
        self.sideconv = SideConv(num_classes=num_classes, n_filters=n_filters, upsample=False)

    def forward(
        self,
        x: torch.Tensor,
        en: Optional[List[torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]], List[torch.Tensor]]]:
        """
        Forward pass:
        - During inference (not training and en is None): returns out_seg1 (or 0.5*(out_seg1+out_seg2) if eval_ensemble=True).
        - During training with feedback loop: returns (out1, out2, [stage_feat2, stage_feat1], deep_outs [finest->coarsest]).
        """
        features = self.encoder(x, en)
        out_seg1, stage_feat1 = self.decoder1(features)

        if not self.training and en is None:
            if self.eval_ensemble:
                out_seg2, _ = self.decoder2(features, stage_feat1)
                return 0.5 * (out_seg1 + out_seg2)
            return out_seg1

        out_seg2, stage_feat2 = self.decoder2(features, stage_feat1)
        deep_out1 = self.sideconv(stage_feat1)

        # masks: [stage_feat2, stage_feat1]
        return out_seg1, out_seg2, [stage_feat2, stage_feat1], deep_out1
