"""
LeFeD Architecture (Learning Feedback Dual-branch Net) for 3D Semi-Supervised Medical Image Segmentation.
Reference: "Learning Feedback for Semi-Supervised 3D Medical Image Segmentation".
Adapted for BHSD Multi-class 3D Brain Hemorrhage Segmentation.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODELS


class ConvBlock(nn.Module):
    def __init__(self, n_stages: int, n_filters_in: int, n_filters_out: int, normalization: str = 'groupnorm'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            input_channel = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=min(16, n_filters_out), num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError(f"Unsupported normalization: {normalization}")
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages: int, n_filters_in: int, n_filters_out: int, normalization: str = 'groupnorm'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            input_channel = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=min(16, n_filters_out), num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError(f"Unsupported normalization: {normalization}")

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in: int, n_filters_out: int, stride: int = 2, normalization: str = 'groupnorm'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=min(16, n_filters_out), num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unsupported normalization: {normalization}")
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsamplingFunction(nn.Module):
    def __init__(self, n_filters_in: int, n_filters_out: int, stride: int = 2, normalization: str = 'groupnorm', mode_upsampling: int = 1):
        super(UpsamplingFunction, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        elif mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        else:
            raise ValueError(f"Unsupported mode_upsampling: {mode_upsampling}")

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=min(16, n_filters_out), num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unsupported normalization: {normalization}")
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LeFeDEncoder(nn.Module):
    """Encoder capable of receiving multi-level feedback signals (en)."""
    def __init__(self, in_channels: int = 1, n_filters: int = 16, normalization: str = 'groupnorm', has_dropout: bool = False, has_residual: bool = False):
        super(LeFeDEncoder, self).__init__()
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


class LeFeDDecoder(nn.Module):
    """Decoder supporting feature-guided gating from complementary decoder."""
    def __init__(self, num_classes: int = 6, n_filters: int = 16, normalization: str = 'groupnorm', has_dropout: bool = False, has_residual: bool = False, up_type: int = 0):
        super(LeFeDDecoder, self).__init__()
        self.has_dropout = has_dropout
        block_cls = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = UpsamplingFunction(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)
        self.block_six = block_cls(3, n_filters * 8, n_filters * 8, normalization=normalization)

        self.block_six_up = UpsamplingFunction(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)
        self.block_seven = block_cls(3, n_filters * 4, n_filters * 4, normalization=normalization)

        self.block_seven_up = UpsamplingFunction(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)
        self.block_eight = block_cls(2, n_filters * 2, n_filters * 2, normalization=normalization)

        self.block_eight_up = UpsamplingFunction(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)
        self.block_nine = block_cls(1, n_filters, n_filters, normalization=normalization)

        self.out_conv = nn.Conv3d(n_filters, num_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features: List[torch.Tensor], f1: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x1, x2, x3, x4, x5 = features

        if f1 is None:
            x5_up_ori = self.block_five_up(x5)
            x5_up = x5_up_ori + x4

            x6 = self.block_six(x5_up)
            x6_up_ori = self.block_six_up(x6)
            x6_up = x6_up_ori + x3

            x7 = self.block_seven(x6_up)
            x7_up_ori = self.block_seven_up(x7)
            x7_up = x7_up_ori + x2

            x8 = self.block_eight(x7_up)
            x8_up_ori = self.block_eight_up(x8)
            x8_up = x8_up_ori + x1

            x9 = self.block_nine(x8_up)
            if self.has_dropout:
                x9 = self.dropout(x9)
            out_seg = self.out_conv(x9)
        else:
            m5, m4, m3, m2, m1 = f1
            w5, w4, w3, w2, w1 = [torch.sigmoid(m).detach() for m in [m5, m4, m3, m2, m1]]

            x5 = x5 + x5 * w5
            x5_up_ori = self.block_five_up(x5)
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


class SideConv(nn.Module):
    """Deep Supervision multi-scale projection head."""
    def __init__(self, num_classes: int = 6, n_filters: int = 16):
        super(SideConv, self).__init__()
        self.side5 = nn.Conv3d(n_filters * 16, num_classes, 1, padding=0)
        self.side4 = nn.Conv3d(n_filters * 8, num_classes, 1, padding=0)
        self.side3 = nn.Conv3d(n_filters * 4, num_classes, 1, padding=0)
        self.side2 = nn.Conv3d(n_filters * 2, num_classes, 1, padding=0)
        self.side1 = nn.Conv3d(n_filters, num_classes, 1, padding=0)

    def forward(self, stage_feat: List[torch.Tensor]) -> List[torch.Tensor]:
        x5, x5_up, x6_up, x7_up, x8_up = stage_feat
        target_size = x8_up.shape[2:]

        out5 = F.interpolate(self.side5(x5), size=target_size, mode='trilinear', align_corners=True)
        out4 = F.interpolate(self.side4(x5_up), size=target_size, mode='trilinear', align_corners=True)
        out3 = F.interpolate(self.side3(x6_up), size=target_size, mode='trilinear', align_corners=True)
        out2 = F.interpolate(self.side2(x7_up), size=target_size, mode='trilinear', align_corners=True)
        out1 = self.side1(x8_up)

        return [out5, out4, out3, out2, out1]


@MODELS.register("lefed_net")
class LeFeD_Net(nn.Module):
    """
    Complete LeFeD Network:
    - Shared Feedback Encoder.
    - Decoder 1 (mode 0 deconvolution) + Decoder 2 (mode 1 trilinear upsample + conv).
    - SideConv for Deep Supervision.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        n_filters: int = 16,
        normalization: str = 'groupnorm',
        has_dropout: bool = True,
        has_residual: bool = False
    ):
        super(LeFeD_Net, self).__init__()
        self.num_classes = num_classes

        self.encoder = LeFeDEncoder(
            in_channels=in_channels,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual
        )
        self.decoder1 = LeFeDDecoder(
            num_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
            up_type=0
        )
        self.decoder2 = LeFeDDecoder(
            num_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
            up_type=1
        )
        self.sideconv = SideConv(num_classes=num_classes, n_filters=n_filters)

    def forward(
        self,
        x: torch.Tensor,
        en: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]], List[torch.Tensor]]:
        """
        Forward pass:
        - If during inference (or standard sliding window), returns out_seg1.
        - If training with feedback loop, returns full tuple (out1, out2, masks, deep_outs).
        """
        features = self.encoder(x, en)
        out_seg1, stage_feat1 = self.decoder1(features)

        # In standard eval/inference mode, return primary prediction
        if not self.training and en is None:
            return out_seg1

        out_seg2, stage_feat2 = self.decoder2(features, stage_feat1)
        deep_out1 = self.sideconv(stage_feat1)

        # masks: [stage_feat2, stage_feat1]
        return out_seg1, out_seg2, [stage_feat2, stage_feat1], deep_out1
