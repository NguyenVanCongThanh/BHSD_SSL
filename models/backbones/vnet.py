"""
V-Net 3D Volumetric Segmentation Architecture and Modular Encoder/Decoder Blocks.
Reference: Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", 3DV 2016.
"""

from typing import List, Tuple, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F

from core.registry import MODELS


class ConvBlock(nn.Module):
    def __init__(
        self,
        n_stages: int,
        n_filters_in: int,
        n_filters_out: int,
        kernel_size: int = 3,
        padding: int = 1,
        normalization: str = 'none'
    ):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, kernel_size=kernel_size, padding=padding))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=min(16, n_filters_out), num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError(f"Unknown normalization: {normalization}")
            ops.append(nn.ReLU(inplace=False))

        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        n_stages: int,
        n_filters_in: int,
        n_filters_out: int,
        normalization: str = 'none'
    ):
        super(ResidualConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=min(16, n_filters_out), num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError(f"Unknown normalization: {normalization}")
            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=False))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + x)


class DownsamplingConvBlock(nn.Module):
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        stride: int = 2,
        padding: int = 0,
        normalization: str = 'none'
    ):
        super(DownsamplingConvBlock, self).__init__()
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride, padding=padding))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=min(16, n_filters_out), num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unknown normalization: {normalization}")
        ops.append(nn.ReLU(inplace=False))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        stride: int = 2,
        padding: int = 0,
        normalization: str = 'none'
    ):
        super(UpsamplingDeconvBlock, self).__init__()
        ops = []
        ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride, padding=padding))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=min(16, n_filters_out), num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unknown normalization: {normalization}")
        ops.append(nn.ReLU(inplace=False))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsampling(nn.Module):
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        stride: int = 2,
        normalization: str = 'none'
    ):
        super(Upsampling, self).__init__()
        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=min(16, n_filters_out), num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unknown normalization: {normalization}")
        ops.append(nn.ReLU(inplace=False))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VNetEncoder(nn.Module):
    """V-Net Feature Extractor (Encoder)."""
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: int = 16,
        normalization: str = 'batchnorm',
        has_dropout: bool = False,
        has_residual: bool = False
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
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

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
    """V-Net Feature Reconstructor (Decoder)."""
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: int = 16,
        normalization: str = 'batchnorm',
        has_dropout: bool = False,
        has_residual: bool = False
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
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

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


# For exact checkpoint compatibility, retain Encoder and Decoder class names
Encoder = VNetEncoder
Decoder = VNetDecoder


class VNetResidualBlock(nn.Module):
    """
    Two-convolution residual block with InstanceNorm3d and LeakyReLU/ReLU for V-Net.
    Implements intra-block residual addition: y = Act(Conv2(Norm(Conv1(x))) + Shortcut(x)).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        activation: str = 'leaky_relu'
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)

        def _make_act(act_name: str):
            if act_name == 'relu':
                return nn.ReLU(inplace=True)
            elif act_name == 'prelu':
                return nn.PReLU()
            elif act_name == 'leaky_relu':
                return nn.LeakyReLU(negative_slope=0.01, inplace=True)
            else:
                raise ValueError(f"Unsupported activation: {act_name}")

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = _make_act(activation)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = _make_act(activation)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act1(self.norm1(self.conv1(x)))
        res = self.norm2(self.conv2(res))
        res = res + self.shortcut(x)
        return self.act2(res)


@MODELS.register("vnet_7stage")
@MODELS.register("resvnet3d")
class VNet7Stage(nn.Module):
    """
    7-Stage Anisotropic Residual V-Net for Volumetric CT Segmentation.
    Designed for controlled 1:1 ablation comparison against 7-Stage nnU-Net 3D and SWDL variants:
    - 7 resolution stages with matching stride schedule.
    - Anisotropic kernels (1, 3, 3) at early stages, (3, 3, 3) at deeper stages.
    - Long skip connections: Concatenation ('concat') or Element-wise Addition ('add').
    - Intra-block connections: Residual addition with 1x1 projection shortcuts.
    - Multi-scale deep supervision (1 main + 5 auxiliary heads).
    """
    CAPABILITIES = {"segmentation", "deep_supervision"}

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_filters: int = 32,
        num_stages: int = 7,
        has_dropout: bool = False,
        deep_supervision: bool = True,
        anisotropic_kernels: bool = True,
        use_strided_conv: bool = True,
        skip_mode: str = 'concat',
        activation: str = 'leaky_relu',
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.deep_supervision = deep_supervision
        self.anisotropic_kernels = anisotropic_kernels
        self.use_strided_conv = use_strided_conv
        self.skip_mode = str(skip_mode).lower()
        self.activation = str(activation).lower()

        if self.skip_mode not in ('concat', 'add'):
            raise ValueError(f"skip_mode must be 'concat' or 'add', got {self.skip_mode}")

        f = [
            base_filters,
            base_filters * 2,
            base_filters * 4,
            base_filters * 8,
            base_filters * 10,
            base_filters * 10,
            base_filters * 10
        ]  # [32, 64, 128, 256, 320, 320, 320]

        k1 = (1, 3, 3) if anisotropic_kernels else (3, 3, 3)
        k2 = (1, 3, 3) if anisotropic_kernels else (3, 3, 3)
        k3 = (3, 3, 3)
        k4 = (3, 3, 3)
        k5 = (3, 3, 3)
        k6 = (3, 3, 3)
        k_btn = (3, 3, 3)

        def _make_downsample(channels: int, stride: Tuple[int, int, int]):
            if use_strided_conv:
                return nn.Conv3d(channels, channels, kernel_size=stride, stride=stride, padding=0, bias=False)
            else:
                return nn.MaxPool3d(kernel_size=stride, stride=stride)

        # Encoder stages & Strides
        self.enc1 = VNetResidualBlock(in_channels, f[0], kernel_size=k1, activation=self.activation)
        self.pool1 = _make_downsample(f[0], stride=(1, 2, 2) if anisotropic_kernels else (2, 2, 2))

        self.enc2 = VNetResidualBlock(f[0], f[1], kernel_size=k2, activation=self.activation)
        self.pool2 = _make_downsample(f[1], stride=(1, 2, 2) if anisotropic_kernels else (2, 2, 2))

        self.enc3 = VNetResidualBlock(f[1], f[2], kernel_size=k3, activation=self.activation)
        self.pool3 = _make_downsample(f[2], stride=(2, 2, 2))

        self.enc4 = VNetResidualBlock(f[2], f[3], kernel_size=k4, activation=self.activation)
        self.pool4 = _make_downsample(f[3], stride=(2, 2, 2))

        self.enc5 = VNetResidualBlock(f[3], f[4], kernel_size=k5, activation=self.activation)
        self.pool5 = _make_downsample(f[4], stride=(1, 2, 2))

        self.enc6 = VNetResidualBlock(f[4], f[5], kernel_size=k6, activation=self.activation)
        self.pool6 = _make_downsample(f[5], stride=(1, 2, 2))

        self.bottleneck = VNetResidualBlock(f[5], f[6], kernel_size=k_btn, activation=self.activation)

        # Decoder stages (skip_mode = 'concat' -> in_channels = f * 2; skip_mode = 'add' -> in_channels = f)
        dec_in6 = f[5] * 2 if self.skip_mode == 'concat' else f[5]
        self.up6 = nn.ConvTranspose3d(f[6], f[5], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec6 = VNetResidualBlock(dec_in6, f[5], kernel_size=k6, activation=self.activation)

        dec_in5 = f[4] * 2 if self.skip_mode == 'concat' else f[4]
        self.up5 = nn.ConvTranspose3d(f[5], f[4], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec5 = VNetResidualBlock(dec_in5, f[4], kernel_size=k5, activation=self.activation)

        self.dropout = nn.Dropout3d(p=0.2) if has_dropout else nn.Identity()

        dec4_in = f[4]
        dec_in4 = f[3] * 2 if self.skip_mode == 'concat' else f[3]
        self.up4 = nn.ConvTranspose3d(dec4_in, f[3], kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec4 = VNetResidualBlock(dec_in4, f[3], kernel_size=k4, activation=self.activation)

        dec_in3 = f[2] * 2 if self.skip_mode == 'concat' else f[2]
        self.up3 = nn.ConvTranspose3d(f[3], f[2], kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec3 = VNetResidualBlock(dec_in3, f[2], kernel_size=k3, activation=self.activation)

        stride2 = (1, 2, 2) if anisotropic_kernels else (2, 2, 2)
        dec_in2 = f[1] * 2 if self.skip_mode == 'concat' else f[1]
        self.up2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=stride2, stride=stride2)
        self.dec2 = VNetResidualBlock(dec_in2, f[1], kernel_size=k2, activation=self.activation)

        stride1 = (1, 2, 2) if anisotropic_kernels else (2, 2, 2)
        dec_in1 = f[0] * 2 if self.skip_mode == 'concat' else f[0]
        self.up1 = nn.ConvTranspose3d(f[1], f[0], kernel_size=stride1, stride=stride1)
        self.dec1 = VNetResidualBlock(dec_in1, f[0], kernel_size=k1, activation=self.activation)

        # Final segmentation head (1x1x1 Conv)
        self.final_conv = nn.Conv3d(f[0], num_classes, kernel_size=1)

        # Auxiliary heads for deep supervision
        if self.deep_supervision:
            self.ds_conv2 = nn.Conv3d(f[1], num_classes, kernel_size=1)
            self.ds_conv3 = nn.Conv3d(f[2], num_classes, kernel_size=1)
            self.ds_conv4 = nn.Conv3d(f[3], num_classes, kernel_size=1)
            self.ds_conv5 = nn.Conv3d(f[4], num_classes, kernel_size=1)
            self.ds_conv6 = nn.Conv3d(f[5], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        nonlinearity = 'relu' if self.activation == 'relu' else 'leaky_relu'
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _combine_skip(self, up_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        if self.skip_mode == 'add':
            return up_feat + skip_feat
        return torch.cat([up_feat, skip_feat], dim=1)

    def forward(self, x: torch.Tensor, return_all_scales: bool = False):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))
        x6 = self.enc6(self.pool5(x5))
        b = self.bottleneck(self.pool6(x6))
        b = self.dropout(b)

        d6 = self.up6(b)
        d6 = self._combine_skip(d6, x6)
        d6 = self.dec6(d6)

        d5 = self.up5(d6)
        d5 = self._combine_skip(d5, x5)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = self._combine_skip(d4, x4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._combine_skip(d3, x3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._combine_skip(d2, x2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._combine_skip(d1, x1)
        d1 = self.dec1(d1)

        logits = self.final_conv(d1)

        if self.deep_supervision and (self.training or return_all_scales):
            aux = [
                self.ds_conv2(d2),  # Scale 0.5
                self.ds_conv3(d3),  # Scale 0.25
                self.ds_conv4(d4),  # Scale 0.125
                self.ds_conv5(d5),  # Scale 0.0625
                self.ds_conv6(d6),  # Scale 0.03125
            ]
            return {
                "seg": logits,
                "aux": aux
            }
        return logits


@MODELS.register("vnet")
@MODELS.register("vnet_single")
class VNet(nn.Module):
    """
    Unified V-Net Architecture:
    - num_stages == 7: 7-Stage Anisotropic Residual V-Net (aligned with nnU-Net topology, 2 convs per block,
      skip concatenation/addition, multi-scale auxiliary heads).
    - num_stages == 5: Classic 5-Stage V-Net (Milletari et al. 2016).
    """
    CAPABILITIES = {"segmentation", "deep_supervision"}

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        n_filters: int = 16,
        base_filters: Optional[int] = None,
        num_stages: int = 5,
        normalization: str = 'batchnorm',
        has_dropout: bool = True,
        has_residual: bool = False,
        deep_supervision: bool = False,
        anisotropic_kernels: bool = True,
        use_strided_conv: bool = True,
        skip_mode: str = 'concat',
        activation: str = 'leaky_relu',
        **kwargs
    ):
        super(VNet, self).__init__()
        self.num_stages = num_stages
        if num_stages == 7 or base_filters is not None:
            bf = base_filters if base_filters is not None else max(32, n_filters * 2)
            self.model = VNet7Stage(
                in_channels=in_channels,
                num_classes=num_classes,
                base_filters=bf,
                num_stages=7,
                has_dropout=has_dropout,
                deep_supervision=deep_supervision,
                anisotropic_kernels=anisotropic_kernels,
                use_strided_conv=use_strided_conv,
                skip_mode=skip_mode,
                activation=activation,
                **kwargs
            )
            self.is_7stage = True
        else:
            self.encoder = VNetEncoder(
                n_channels=in_channels,
                n_classes=num_classes,
                n_filters=n_filters,
                normalization=normalization,
                has_dropout=has_dropout,
                has_residual=has_residual
            )
            self.decoder = VNetDecoder(
                n_channels=in_channels,
                n_classes=num_classes,
                n_filters=n_filters,
                normalization=normalization,
                has_dropout=has_dropout,
                has_residual=has_residual
            )
            self.is_7stage = False

    def forward(self, x: torch.Tensor, **kwargs):
        if self.is_7stage:
            return self.model(x, **kwargs)
        features = self.encoder(x)
        out_seg, feat_de = self.decoder(features)
        return out_seg
