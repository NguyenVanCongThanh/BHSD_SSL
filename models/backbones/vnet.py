"""
Unified V-Net Facade and Model Router for 3D Medical Volumetric Segmentation.
Routes dynamically to:
  1. VNet7Stage: 7-Stage Anisotropic Residual V-Net (Option 1 design, Deep Supervision)
  2. VNetClassic: Milletari 5-Stage V-Net (VNetEncoder + VNetDecoder, PICK baseline)
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import nn

from core.registry import MODELS
from models.blocks import (
    ConvBlock,
    ResidualConvBlock,
    DownsamplingConvBlock,
    UpsamplingDeconvBlock,
    Upsampling,
    VNetResidualBlock,
)
from models.backbones.vnet_classic import (
    VNetEncoder,
    VNetDecoder,
    Encoder,
    Decoder,
    LEGACY_DROPOUT_PROB,
)
from models.backbones.vnet_7stage import (
    VNet7Stage,
    DEFAULT_DROPOUT_PROB,
    DEFAULT_STAGE_MULTIPLIERS,
    DEFAULT_ANISOTROPIC_KERNELS,
    DEFAULT_ISOTROPIC_KERNELS,
    DEFAULT_ANISOTROPIC_POOL_STRIDES,
    DEFAULT_ISOTROPIC_POOL_STRIDES,
)


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
        normalization: str = "batchnorm",
        has_dropout: bool = True,
        has_residual: bool = False,
        deep_supervision: bool = False,
        anisotropic_kernels: bool = True,
        use_strided_conv: bool = True,
        skip_mode: str = "concat",
        activation: str = "leaky_relu",
        dropout_prob: Optional[float] = None,
        stage_multipliers: Optional[Sequence[int]] = None,
        pool_strides: Optional[Sequence[Tuple[int, int, int]]] = None,
        stage_kernels: Optional[Sequence[Tuple[int, int, int]]] = None,
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
                dropout_prob=DEFAULT_DROPOUT_PROB if dropout_prob is None else dropout_prob,
                stage_multipliers=stage_multipliers,
                pool_strides=pool_strides,
                stage_kernels=stage_kernels,
                **kwargs
            )
            self.is_7stage = True
        else:
            legacy_drop = LEGACY_DROPOUT_PROB if dropout_prob is None else dropout_prob
            self.encoder = VNetEncoder(
                n_channels=in_channels,
                n_classes=num_classes,
                n_filters=n_filters,
                normalization=normalization,
                has_dropout=has_dropout,
                has_residual=has_residual,
                dropout_prob=legacy_drop,
            )
            self.decoder = VNetDecoder(
                n_channels=in_channels,
                n_classes=num_classes,
                n_filters=n_filters,
                normalization=normalization,
                has_dropout=has_dropout,
                has_residual=has_residual,
                dropout_prob=legacy_drop,
            )
            self.is_7stage = False

    def forward(self, x: torch.Tensor, **kwargs):
        if self.is_7stage:
            return self.model(x, **kwargs)
        features = self.encoder(x)
        out_seg, feat_de = self.decoder(features)
        return out_seg


__all__ = [
    "VNet",
    "VNet7Stage",
    "VNetEncoder",
    "VNetDecoder",
    "Encoder",
    "Decoder",
    "ConvBlock",
    "ResidualConvBlock",
    "DownsamplingConvBlock",
    "UpsamplingDeconvBlock",
    "Upsampling",
    "VNetResidualBlock",
    "DEFAULT_DROPOUT_PROB",
    "LEGACY_DROPOUT_PROB",
    "DEFAULT_STAGE_MULTIPLIERS",
    "DEFAULT_ANISOTROPIC_KERNELS",
    "DEFAULT_ISOTROPIC_KERNELS",
    "DEFAULT_ANISOTROPIC_POOL_STRIDES",
    "DEFAULT_ISOTROPIC_POOL_STRIDES",
]
