"""
PICK Composite Model: 1 Shared Encoder Backbone + 3 Multi-Task Decoders (Main, MIM, Aux).
Seamless 100% parameter naming compatibility with historical VNet_my checkpoints.
"""

from typing import Tuple, List, Union, Optional
import torch
from torch import nn

from core.registry import MODELS
from models.backbones.vnet import VNetEncoder, VNetDecoder


@MODELS.register("pick_vnet")
@MODELS.register("pick_model")
@MODELS.register("vnet_my")
class PICKModel(nn.Module):
    """
    PICK Multi-head Model architecture:
    - Shared Encoder trích xuất đặc trưng đa tầng.
    - Main Decoder: Sinh nhãn phân đoạn chính và dự đoán giả (pseudo-labels).
    - MIM Decoder: Tái tạo ảnh 3D bị che (Masked Image Modeling Reconstruction).
    - Aux Decoder: Nhánh phân đoạn phụ trợ trên ảnh tái tạo đã lật/tăng cường.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        n_filters: int = 16,
        normalization: str = 'batchnorm',
        has_dropout: bool = True,
        has_residual: bool = False
    ):
        super(PICKModel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Shared feature encoder
        self.encoder = VNetEncoder(
            n_channels=in_channels,
            n_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual
        )

        # 1. Main Segmentation Decoder
        self.main_decoder = VNetDecoder(
            n_channels=in_channels,
            n_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual
        )

        # 2. Masked Image Modeling (MIM) Reconstruction Decoder (reconstructs input image, so classes = in_channels)
        self.mim_decoder = VNetDecoder(
            n_channels=in_channels,
            n_classes=in_channels,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual
        )

        # 3. Auxiliary Segmentation Decoder
        self.aux_decoder = VNetDecoder(
            n_channels=in_channels,
            n_classes=num_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual
        )

    def mim_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shorthand to forward input through MIM branch."""
        return self.forward(x, mode='mim')

    def aux_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shorthand to forward input through Auxiliary branch."""
        return self.forward(x, mode='aux')

    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'main'
    ) -> Union[Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
        features = self.encoder(x)

        if mode == 'mim':
            out_mim, _ = self.mim_decoder(features)
            return out_mim
        elif mode == 'aux':
            out_seg, _ = self.aux_decoder(features)
            return out_seg
        else:
            # Main forward: returns (features, main_outputs, feat_de)
            out_seg, feat_de = self.main_decoder(features)
            return features, out_seg, feat_de


# Historical alias for exact backward compatibility with legacy scripts
VNet_my = PICKModel
