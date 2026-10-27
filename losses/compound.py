"""
Compound and multi-task loss utilities for medical image segmentation.
"""

from typing import Optional, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import LOSSES
from losses.dice import MaskDiceLoss, expand_spatial_mask


@LOSSES.register("dice_ce")
class DiceCELoss(nn.Module):
    """
    Compound Dice + Cross-Entropy Loss standard in modern 3D medical segmentation (nnU-Net style).
    Loss = w_dice * DiceLoss + w_ce * CELoss.
    """
    def __init__(
        self,
        nclass: int,
        weight_dice: float = 1.0,
        weight_ce: float = 1.0,
        smooth: float = 1e-5,
        include_background: bool = False,
    ):
        super().__init__()
        self.nclass = nclass
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = MaskDiceLoss(
            nclass=nclass,
            smooth=smooth,
            include_background=include_background
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dice_l = self.dice_loss(logits, target, mask=mask)

        # Cross Entropy
        target_squeezed = target.squeeze(1).long() if target.dim() > logits.dim() - 1 else target.long()
        if mask is not None:
            ce_raw = F.cross_entropy(logits, target_squeezed, reduction='none')
            mask_expanded = expand_spatial_mask(mask, target_squeezed)
            ce_l = (ce_raw * mask_expanded).sum() / (mask_expanded.sum() + 1e-7)
        else:
            ce_l = F.cross_entropy(logits, target_squeezed)

        return self.weight_dice * dice_l + self.weight_ce * ce_l
