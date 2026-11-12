"""
Semi-Supervised Consistency Regularization Losses for Volumetric Segmentation.
Provides class-balanced channel MSE, continuous foreground Soft Dice, Hybrid, and KL Divergence.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import LOSSES
from losses.dice import reduce_batch_stats
from data.constants import (
    DEFAULT_CONSISTENCY_DICE_WEIGHT,
    DEFAULT_CONSISTENCY_MSE_WEIGHT,
    DEFAULT_FG_CONFIDENCE_MAX_THRESH,
)

DEFAULT_SMOOTH = 1e-5
DEFAULT_EPS = 1e-6


def compute_soft_dice_consistency(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    num_classes: int = 6,
    is_binary: bool = False,
    conf_mask: Optional[torch.Tensor] = None,
    smooth: float = DEFAULT_SMOOTH,
    batch_dice: bool = False,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    """
    Continuous Probability-to-Probability Soft Dice Consistency on Foregrounds.
    Excludes background (class 0) whenever num_classes > 1 to eliminate background confirmation bias.
    """
    fg_s = student_prob[:, 1:] if num_classes > 1 else student_prob
    fg_t = teacher_prob[:, 1:] if num_classes > 1 else teacher_prob

    if conf_mask is not None:
        fg_s = fg_s * conf_mask
        fg_t = fg_t * conf_mask

    # Spatial sum over 3D volume (D, H, W)
    dims = (2, 3, 4)
    intersection = torch.sum(fg_s * fg_t, dim=dims)
    cardinality = torch.sum(fg_s ** 2 + fg_t ** 2, dim=dims)

    if batch_dice:
        intersection, cardinality = reduce_batch_stats(intersection, cardinality, ddp=ddp)

    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(1e-8)
    return 1.0 - torch.mean(dice_score)


@LOSSES.register("consistency")
class ConsistencyLoss(nn.Module):
    """
    Consistency loss engine for Teacher-Student and Perturbation-based SSL methods.
    Supports 'hybrid', 'dice', 'weighted_mse', and 'kl'.
    """
    def __init__(
        self,
        consistency_type: str = 'hybrid',
        num_classes: int = 6,
        is_binary: bool = False,
        confidence_thresh: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        fg_confidence_max_thresh: float = DEFAULT_FG_CONFIDENCE_MAX_THRESH,
        weight_dice: float = DEFAULT_CONSISTENCY_DICE_WEIGHT,
        weight_mse: float = DEFAULT_CONSISTENCY_MSE_WEIGHT,
        eps: float = DEFAULT_EPS,
        smooth: float = DEFAULT_SMOOTH,
        batch_dice: bool = False,
        ddp: Optional[bool] = None,
    ):
        super(ConsistencyLoss, self).__init__()
        self.consistency_type = str(consistency_type).lower()
        self.num_classes = num_classes
        self.is_binary = is_binary
        self.confidence_thresh = confidence_thresh
        self.class_weights = class_weights
        self.fg_confidence_max_thresh = float(fg_confidence_max_thresh)
        self.weight_dice = float(weight_dice)
        self.weight_mse = float(weight_mse)
        self.eps = float(eps)
        self.smooth = float(smooth)
        self.batch_dice = bool(batch_dice)
        self.ddp = ddp

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        student_prob = F.softmax(student_logits.float(), dim=1)
        teacher_prob = F.softmax(teacher_logits.float(), dim=1)

        # Confidence thresholding with CT foreground preservation
        if self.confidence_thresh > 0.0:
            if self.num_classes > 1 and not self.is_binary:
                fg_max_prob, _ = torch.max(teacher_prob[:, 1:], dim=1, keepdim=True)
                bg_prob = teacher_prob[:, :1]
                fg_thresh = min(self.confidence_thresh, self.fg_confidence_max_thresh)
                conf_mask = ((fg_max_prob >= fg_thresh) | (bg_prob >= self.confidence_thresh)).float()
            else:
                max_prob, _ = torch.max(teacher_prob, dim=1, keepdim=True)
                conf_mask = (max_prob >= self.confidence_thresh).float()
        else:
            conf_mask = None

        # 1. Pure Continuous Soft Dice on Foreground Classes
        if self.consistency_type == 'dice':
            return compute_soft_dice_consistency(
                student_prob, teacher_prob, self.num_classes, self.is_binary,
                conf_mask=conf_mask, smooth=self.smooth, batch_dice=self.batch_dice, ddp=self.ddp,
            )

        # 2. Hybrid Consistency: Soft Dice + Class-balanced Channel MSE
        elif self.consistency_type == 'hybrid':
            loss_dice = compute_soft_dice_consistency(
                student_prob, teacher_prob, self.num_classes, self.is_binary,
                conf_mask=conf_mask, smooth=self.smooth, batch_dice=self.batch_dice, ddp=self.ddp,
            )
            sq_diff = (student_prob - teacher_prob) ** 2
            if conf_mask is not None:
                sq_diff = sq_diff * conf_mask
                mask_sum = conf_mask.sum(dim=(0, 2, 3, 4)) + self.eps
                mse_per_class = sq_diff.sum(dim=(0, 2, 3, 4)) / mask_sum
            else:
                mse_per_class = sq_diff.mean(dim=(0, 2, 3, 4))

            if self.class_weights is not None:
                w = self.class_weights.view(-1)
                loss_mse = (mse_per_class * w).sum() / (w.sum() + self.eps)
            else:
                loss_mse = mse_per_class.mean()
            return self.weight_mse * loss_mse + self.weight_dice * loss_dice

        # 3. Class-balanced Channel MSE
        elif self.consistency_type in ('weighted_mse', 'mse', 'class_balanced_mse'):
            sq_diff = (student_prob - teacher_prob) ** 2
            if conf_mask is not None:
                sq_diff = sq_diff * conf_mask
                mask_sum = conf_mask.sum(dim=(0, 2, 3, 4)) + self.eps
                mse_per_class = sq_diff.sum(dim=(0, 2, 3, 4)) / mask_sum
            else:
                mse_per_class = sq_diff.mean(dim=(0, 2, 3, 4))

            if self.class_weights is not None:
                w = self.class_weights.view(-1)
                return (mse_per_class * w).sum() / (w.sum() + self.eps)
            return mse_per_class.mean()

        # 4. KL divergence: Teacher -> Student
        elif self.consistency_type == 'kl':
            if conf_mask is not None:
                kl = F.kl_div(F.log_softmax(student_logits.float(), dim=1), teacher_prob, reduction='none')
                return (kl * conf_mask).sum() / (conf_mask.sum() + self.eps)
            return F.kl_div(F.log_softmax(student_logits.float(), dim=1), teacher_prob, reduction='batchmean')

        # Fallback raw MSE
        else:
            sq_diff = (student_prob - teacher_prob) ** 2
            if conf_mask is not None:
                return (sq_diff * conf_mask).sum() / (conf_mask.sum() + self.eps)
            return sq_diff.mean()
