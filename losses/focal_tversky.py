"""
Focal Loss, Tversky Loss, and Focal Tversky Loss for 3D Medical Image Segmentation.
Designed for handling extreme class imbalance and asymmetric penalization of False Negatives.
"""

from typing import Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import LOSSES
from losses.dice import expand_spatial_mask


@LOSSES.register("focal_loss")
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for dense segmentation.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, Sequence[float]]] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
        else:
            self.register_buffer("alpha_tensor", torch.tensor(alpha, dtype=torch.float32))
            self.alpha = "tensor"

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape (B, C, D, H, W) or (B, C, H, W).
            target: Ground truth indices of shape (B, D, H, W) or (B, 1, D, H, W).
            mask: Optional binary mask of shape (B, 1, D, H, W) or broadcastable.
        """
        if target.dim() == logits.dim():
            target = target.squeeze(1).long()
        else:
            target = target.long()

        num_classes = logits.shape[1]
        log_p = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(log_p, target, reduction="none", ignore_index=self.ignore_index)
        p = torch.exp(-ce_loss)
        focal_term = (1.0 - p) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            if self.alpha == "tensor":
                at = self.alpha_tensor.gather(0, target.clamp(0, num_classes - 1))
                loss = at * loss
            else:
                # Scalar alpha applies to foreground classes
                alpha_factor = torch.where(target > 0, self.alpha, 1.0 - self.alpha)
                loss = alpha_factor * loss

        if mask is not None:
            expanded_mask = expand_spatial_mask(mask, target)
            loss = (loss * expanded_mask).sum() / (expanded_mask.sum() + 1e-7)
            return loss

        if self.reduction == "mean":
            valid = (target != self.ignore_index)
            return loss.sum() / (valid.sum().float() + 1e-7)
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@LOSSES.register("tversky_loss")
class TverskyLoss(nn.Module):
    """
    Tversky Loss with asymmetric penalties on False Positives (alpha) and False Negatives (beta).
    When alpha = beta = 0.5, this is equivalent to standard Dice Loss.
    When beta > alpha (e.g. beta=0.7, alpha=0.3), it heavily penalizes False Negatives (boosts Recall).
    """
    def __init__(
        self,
        nclass: int,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        include_background: bool = False,
    ):
        super().__init__()
        self.nclass = int(nclass)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.include_background = bool(include_background)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target.dim() == logits.dim():
            target = target.squeeze(1).long()
        else:
            target = target.long()

        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target.clamp(0, self.nclass - 1), num_classes=self.nclass)
        # Permute one-hot from (B, D, H, W, C) -> (B, C, D, H, W)
        perm = (0, -1) + tuple(range(1, target.dim()))
        one_hot = one_hot.permute(*perm).float()

        if mask is not None:
            if mask.dim() < probs.dim():
                mask = mask.unsqueeze(1)
            probs = probs * mask
            one_hot = one_hot * mask

        dims = tuple(range(2, probs.dim()))
        tp = (probs * one_hot).sum(dim=dims)
        fp = (probs * (1.0 - one_hot)).sum(dim=dims)
        fn = ((1.0 - probs) * one_hot).sum(dim=dims)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1.0 - tversky_index

        start_cls = 0 if self.include_background else 1
        valid_loss = loss[:, start_cls:]
        return valid_loss.mean()


@LOSSES.register("focal_tversky")
class FocalTverskyLoss(nn.Module):
    """
    Compound Focal Loss + Tversky Loss for 3D Volumetric Hemorrhage Segmentation.
    Loss = weight_tversky * TverskyLoss + weight_focal * FocalLoss
    """
    def __init__(
        self,
        nclass: int,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 2.0,
        weight_tversky: float = 1.0,
        weight_focal: float = 1.0,
        smooth: float = 1e-5,
        include_background: bool = False,
    ):
        super().__init__()
        self.tversky = TverskyLoss(
            nclass=nclass,
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            include_background=include_background,
        )
        self.focal = FocalLoss(gamma=gamma)
        self.weight_tversky = float(weight_tversky)
        self.weight_focal = float(weight_focal)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l_tversky = self.tversky(logits, target, mask=mask)
        l_focal = self.focal(logits, target, mask=mask)
        return self.weight_tversky * l_tversky + self.weight_focal * l_focal

