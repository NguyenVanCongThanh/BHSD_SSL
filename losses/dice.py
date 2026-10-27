"""
Dice Loss variants for Medical Image Segmentation.
Optimized for 3D volumetric data (BHSD dataset) supporting binary and multi-class tasks,
with optional voxel-wise masking for semi-supervised / unlabelled scenarios.
"""

from typing import Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import LOSSES


def expand_spatial_mask(mask, target):
    """Broadcast spatial/shared masks to every target voxel before reduction."""
    if mask.dim() == target.dim() + 1 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    try:
        return torch.broadcast_to(mask, target.shape).to(device=target.device, dtype=torch.float32)
    except RuntimeError as exc:
        raise ValueError(f'Mask {tuple(mask.shape)} cannot cover targets {tuple(target.shape)}') from exc


def to_one_hot(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert target integer tensor [N, 1, ...] to one-hot format [N, num_classes, ...].
    """
    assert tensor.max().item() < num_classes, (
        f"Target label max ({tensor.max().item()}) >= num_classes ({num_classes})"
    )
    assert tensor.min().item() >= 0, f"Target label min ({tensor.min().item()}) < 0"

    shape = list(tensor.shape)
    assert shape[1] == 1, f"Expected channel dimension 1, got shape {shape}"
    shape[1] = num_classes

    one_hot = torch.zeros(shape, dtype=torch.float32, device=tensor.device)
    return one_hot.scatter_(1, tensor.long(), 1.0)


def get_probability(logits: torch.Tensor):
    """
    Convert logits to probability distribution.
    If C == 1, converts via sigmoid into 2-class probability [1 - p, p].
    If C > 1, applies softmax across channel dimension.
    """
    logits = logits.float()
    num_classes = logits.shape[1]
    if num_classes > 1:
        pred = F.softmax(logits, dim=1)
    else:
        p = torch.sigmoid(logits)
        pred = torch.cat([1.0 - p, p], dim=1)
        num_classes = 2
    return pred, num_classes


@LOSSES.register("mask_dice")
class MaskDiceLoss(nn.Module):
    """
    Dice loss with support for spatial/voxel masking (used extensively in SSL).
    Computes soft Dice coefficient across specified classes, with option to include or exclude background.
    """
    def __init__(
        self,
        nclass: int,
        class_weights: Optional[Union[Sequence[float], np.ndarray, torch.Tensor]] = None,
        smooth: float = 1e-5,
        include_background: bool = False,
    ):
        super().__init__()
        self.nclass = nclass
        self.smooth = float(smooth)
        self.include_background = include_background

        if class_weights is None:
            weights = torch.ones((1, nclass), dtype=torch.float32)
        else:
            w_arr = (class_weights.detach().cpu().numpy() if isinstance(class_weights, torch.Tensor)
                     else np.asarray(class_weights, dtype=np.float32))
            assert w_arr.shape[0] == nclass, (
                f"class_weights length {w_arr.shape[0]} does not match nclass {nclass}"
            )
            weights = torch.from_numpy(w_arr).unsqueeze(0)

        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Class weights must be finite, nonnegative, and have a positive sum')
        self.register_buffer("class_weights", weights)

    def _reduce_dice(self, dice):
        weights = self.class_weights.to(device=dice.device, dtype=dice.dtype)
        if weights.shape[1] != dice.shape[1]:
            raise ValueError('Dice loss class count does not match predictions')
        if not self.include_background and dice.shape[1] > 1:
            dice, weights = dice[:, 1:], weights[:, 1:]
        if weights.sum() <= 0:
            raise ValueError('Selected foreground classes must have positive total weight')
        return 1.0 - (dice * weights).sum() / (weights.sum() * dice.shape[0])

    def prob_forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss from probability maps [N, C, ...].
        """
        pred = pred.float()
        N = pred.shape[0]
        nclass = pred.shape[1]

        # Flatten spatial dimensions: [N, C, V]
        pred_flat = pred.view(N, nclass, -1)
        target_flat = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target_flat, nclass)

        inter = pred_flat * target_one_hot
        union = pred_flat + target_one_hot

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)
            if mask.shape[0] != N:
                mask = mask.expand(N, -1, -1, -1)
            mask_flat = mask.reshape(N, 1, -1)
            inter = (inter * mask_flat).sum(dim=2)
            union = (union * mask_flat).sum(dim=2)
        else:
            inter = inter.sum(dim=2)
            union = union.sum(dim=2)

        dice = (2.0 * inter + self.smooth) / (union + self.smooth)

        return self._reduce_dice(dice)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss from raw model logits [N, C, ...].
        """
        N = logits.shape[0]
        nclass = logits.shape[1]

        logits_flat = logits.view(N, nclass, -1)
        target_flat = target.view(N, 1, -1)

        pred_flat, nclass = get_probability(logits_flat)
        target_one_hot = to_one_hot(target_flat, nclass)

        inter = pred_flat * target_one_hot
        union = pred_flat + target_one_hot

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)
            if mask.shape[0] != N:
                mask = mask.expand(N, -1, -1, -1)
            mask_flat = mask.reshape(N, 1, -1)
            inter = (inter * mask_flat).sum(dim=2)
            union = (union * mask_flat).sum(dim=2)
        else:
            inter = inter.sum(dim=2)
            union = union.sum(dim=2)

        dice = (2.0 * inter + self.smooth) / (union + self.smooth)

        return self._reduce_dice(dice)


# Backward-compatible alias for existing imports
mask_DiceLoss = MaskDiceLoss
