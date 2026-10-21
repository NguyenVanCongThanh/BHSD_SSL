"""
Common SSL Utilities and Helper Functions for Semi-Supervised 3D Medical Segmentation.
Provides unified logit extraction, SSL batch partitioning, and masked loss computations.
"""

from typing import Dict, Any, Tuple, Optional, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_logits(model_output: Any) -> torch.Tensor:
    """
    Safely extract primary segmentation logits from diverse model architectures:
    - Standard Net: Tensor of shape (B, C, D, H, W)
    - Multi-Head / PICK Composite: Tuple/List -> typically out[1] for main_outputs
    - Dictionary Output: dict with 'seg', 'logits', or 'out'
    - Deep Supervision: List/Tuple where first element or element 1 contains primary logits
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, (tuple, list)):
        if len(model_output) >= 2 and isinstance(model_output[1], torch.Tensor) and model_output[1].ndim >= 4:
            return model_output[1]
        if len(model_output) > 0 and isinstance(model_output[0], torch.Tensor):
            return model_output[0]
        raise ValueError(f"Unable to extract tensor logits from sequence of types: {[type(x) for x in model_output]}")

    if isinstance(model_output, dict):
        for key in ('seg', 'logits', 'out', 'main'):
            if key in model_output and isinstance(model_output[key], torch.Tensor):
                return model_output[key]
        first_val = next(iter(model_output.values()))
        if isinstance(first_val, torch.Tensor):
            return first_val

    raise TypeError(f"Unsupported model output type for logit extraction: {type(model_output)}")


class SSLBatch(NamedTuple):
    """Structured container for partitioned semi-supervised 3D batches."""
    volume: torch.Tensor
    label: torch.Tensor
    labeled_volume: torch.Tensor
    labeled_label: torch.Tensor
    unlabeled_volume: torch.Tensor
    sub_bs: int
    img_a: torch.Tensor
    img_b: torch.Tensor
    lab_a: torch.Tensor
    lab_b: torch.Tensor
    unimg_a: torch.Tensor
    unimg_b: torch.Tensor


def split_ssl_batch(
    sampled_batch: Dict[str, Any],
    labeled_bs: int,
    device: Optional[torch.device] = None
) -> SSLBatch:
    """
    Split a batch into labeled and unlabeled streams along with sub-batches
    frequently used by CutMix, BCP, and PICK algorithms.
    """
    volume = sampled_batch['image']
    label = sampled_batch['label']
    if device is not None:
        volume = volume.to(device)
        label = label.to(device)

    bs = volume.shape[0]
    actual_labeled_bs = min(labeled_bs, bs)
    sub_bs = max(int(actual_labeled_bs // 2), 1)

    labeled_volume = volume[:actual_labeled_bs]
    labeled_label = label[:actual_labeled_bs]
    unlabeled_volume = volume[actual_labeled_bs:]

    img_a = volume[:sub_bs]
    img_b = volume[sub_bs:actual_labeled_bs]
    lab_a = label[:sub_bs]
    lab_b = label[sub_bs:actual_labeled_bs]

    unimg_a = unlabeled_volume[:sub_bs] if unlabeled_volume.shape[0] >= sub_bs else unlabeled_volume
    unimg_b = unlabeled_volume[sub_bs:] if unlabeled_volume.shape[0] > sub_bs else unlabeled_volume

    return SSLBatch(
        volume=volume,
        label=label,
        labeled_volume=labeled_volume,
        labeled_label=labeled_label,
        unlabeled_volume=unlabeled_volume,
        sub_bs=sub_bs,
        img_a=img_a,
        img_b=img_b,
        lab_a=lab_a,
        lab_b=lab_b,
        unimg_a=unimg_a,
        unimg_b=unimg_b
    )


def compute_masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """Compute Cross Entropy loss restricted only to region specified by mask."""
    mask_f = mask.float()
    target_squeezed = targets.squeeze(1).long() if targets.dim() > logits.dim() - 1 else targets.long()
    ce_unreduced = F.cross_entropy(logits, target_squeezed, reduction='none')
    mask_expanded = mask_f.squeeze(1) if mask_f.dim() > ce_unreduced.dim() else mask_f
    loss = (ce_unreduced * mask_expanded).sum() / (mask_expanded.sum() + eps)
    return loss


def update_ema_variables(
    model: nn.Module,
    ema_model: nn.Module,
    alpha: float,
    global_step: int = -1,
    max_steps: int = 12000,
    schedule: str = 'cosine'
):
    """
    Update EMA teacher parameters from student model parameters.
    Supports dynamic alpha rampup early on or cosine schedule across training.
    """
    import math
    if schedule == 'cosine' and global_step >= 0 and max_steps > 0:
        min_alpha = 0.99
        progress = min(max(global_step / max_steps, 0.0), 1.0)
        actual_alpha = min_alpha + (alpha - min_alpha) * 0.5 * (1.0 - math.cos(math.pi * progress))
    elif global_step >= 0:
        actual_alpha = min(1.0 - 1.0 / (global_step + 1), alpha)
    else:
        actual_alpha = alpha

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(actual_alpha).add_((1.0 - actual_alpha) * param.data)


def get_or_create_mim_mask(
    batch_data: Any,
    bs: int,
    patch_size: Tuple[int, int, int] = (32, 160, 160)
) -> torch.Tensor:
    """Extract or generate MIM mask for volumetric input."""
    if isinstance(batch_data, (tuple, list)) and len(batch_data) > 1 and batch_data[1] is not None:
        return batch_data[1]
    import numpy as np
    from data.masking import MaskGeneratorBHSD
    generator = MaskGeneratorBHSD(input_size=patch_size)
    return torch.from_numpy(np.stack([generator() for _ in range(bs)]))

