"""
FixMatch 3D Method for Semi-Supervised 3D Medical Segmentation.
Reference: Sohn et al., "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence", NeurIPS 2020.

GPU In-Memory Perturbation:
- Weak stream: Native spatial cropped patches from DataLoader.
- Strong stream: Generated dynamically on GPU for unlabeled patches via:
  1. Additive 3D Gaussian noise.
  2. Random 3D contrast / intensity perturbation.
  3. 3D Cutout (random spatial sub-volume zeroing).
- Confident Pseudo-Labeling: Pseudo-labels from Weak predictions with max probability >= threshold
  supervise predictions on the Strong augmented stream.
"""

from typing import Dict, Tuple, Optional
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from data.constants import DEFAULT_SUPERVISED_CE_WEIGHT, DEFAULT_SUPERVISED_DICE_WEIGHT
from methods.base_method import BaseMethod
from methods.ssl_utils import extract_logits, compute_masked_ce_loss
from losses import MaskDiceLoss
from utils.ramps import sigmoid_rampup


@METHODS.register("fixmatch")
class FixMatchMethod(BaseMethod):
    """
    FixMatch 3D Semi-Supervised Segmentation Framework:
    - Zero global data alteration: Applies 3D Strong Augmentation directly in GPU VRAM.
    - Confidence thresholding on Weak predictions to filter noisy pseudo-labels.
    """
    REQUIRED_MODEL_CAPABILITIES = {"segmentation"}

    def __init__(self, model: nn.Module, cfg):
        super(FixMatchMethod, self).__init__(model, cfg)

        self.conf_threshold = float(getattr(self.cfg.method, 'conf_threshold', 0.95))
        self.u_weight = float(getattr(self.cfg.method, 'u_weight', 1.0))
        self.consistency_rampup = float(getattr(self.cfg.method, 'consistency_rampup', 4000.0))

        # Strong augmentation hyperparameters
        self.noise_std_range = getattr(self.cfg.method, 'noise_std_range', [0.02, 0.10])
        self.intensity_scale_range = getattr(self.cfg.method, 'intensity_scale_range', [0.8, 1.2])
        self.cutout_prob = float(getattr(self.cfg.method, 'cutout_prob', 0.5))
        self.cutout_size_ratio = float(getattr(self.cfg.method, 'cutout_size_ratio', 0.25))

        self.dice_loss = MaskDiceLoss(
            nclass=self.num_classes,
            batch_dice=self.batch_dice,
            ddp=self.is_ddp
        )
        sup_ce_cfg = getattr(self.cfg.method, 'sup_ce_weight', None)
        sup_dice_cfg = getattr(self.cfg.method, 'sup_dice_weight', None)
        self.sup_ce_weight = DEFAULT_SUPERVISED_CE_WEIGHT if sup_ce_cfg is None else float(sup_ce_cfg)
        self.sup_dice_weight = DEFAULT_SUPERVISED_DICE_WEIGHT if sup_dice_cfg is None else float(sup_dice_cfg)

    def _apply_strong_augmentation_gpu(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply 3D Strong Augmentation (Noise, Contrast, 3D Cutout) directly on GPU tensor."""
        bs, c, d, h, w = volume.shape
        augmented = volume.clone()

        # 1. Random Additive Gaussian Noise
        noise_std = random.uniform(self.noise_std_range[0], self.noise_std_range[1])
        augmented = augmented + torch.randn_like(augmented) * noise_std

        # 2. Random Contrast / Intensity Scaling
        scale = random.uniform(self.intensity_scale_range[0], self.intensity_scale_range[1])
        augmented = augmented * scale

        # 3. Random 3D Cutout
        if random.random() < self.cutout_prob:
            cd = max(1, int(d * self.cutout_size_ratio))
            ch = max(1, int(h * self.cutout_size_ratio))
            cw = max(1, int(w * self.cutout_size_ratio))

            for b in range(bs):
                z0 = random.randint(0, d - cd)
                y0 = random.randint(0, h - ch)
                x0 = random.randint(0, w - cw)
                augmented[b, :, z0:z0 + cd, y0:y0 + ch, x0:x0 + cw] = 0.0

        return augmented

    def train_step(
        self,
        sampled_batch: Dict,
        iter_num: int,
        mim_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        volume_batch = sampled_batch['image'].to(self.device)
        label_batch = sampled_batch['label'].to(self.device)

        # 1. Supervised loss on labeled partition (0..labeled_bs)
        labeled_volume = volume_batch[:self.labeled_bs]
        labeled_target = label_batch[:self.labeled_bs]

        lab_out = extract_logits(self.model(labeled_volume))
        loss_ce = F.cross_entropy(lab_out, labeled_target)
        loss_dice = self.dice_loss(lab_out, labeled_target)
        loss_supervised = self.sup_ce_weight * loss_ce + self.sup_dice_weight * loss_dice

        # 2. Unsupervised loss on unlabeled partition (labeled_bs..end)
        unlabeled_volume = volume_batch[self.labeled_bs:]
        if unlabeled_volume.shape[0] > 0:
            # (A) Weak Stream: Unaltered patch evaluated without gradient
            with torch.no_grad():
                weak_logits = extract_logits(self.model(unlabeled_volume))
                weak_prob = F.softmax(weak_logits, dim=1)
                max_prob, pseudo_label = torch.max(weak_prob, dim=1)
                confidence_mask = (max_prob >= self.conf_threshold).float()

            # (B) Strong Stream: Synthesized on GPU and forwarded through model
            strong_volume = self._apply_strong_augmentation_gpu(unlabeled_volume)
            strong_logits = extract_logits(self.model(strong_volume))

            # (C) Cross-Entropy and Dice supervision under confidence mask
            loss_u_ce = compute_masked_ce_loss(strong_logits, pseudo_label, confidence_mask)
            loss_u_dice = self.dice_loss(strong_logits, pseudo_label, mask=confidence_mask)
            loss_unsupervised = self.sup_ce_weight * loss_u_ce + self.sup_dice_weight * loss_u_dice

            conf_ratio = float(confidence_mask.mean().item())
        else:
            loss_unsupervised = torch.tensor(0.0, device=self.device)
            conf_ratio = 1.0

        # Rampup schedule for unsupervised loss weight
        ramp_factor = sigmoid_rampup(iter_num, self.consistency_rampup)
        current_u_weight = self.u_weight * ramp_factor
        total_loss = loss_supervised + current_u_weight * loss_unsupervised

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_supervised": float(loss_supervised.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_dice": float(loss_dice.item()),
            "loss_unsupervised": float(loss_unsupervised.item()),
            "u_weight": float(current_u_weight),
            "confidence_ratio": conf_ratio,
        }
        return total_loss, loss_dict
