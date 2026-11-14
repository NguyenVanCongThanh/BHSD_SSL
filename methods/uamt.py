"""
Uncertainty-Aware Mean Teacher (UA-MT) Method for Semi-Supervised 3D Medical Segmentation.
Reference: Yu et al., "Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation", MICCAI 2019.

Core Mechanism:
- Evaluates prediction uncertainty via Monte Carlo perturbation (T forward passes through Teacher).
- Computes voxel-wise entropy H(x) = -sum(p * log(p)).
- Filters out high-uncertainty regions with a progressive ramp-up threshold, allowing consistency
  regularization to apply only to confident, anatomically reliable predictions.
"""

from typing import Dict, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from methods.mean_teacher import MeanTeacherMethod
from methods.ssl_utils import extract_logits, update_ema_variables
from utils.ramps import sigmoid_rampup
from utils.checkpoint import get_raw_model


@METHODS.register("uamt")
class UAMTMethod(MeanTeacherMethod):
    """
    Uncertainty-Aware Mean Teacher (UA-MT) Framework.
    Inherits from MeanTeacherMethod to maintain architecture contract and deep supervision support.
    """
    REQUIRED_MODEL_CAPABILITIES = {"segmentation"}

    def __init__(self, model: nn.Module, cfg):
        super(UAMTMethod, self).__init__(model, cfg)

        # Number of Monte Carlo perturbations for Teacher uncertainty estimation (default T=8)
        self.mc_t = int(getattr(self.cfg.method, 'mc_t', 8))
        self.mc_noise_std = float(getattr(self.cfg.method, 'mc_noise_std', 0.1))
        self.mc_noise_clamp = float(getattr(self.cfg.method, 'mc_noise_clamp', 0.2))

        # Dynamic uncertainty threshold parameters
        self.max_iteration = int(getattr(self.cfg.method, 'max_iteration', 12000))
        self.uncertainty_min_ratio = float(getattr(self.cfg.method, 'uncertainty_min_ratio', 0.75))
        self.uncertainty_max_ratio = float(getattr(self.cfg.method, 'uncertainty_max_ratio', 1.0))

    def _estimate_teacher_uncertainty(self, unlabeled_volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform T stochastic forward passes through EMA Teacher to compute mean prediction and entropy.
        Returns:
            mean_prob: Tensor [B, C, D, H, W]
            uncertainty_entropy: Tensor [B, 1, D, H, W]
        """
        bs, _, d, h, w = unlabeled_volume.shape
        prob_accum = torch.zeros((bs, self.num_classes, d, h, w), device=self.device)

        with torch.no_grad():
            for _ in range(self.mc_t):
                noise = torch.randn_like(unlabeled_volume) * self.mc_noise_std
                noise = torch.clamp(noise, -self.mc_noise_clamp, self.mc_noise_clamp)
                perturbed_input = unlabeled_volume + noise

                t_out = self.ema_model(perturbed_input)
                t_logits = extract_logits(t_out)
                prob_accum += F.softmax(t_logits, dim=1)

        mean_prob = prob_accum / float(self.mc_t)
        entropy = -torch.sum(mean_prob * torch.log(mean_prob + 1e-6), dim=1, keepdim=True)
        return mean_prob, entropy

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

        student_lab_out = self.model(labeled_volume)
        lab_logits = extract_logits(student_lab_out)

        loss_ce = F.cross_entropy(lab_logits, labeled_target)
        loss_dice = self.dice_loss(lab_logits, labeled_target)
        loss_supervised = self.sup_ce_weight * loss_ce + self.sup_dice_weight * loss_dice

        # 2. Uncertainty-filtered consistency on unlabeled partition
        unlabeled_volume = volume_batch[self.labeled_bs:]
        if unlabeled_volume.shape[0] > 0:
            # Teacher Monte Carlo passes
            mean_t_prob, uncertainty = self._estimate_teacher_uncertainty(unlabeled_volume)

            # Progressive uncertainty threshold: tau = (0.75 + 0.25 * rampup) * ln(C)
            ramp_factor = sigmoid_rampup(iter_num, self.consistency_rampup)
            threshold_ratio = self.uncertainty_min_ratio + (self.uncertainty_max_ratio - self.uncertainty_min_ratio) * ramp_factor
            threshold = threshold_ratio * math.log(max(self.num_classes, 2))
            reliable_mask = (uncertainty < threshold).float()

            # Student forward on perturbed unlabeled volume
            if self.perturb_student:
                s_noise = torch.randn_like(unlabeled_volume) * self.student_noise_std
                s_scale = 1.0 + (torch.rand((unlabeled_volume.shape[0], 1, 1, 1, 1), device=self.device) * 2.0 - 1.0) * self.student_intensity_scale
                student_un_in = unlabeled_volume * s_scale + s_noise
            else:
                student_un_in = unlabeled_volume

            student_un_logits = extract_logits(self.model(student_un_in))
            student_un_prob = F.softmax(student_un_logits, dim=1)

            # Softmax MSE consistency filtered by reliable uncertainty mask
            diff_sq = (student_un_prob - mean_t_prob) ** 2
            mask_expanded = reliable_mask.expand_as(diff_sq)
            loss_consistency = torch.sum(diff_sq * mask_expanded) / (2.0 * torch.sum(mask_expanded) + 1e-12)
            reliable_fraction = float(reliable_mask.mean().item())
        else:
            loss_consistency = torch.tensor(0.0, device=self.device)
            reliable_fraction = 1.0

        # Current consistency weight with rampup and rampdown
        cur_epoch = iter_num / max(len(self.trainloader), 1)
        cur_weight = self.get_current_consistency_weight(iter_num, cur_epoch)
        total_loss = loss_supervised + cur_weight * loss_consistency

        # 3. Update EMA Teacher weights
        raw_net = get_raw_model(self.model)
        update_ema_variables(raw_net, self.ema_model, self.ema_decay, iter_num, max_steps=self.max_iteration, schedule=self.ema_schedule)

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_supervised": float(loss_supervised.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_dice": float(loss_dice.item()),
            "loss_consistency": float(loss_consistency.item()),
            "consistency_weight": float(cur_weight),
            "reliable_mask_ratio": reliable_fraction,
        }
        return total_loss, loss_dict
