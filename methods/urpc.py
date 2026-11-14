"""
Uncertainty Rectified Pyramid Consistency (URPC) Method for Semi-Supervised 3D Medical Segmentation.
Reference: Luo et al., "Semi-supervised medical image segmentation via uncertainty rectified pyramid consistency",
Medical Image Analysis (MedIA) 2022.

Core Mechanism:
- Leverages multi-scale auxiliary decoder predictions.
- Quantifies discrepancy (variance) between intermediate pyramid scale outputs and ensembled consensus.
- Rectifies consistency loss using exponential variance weights: high-variance (uncertain) regions receive lower consistency penalty.
"""

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from data.constants import DEFAULT_SUPERVISED_CE_WEIGHT, DEFAULT_SUPERVISED_DICE_WEIGHT
from methods.base_method import BaseMethod
from models.outputs import extract_multiscale_outputs
from losses import MaskDiceLoss
from utils.ramps import sigmoid_rampup


@METHODS.register("urpc")
class URPCMethod(BaseMethod):
    """
    Uncertainty Rectified Pyramid Consistency (URPC) Framework:
    - Requires backbone with multi-scale deep supervision capability.
    - Rectifies multi-scale consistency with uncertainty weighting.
    """
    REQUIRED_MODEL_CAPABILITIES = {"segmentation", "deep_supervision"}

    def __init__(self, model: nn.Module, cfg):
        super(URPCMethod, self).__init__(model, cfg)

        self.consistency = float(getattr(self.cfg.method, 'consistency', 1.0))
        self.consistency_rampup = float(getattr(self.cfg.method, 'consistency_rampup', 40.0))

        self.dice_loss = MaskDiceLoss(
            nclass=self.num_classes,
            batch_dice=self.batch_dice,
            ddp=self.is_ddp
        )
        sup_ce_cfg = getattr(self.cfg.method, 'sup_ce_weight', None)
        sup_dice_cfg = getattr(self.cfg.method, 'sup_dice_weight', None)
        self.sup_ce_weight = DEFAULT_SUPERVISED_CE_WEIGHT if sup_ce_cfg is None else float(sup_ce_cfg)
        self.sup_dice_weight = DEFAULT_SUPERVISED_DICE_WEIGHT if sup_dice_cfg is None else float(sup_dice_cfg)

    def train_step(
        self,
        sampled_batch: Dict,
        iter_num: int,
        mim_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        volume_batch = sampled_batch['image'].to(self.device)
        label_batch = sampled_batch['label'].to(self.device)

        # 1. Forward model across full batch to retrieve multi-scale outputs
        out = self.model(volume_batch)
        main_logits, aux_logits = extract_multiscale_outputs(out)
        all_logits = [main_logits] + aux_logits

        target_shape = volume_batch.shape[2:]

        # Rescale all pyramid heads to target resolution
        upsampled_logits = []
        for l in all_logits:
            if l.shape[2:] != target_shape:
                upsampled_logits.append(F.interpolate(l, size=target_shape, mode='trilinear', align_corners=False))
            else:
                upsampled_logits.append(l)

        # 2. Supervised loss across all pyramid heads on labeled partition
        lab_targets = label_batch[:self.labeled_bs]
        loss_sup_accum = torch.tensor(0.0, device=self.device)

        for l in upsampled_logits:
            l_lab = l[:self.labeled_bs]
            l_ce = F.cross_entropy(l_lab, lab_targets)
            l_dice = self.dice_loss(l_lab, lab_targets)
            loss_sup_accum = loss_sup_accum + (self.sup_ce_weight * l_ce + self.sup_dice_weight * l_dice)

        loss_supervised = loss_sup_accum / float(len(upsampled_logits))

        # 3. Uncertainty Rectified Pyramid Consistency on unlabeled partition
        num_unlabeled = volume_batch.shape[0] - self.labeled_bs
        if num_unlabeled > 0:
            un_probs = [F.softmax(l[self.labeled_bs:], dim=1) for l in upsampled_logits]
            # Ensemble average across all scales
            preds_avg = torch.mean(torch.stack(un_probs, dim=0), dim=0)

            loss_cons_accum = torch.tensor(0.0, device=self.device)
            for p_scale in un_probs:
                # KL divergence between scale prediction and ensembled average
                kl_dist = torch.sum(p_scale * torch.log((p_scale + 1e-8) / (preds_avg + 1e-8)), dim=1, keepdim=True)
                exp_variance = torch.exp(-kl_dist)

                dist_sq = (preds_avg - p_scale) ** 2
                scale_cons = torch.mean(dist_sq * exp_variance) / (torch.mean(exp_variance) + 1e-8) + torch.mean(kl_dist)
                loss_cons_accum = loss_cons_accum + scale_cons

            loss_consistency = loss_cons_accum / float(len(un_probs))
        else:
            loss_consistency = torch.tensor(0.0, device=self.device)

        cur_epoch = iter_num / max(len(self.trainloader), 1)
        cur_weight = self.consistency * sigmoid_rampup(cur_epoch, self.consistency_rampup)
        total_loss = loss_supervised + cur_weight * loss_consistency

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_supervised": float(loss_supervised.item()),
            "loss_consistency": float(loss_consistency.item()),
            "consistency_weight": float(cur_weight),
            "num_scales": len(upsampled_logits),
        }
        return total_loss, loss_dict
