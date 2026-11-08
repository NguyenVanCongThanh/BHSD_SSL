"""
Supervised Baseline Method (Lower Bound & Fair Comparison).
Trains network purely on available labeled samples using combined Cross-Entropy + Dice Loss,
with full support for Deep Supervision matching nnU-Net design.
"""

from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from data.constants import DEFAULT_SUPERVISED_CE_WEIGHT, DEFAULT_SUPERVISED_DICE_WEIGHT
from methods.base_method import BaseMethod
from models.outputs import extract_multiscale_outputs
from losses import MaskDiceLoss
from losses.deep_supervision import resolve_ds_weights


@METHODS.register("supervised")
class SupervisedMethod(BaseMethod):
    """
    Supervised Baseline Engine.
    Supports single-scale and multi-scale deep supervision matching nnU-Net PlainConvUNet.
    """
    def __init__(self, model: nn.Module, cfg):
        super(SupervisedMethod, self).__init__(model, cfg)
        self.dice_loss = MaskDiceLoss(nclass=self.num_classes)
        sup_ce_cfg = getattr(self.cfg.method, 'sup_ce_weight', None)
        sup_dice_cfg = getattr(self.cfg.method, 'sup_dice_weight', None)
        self.sup_ce_weight = DEFAULT_SUPERVISED_CE_WEIGHT if sup_ce_cfg is None else float(sup_ce_cfg)
        self.sup_dice_weight = DEFAULT_SUPERVISED_DICE_WEIGHT if sup_dice_cfg is None else float(sup_dice_cfg)

        # Deep supervision configuration
        model_caps = set(getattr(getattr(self.model, 'module', self.model), 'CAPABILITIES', set()))
        model_ds = bool(getattr(self.cfg.model, 'deep_supervision', False))
        supports_ds = ('deep_supervision' in model_caps) or model_ds

        method_ds = getattr(self.cfg.method, 'deep_supervision', False)
        if isinstance(method_ds, dict) or hasattr(method_ds, 'weights'):
            ds_cfg = method_ds
            ds_flag = bool(getattr(ds_cfg, 'enabled', True)) if 'enabled' in ds_cfg else True
            self.ds_weights = ds_cfg.get('weights')
        elif isinstance(method_ds, bool):
            ds_flag = method_ds
            self.ds_weights = None
        else:
            ds_flag = False
            self.ds_weights = None

        self.ds_enabled = bool(supports_ds and (ds_flag or model_ds))

        # Loss type configuration (backward-compatible: defaults to standard 'ce_dice')
        self.loss_type = str(getattr(self.cfg.method, 'loss_type', 'ce_dice')).lower()
        if self.loss_type == 'focal_tversky':
            from losses.focal_tversky import FocalTverskyLoss
            tversky_alpha = float(getattr(self.cfg.method, 'tversky_alpha', 0.3))
            tversky_beta = float(getattr(self.cfg.method, 'tversky_beta', 0.7))
            focal_gamma = float(getattr(self.cfg.method, 'focal_gamma', 2.0))
            self.focal_tversky_loss = FocalTverskyLoss(
                nclass=self.num_classes,
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=focal_gamma,
                weight_tversky=self.sup_dice_weight,
                weight_focal=self.sup_ce_weight,
            )
        else:
            self.focal_tversky_loss = None

        self.ds_rampdown_start = getattr(self.cfg.method, 'ds_rampdown_start', None)
        self.max_iteration = int(getattr(self.cfg.method, 'max_iteration', 25000))

    def _compute_stage_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'focal_tversky' and self.focal_tversky_loss is not None:
            return self.focal_tversky_loss(pred, target)
        return self.sup_ce_weight * F.cross_entropy(pred, target) + self.sup_dice_weight * self.dice_loss(pred, target)

    def train_step(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        volume_batch = sampled_batch['image'][:self.labeled_bs].to(self.device)
        label_batch = sampled_batch['label'][:self.labeled_bs].to(self.device)

        out = self.model(volume_batch)
        main_logits, aux_logits = extract_multiscale_outputs(out)

        if self.loss_type == 'focal_tversky' and self.focal_tversky_loss is not None:
            loss_ce = self.focal_tversky_loss.focal(main_logits, label_batch)
            loss_dice = self.focal_tversky_loss.tversky(main_logits, label_batch)
            loss_main = self.focal_tversky_loss(main_logits, label_batch)
        else:
            loss_ce = F.cross_entropy(main_logits, label_batch)
            loss_dice = self.dice_loss(main_logits, label_batch)
            loss_main = self.sup_ce_weight * loss_ce + self.sup_dice_weight * loss_dice

        loss_ds_accum = 0.0
        if self.ds_enabled and len(aux_logits) > 0:
            all_logits = [main_logits] + aux_logits
            norm_w = list(resolve_ds_weights(len(all_logits), self.ds_weights))

            # Optional deep supervision decay/rampdown in late iterations
            if self.ds_rampdown_start is not None and iter_num >= int(self.ds_rampdown_start):
                ramp_span = max(1, self.max_iteration - int(self.ds_rampdown_start))
                ramp_factor = max(0.0, 1.0 - (iter_num - int(self.ds_rampdown_start)) / ramp_span)
                for i in range(1, len(norm_w)):
                    norm_w[i] = norm_w[i] * ramp_factor

            total_loss = norm_w[0] * loss_main
            target_shape = label_batch.shape[1:]

            for idx in range(1, len(all_logits)):
                if norm_w[idx] <= 0.0:
                    continue
                aux_pred = all_logits[idx]
                if aux_pred.shape[2:] != target_shape:
                    aux_pred = F.interpolate(aux_pred, size=target_shape, mode='trilinear', align_corners=False)
                l_stage = self._compute_stage_loss(aux_pred, label_batch)
                total_loss = total_loss + norm_w[idx] * l_stage
                loss_ds_accum = loss_ds_accum + norm_w[idx] * float(l_stage.item())
        else:
            total_loss = loss_main

        loss_dict = {
            "total": float(total_loss.item()),
            "ce": float(loss_ce.item()),
            "dice": float(loss_dice.item())
        }
        if self.ds_enabled and len(aux_logits) > 0:
            loss_dict["main_loss"] = float(loss_main.item())
            loss_dict["deep_supervision"] = float(loss_ds_accum)

        return total_loss, loss_dict

