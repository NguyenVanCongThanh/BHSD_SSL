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

    def train_step(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        volume_batch = sampled_batch['image'][:self.labeled_bs].to(self.device)
        label_batch = sampled_batch['label'][:self.labeled_bs].to(self.device)

        out = self.model(volume_batch)
        main_logits, aux_logits = extract_multiscale_outputs(out)

        loss_ce = F.cross_entropy(main_logits, label_batch)
        loss_dice = self.dice_loss(main_logits, label_batch)
        loss_main = self.sup_ce_weight * loss_ce + self.sup_dice_weight * loss_dice

        loss_ds_accum = 0.0
        if self.ds_enabled and len(aux_logits) > 0:
            all_logits = [main_logits] + aux_logits
            norm_w = resolve_ds_weights(len(all_logits), self.ds_weights)
            total_loss = norm_w[0] * loss_main
            target_shape = label_batch.shape[1:]

            for idx in range(1, len(all_logits)):
                aux_pred = all_logits[idx]
                if aux_pred.shape[2:] != target_shape:
                    aux_pred = F.interpolate(aux_pred, size=target_shape, mode='trilinear', align_corners=False)
                l_stage = self.sup_ce_weight * F.cross_entropy(aux_pred, label_batch) + self.sup_dice_weight * self.dice_loss(aux_pred, label_batch)
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

