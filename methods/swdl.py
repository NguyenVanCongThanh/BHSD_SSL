"""
Stratum-Wise Difference Learning (SWDL) Method for Semi-Supervised 3D Medical Segmentation.
Reference: Wang et al., "SWDL: Stratum-Wise Difference Learning with Deep Laplacian Pyramid
for Semi-Supervised 3D Intracranial Hemorrhage Segmentation", BSPC 2026.

Integrates:
- Multi-step iterative feedback loop between Dual Decoders.
- Deep Laplacian Pyramid edge reconstruction with multi-stratum feature consistency.
- Deep supervision across all hierarchical feature levels.
"""

from typing import Dict, Tuple, List, Optional, Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from data.constants import DEFAULT_SUPERVISED_CE_WEIGHT, DEFAULT_SUPERVISED_DICE_WEIGHT
from methods.base_method import BaseMethod
from losses import MaskDiceLoss
from utils.ramps import sigmoid_rampup

DEFAULT_SWDL_DS_WEIGHTS = [0.8, 0.6, 0.4, 0.2, 0.1]


@METHODS.register("swdl")
class SWDLMethod(BaseMethod):
    """
    SWDL Semi-Supervised Learning Framework:
    - Multi-step feedback loop (default 3 steps per batch).
    - Laplacian Pyramid dual-decoder difference supervision.
    - Consistency regularization across dual branches.
    """
    REQUIRED_MODEL_CAPABILITIES = {"segmentation", "dual_feedback"}
    DEFAULT_MODEL_ALIAS = {"swdl": "swdl_net", "vnet": "swdl_net"}

    def __init__(self, model: nn.Module, cfg):
        super(SWDLMethod, self).__init__(model, cfg)

        self.consistency = float(getattr(self.cfg.method, 'consistency', 1.0))
        self.consistency_rampup = float(getattr(self.cfg.method, 'consistency_rampup', 40.0))
        self.feedback_steps = int(getattr(self.cfg.method, 'feedback_steps', 3))
        self.feedback_scale = float(getattr(self.cfg.method, 'feedback_scale', 1e-3))
        self.ds_weights = getattr(self.cfg.method, 'ds_weights', DEFAULT_SWDL_DS_WEIGHTS)

        sup_ce_cfg = getattr(self.cfg.method, 'sup_ce_weight', None)
        sup_dice_cfg = getattr(self.cfg.method, 'sup_dice_weight', None)
        self.sup_ce_weight = DEFAULT_SUPERVISED_CE_WEIGHT if sup_ce_cfg is None else float(sup_ce_cfg)
        self.sup_dice_weight = DEFAULT_SUPERVISED_DICE_WEIGHT if sup_dice_cfg is None else float(sup_dice_cfg)

        self.dice_loss = MaskDiceLoss(
            nclass=self.num_classes,
            batch_dice=self.batch_dice,
            ddp=self.is_ddp
        )

    def get_current_consistency_weight(self, epoch: float) -> float:
        return self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

    def _feedback_steps(
        self,
        sampled_batch: Dict,
        iter_num: int,
        mim_mask: Optional[torch.Tensor] = None
    ) -> Iterator[Tuple[torch.Tensor, Dict[str, float]]]:
        volume_batch = sampled_batch['image'].to(self.device)
        label_batch = sampled_batch['label'].to(self.device)

        cur_epoch = iter_num / max(len(self.trainloader), 1)
        consistency_weight = self.get_current_consistency_weight(cur_epoch)

        en: Optional[List[torch.Tensor]] = None

        for step in range(self.feedback_steps):
            if step == 0:
                outputs1, outputs2, masks, deep_outs = self.model(volume_batch, None)
            else:
                outputs1, outputs2, masks, deep_outs = self.model(volume_batch, en)

            # Compute Stratum-Wise Difference Signal: en = scale * (feat2 - feat1)
            en = []
            feat2_list, feat1_list = masks[0], masks[1]
            for idx in range(len(feat1_list)):
                m1 = feat1_list[idx].detach()
                m2 = feat2_list[idx].detach()
                en.append(self.feedback_scale * (m2 - m1))

            # 1. Supervised loss on labeled partition (0..labeled_bs)
            out1_lab = outputs1[:self.labeled_bs]
            out2_lab = outputs2[:self.labeled_bs]
            target_lab = label_batch[:self.labeled_bs]

            l_ce1 = F.cross_entropy(out1_lab, target_lab)
            l_dice1 = self.dice_loss(out1_lab, target_lab)
            loss_sup1 = self.sup_ce_weight * l_ce1 + self.sup_dice_weight * l_dice1

            l_ce2 = F.cross_entropy(out2_lab, target_lab)
            l_dice2 = self.dice_loss(out2_lab, target_lab)
            loss_sup2 = self.sup_ce_weight * l_ce2 + self.sup_dice_weight * l_dice2

            loss_sup = loss_sup1 + loss_sup2

            # 2. Multi-level deep supervision
            loss_ds = torch.tensor(0.0, device=self.device)
            for ds_idx, ds_out in enumerate(deep_outs):
                ds_lab = ds_out[:self.labeled_bs]
                w = self.ds_weights[ds_idx] if ds_idx < len(self.ds_weights) else 0.1
                l_ce = F.cross_entropy(ds_lab, target_lab)
                l_dice = self.dice_loss(ds_lab, target_lab)
                loss_ds = loss_ds + w * (self.sup_ce_weight * l_ce + self.sup_dice_weight * l_dice)

            # 3. Stratum-wise consistency between Decoder 1 and Decoder 2
            prob1 = F.softmax(outputs1, dim=1)
            prob2 = F.softmax(outputs2, dim=1)
            loss_cons = torch.mean((prob1 - prob2) ** 2)

            loss_step = loss_sup + loss_ds + consistency_weight * loss_cons

            yield loss_step, {
                'loss_total': float(loss_step.detach()),
                'loss_supervised': float(loss_sup.detach()),
                'loss_deep_supervision': float(loss_ds.detach()),
                'loss_consistency': float(loss_cons.detach()),
                'consistency_weight': float(consistency_weight),
            }

    def train_step(
        self,
        sampled_batch: Dict,
        iter_num: int,
        mim_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        steps = list(self._feedback_steps(sampled_batch, iter_num, mim_mask))
        mean_loss = sum(loss for loss, _ in steps) / len(steps)
        mean_logs = {k: sum(log[k] for _, log in steps) / len(steps) for k in steps[0][1]}
        return mean_loss, mean_logs

    def optimize_batch(self, sampled_batch, iter_num, mim_mask=None, step_fn=None):
        """Execute gradient updates across feedback steps with AMP support."""
        if step_fn is not None and step_fn != self.train_step:
            return super().optimize_batch(sampled_batch, iter_num, mim_mask, step_fn)
        steps = []
        generator = self._feedback_steps(sampled_batch, iter_num, mim_mask)
        while True:
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                try:
                    loss, logs = next(generator)
                except StopIteration:
                    break
            self.apply_gradients(loss, iter_num)
            steps.append((loss.detach(), logs))
        mean_logs = {k: sum(log[k] for _, log in steps) / len(steps) for k in steps[0][1]}
        return steps[-1][0], mean_logs
