"""
LeFeD (Learning Feedback Dual-Branch) Semi-Supervised Segmentation Method.
Reference: "Learning Feedback for Semi-Supervised 3D Medical Image Segmentation".
Adapted & Optimized for 3D Brain Hemorrhage Segmentation (BHSD) Multi-class setting.
"""

from typing import Dict, Tuple, List, Optional, Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from methods.base_method import BaseMethod
from losses import MaskDiceLoss
from utils.ramps import sigmoid_rampup

@METHODS.register("lefed")
class LeFeDMethod(BaseMethod):
    """
    LeFeD Framework for Semi-Supervised 3D Medical Image Segmentation:
    - Multi-step Feedback Loop (3 forward passes per batch).
    - Dual-Decoders with feature-guided modulation.
    - Multi-scale Deep Supervision loss.
    - Class-balanced Consistency Regularization across dual branches.
    """
    def __init__(self, model: nn.Module, cfg):
        super(LeFeDMethod, self).__init__(model, cfg)

        self.consistency = float(getattr(self.cfg.method, 'consistency', 1.0))
        self.consistency_rampup = float(getattr(self.cfg.method, 'consistency_rampup', 40.0))
        self.feedback_steps = int(getattr(self.cfg.method, 'feedback_steps', 3))
        self.feedback_scale = float(getattr(self.cfg.method, 'feedback_scale', 1e-3))
        self.ds_weights = getattr(self.cfg.method, 'ds_weights', [0.8, 0.6, 0.4, 0.2, 0.1])

        # Background vs Foregrounds weight balance for consistency
        if not self.is_binary and self.num_classes == 6:
            bg_weight = float(getattr(self.cfg.method, 'bg_consistency_weight', 0.1))
            self.class_weights = torch.tensor([bg_weight, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.device).view(1, -1, 1, 1, 1)
        else:
            self.class_weights = None

        # Supervised loss (exclude background from Dice for multi-class)
        self.dice_loss = MaskDiceLoss(nclass=self.num_classes, include_background=self.is_binary)

    def get_current_consistency_weight(self, epoch: float) -> float:
        if self.consistency_rampup <= 0:
            return self.consistency
        return self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

    def _feedback_steps(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Iterator[Tuple[torch.Tensor, Dict[str, float]]]:
        volume_batch = sampled_batch['image'].to(self.device)
        label_batch = sampled_batch['label'].to(self.device)

        cur_epoch = iter_num / max(len(self.trainloader), 1)
        consistency_weight = self.get_current_consistency_weight(cur_epoch)

        en: Optional[List[torch.Tensor]] = None

        # Feedback iteration loop (default: 3 steps)
        if self.feedback_steps < 1:
            raise ValueError('feedback_steps must be positive')
        for step in range(self.feedback_steps):
            if step == 0:
                outputs1, outputs2, masks, deep_outs = self.model(volume_batch, None)
            else:
                outputs1, outputs2, masks, deep_outs = self.model(volume_batch, en)

            # Compute feedback signal en for the next step: en = scale * (stage_feat2 - stage_feat1)
            # masks is [stage_feat2, stage_feat1]
            en = []
            feat2_list, feat1_list = masks[0], masks[1]
            for idx in range(len(feat1_list)):
                m1 = feat1_list[idx].detach()
                m2 = feat2_list[idx].detach()
                en.append(self.feedback_scale * (m2 - m1))

            # 1. Supervised loss on labeled samples (Decoder 1 & Decoder 2)
            out1_lab = outputs1[:self.labeled_bs]
            out2_lab = outputs2[:self.labeled_bs]
            target_lab = label_batch[:self.labeled_bs]

            loss_sup1 = (F.cross_entropy(out1_lab, target_lab) + self.dice_loss(out1_lab, target_lab)) / 2.0
            loss_sup2 = (F.cross_entropy(out2_lab, target_lab) + self.dice_loss(out2_lab, target_lab)) / 2.0
            loss_sup = loss_sup1 + loss_sup2

            # 2. Deep Supervision loss on multi-scale side outputs
            loss_ds = 0.0
            for ds_idx, ds_out in enumerate(deep_outs):
                ds_lab = ds_out[:self.labeled_bs]
                w = self.ds_weights[ds_idx] if ds_idx < len(self.ds_weights) else 0.1
                l_ce = F.cross_entropy(ds_lab, target_lab)
                l_dice = self.dice_loss(ds_lab, target_lab)
                loss_ds = loss_ds + w * ((l_ce + l_dice) / 2.0)

            # 3. Consistency loss between Decoder 1 and Decoder 2 across all samples
            prob1 = F.softmax(outputs1, dim=1)
            prob2 = F.softmax(outputs2, dim=1)
            sq_diff = (prob1 - prob2) ** 2
            if self.class_weights is not None:
                sq_diff = sq_diff * self.class_weights
            loss_cons = sq_diff.mean()

            # Step total loss
            loss_step = loss_sup + loss_ds + consistency_weight * loss_cons

            yield loss_step, {
                'total': float(loss_step.detach()), 'supervised': float(loss_sup.detach()),
                'deep_supervision': float(loss_ds.detach()) if isinstance(loss_ds, torch.Tensor) else float(loss_ds),
                'consistency': float(loss_cons.detach()),
                'weight': float(consistency_weight)}


    def train_step(self, sampled_batch, iter_num, mim_mask=None):
        """Loss-only API for inspection; optimization uses sequential feedback steps."""
        steps = list(self._feedback_steps(sampled_batch, iter_num, mim_mask))
        return sum(loss for loss, _ in steps) / len(steps), self._mean_logs(steps)

    @staticmethod
    def _mean_logs(steps):
        return {key: sum(log[key] for _, log in steps) / len(steps) for key in steps[0][1]}

    def optimize_batch(self, sampled_batch, iter_num, mim_mask=None, step_fn=None):
        """Each feedback pass gets exactly one backward/update, including AMP."""
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
        return steps[-1][0], self._mean_logs(steps)
