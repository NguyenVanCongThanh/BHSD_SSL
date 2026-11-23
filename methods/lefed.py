"""
LeFeD (Learning Feedback Dual-Branch) Semi-Supervised Segmentation Method.
Reference: "Learning Feedback for Semi-Supervised 3D Medical Image Segmentation".
Adapted & Optimized for 3D Brain Hemorrhage Segmentation (BHSD) Binary & Multi-class settings.
"""

from contextlib import nullcontext
import logging
import math
from typing import Dict, Tuple, List, Optional, Iterator, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from data.constants import (
    DEFAULT_CLIP_NORM,
    DEFAULT_CONSISTENCY_DICE_WEIGHT,
    DEFAULT_CONSISTENCY_MSE_WEIGHT,
    DEFAULT_FG_CONFIDENCE_MAX_THRESH,
    DEFAULT_SUPERVISED_CE_WEIGHT,
    DEFAULT_SUPERVISED_DICE_WEIGHT,
)
from losses import MaskDiceLoss, ConsistencyLoss
from losses.deep_supervision import downsample_target_for_ds
from methods.base_method import BaseMethod
from utils.checkpoint import get_raw_model
from utils.ramps import sigmoid_rampup

DEFAULT_LEFED_DS_WEIGHTS = [0.8, 0.6, 0.4, 0.2, 0.1]


@METHODS.register("lefed")
class LeFeDMethod(BaseMethod):
    """
    LeFeD Framework for Semi-Supervised 3D Medical Image Segmentation:
    - Multi-step Feedback Loop (sequential or gradient-accumulated across feedback passes).
    - Dual-Decoders with feature-guided modulation.
    - Multi-scale Deep Supervision loss (ordered finest -> coarsest, respecting ds_zero_lowest_scale & ds_downsample_target).
    - Class-balanced Consistency Regularization across dual branches.
    """
    REQUIRED_MODEL_CAPABILITIES = {"segmentation", "dual_feedback"}
    DEFAULT_MODEL_ALIAS = {"lefed": "lefed_net"}

    def __init__(self, model: nn.Module, cfg):
        super(LeFeDMethod, self).__init__(model, cfg)

        self.consistency = float(getattr(self.cfg.method, 'consistency', 1.0))
        self.consistency_rampup = float(getattr(self.cfg.method, 'consistency_rampup', 40.0))
        rampdown_val = getattr(self.cfg.method, 'consistency_rampdown_start', None)
        self.consistency_rampdown_start = 0 if (rampdown_val is None or rampdown_val == 'null') else int(rampdown_val)
        self.consistency_min_factor = float(getattr(self.cfg.method, 'consistency_min_factor', 0.25))
        self.consistency_type = str(getattr(self.cfg.method, 'consistency_type', 'mse')).lower()
        self.confidence_thresh = float(getattr(self.cfg.method, 'confidence_thresh', 0.0))
        self.feedback_steps = int(getattr(self.cfg.method, 'feedback_steps', 3))
        self.feedback_scale = float(getattr(self.cfg.method, 'feedback_scale', 1e-3))
        self.feedback_update_mode = str(getattr(self.cfg.method, 'feedback_update_mode', 'sequential')).strip().lower()
        if self.feedback_update_mode not in ('sequential', 'accumulate'):
            raise ValueError(
                f"feedback_update_mode must be 'sequential' or 'accumulate', got {self.feedback_update_mode!r}"
            )
        self.ds_weights = list(getattr(self.cfg.method, 'ds_weights', DEFAULT_LEFED_DS_WEIGHTS))
        ds_loss_w_cfg = getattr(self.cfg.method, 'ds_loss_weight', None)
        self.ds_loss_weight = 0.5 if ds_loss_w_cfg is None else float(ds_loss_w_cfg)
        sup_ce_cfg = getattr(self.cfg.method, 'sup_ce_weight', None)
        sup_dice_cfg = getattr(self.cfg.method, 'sup_dice_weight', None)
        self.sup_ce_weight = DEFAULT_SUPERVISED_CE_WEIGHT if sup_ce_cfg is None else float(sup_ce_cfg)
        self.sup_dice_weight = DEFAULT_SUPERVISED_DICE_WEIGHT if sup_dice_cfg is None else float(sup_dice_cfg)

        # Background vs Foreground weight balance for consistency (applies to both binary and multiclass)
        if self.num_classes >= 2:
            bg_weight = float(getattr(self.cfg.method, 'bg_consistency_weight', 0.1))
            weights = [bg_weight] + [1.0] * (self.num_classes - 1)
            self.class_weights = torch.tensor(weights, device=self.device).view(1, -1, 1, 1, 1)
        else:
            self.class_weights = None

        # Supervised Dice excludes background for both binary and multi-class (aligned with SupervisedMethod & paper)
        self.dice_loss = MaskDiceLoss(
            nclass=self.num_classes,
            include_background=False,
            batch_dice=self.batch_dice,
            ddp=self.is_ddp,
        )

        fg_conf_cfg = getattr(self.cfg.method, 'fg_confidence_max_thresh', None)
        cons_dice_cfg = getattr(self.cfg.method, 'consistency_dice_weight', None)
        cons_mse_cfg = getattr(self.cfg.method, 'consistency_mse_weight', None)
        self.consistency_loss = ConsistencyLoss(
            consistency_type=self.consistency_type,
            num_classes=self.num_classes,
            is_binary=self.is_binary,
            confidence_thresh=self.confidence_thresh,
            class_weights=self.class_weights,
            fg_confidence_max_thresh=(
                DEFAULT_FG_CONFIDENCE_MAX_THRESH if fg_conf_cfg is None else float(fg_conf_cfg)
            ),
            weight_dice=(
                DEFAULT_CONSISTENCY_DICE_WEIGHT if cons_dice_cfg is None else float(cons_dice_cfg)
            ),
            weight_mse=(
                DEFAULT_CONSISTENCY_MSE_WEIGHT if cons_mse_cfg is None else float(cons_mse_cfg)
            ),
            batch_dice=self.batch_dice,
            ddp=self.is_ddp,
        )

    def _resolve_ds_stage_weights(self, num_scales: int) -> List[float]:
        """Resolve normalized high-to-low deep supervision weights, honoring ds_zero_lowest_scale."""
        if num_scales <= 0:
            return []
        raw = [
            float(self.ds_weights[idx]) if idx < len(self.ds_weights) else float(self.ds_weights[-1] if self.ds_weights else 0.1)
            for idx in range(num_scales)
        ]
        if self.ds_zero_lowest_scale and num_scales >= 3:
            raw[-1] = 0.0
        total = sum(raw)
        if total <= 0.0:
            return [0.0] * num_scales
        return [self.ds_loss_weight * (w / total) for w in raw]

    def get_current_consistency_weight(self, epoch: float, iter_num: int = 0) -> float:
        if self.consistency_rampup <= 0:
            base_w = self.consistency
        elif self.consistency_rampup > 100:
            base_w = self.consistency * sigmoid_rampup(iter_num, self.consistency_rampup)
        else:
            base_w = self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

        max_iterations = int(getattr(self.cfg.method, 'max_iteration', 12000))
        if self.consistency_rampdown_start > 0 and iter_num > self.consistency_rampdown_start:
            span = max(1, max_iterations - self.consistency_rampdown_start)
            progress = min(1.0, (iter_num - self.consistency_rampdown_start) / span)
            decay_factor = self.consistency_min_factor + 0.5 * (1.0 - self.consistency_min_factor) * (1.0 + math.cos(math.pi * progress))
            return base_w * decay_factor
        return base_w

    def _feedback_steps(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Iterator[Tuple[torch.Tensor, Dict[str, float]]]:
        volume_batch = sampled_batch['image'].to(self.device)
        label_batch = sampled_batch['label'].to(self.device)

        cur_epoch = iter_num / max(len(self.trainloader), 1)
        consistency_weight = self.get_current_consistency_weight(cur_epoch, iter_num=iter_num)

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

            # 1. Balanced supervised loss on labeled samples (Decoder 1 & Decoder 2)
            out1_lab = outputs1[:self.labeled_bs]
            out2_lab = outputs2[:self.labeled_bs]
            target_lab = label_batch[:self.labeled_bs]

            loss_sup1 = self.sup_ce_weight * F.cross_entropy(out1_lab, target_lab) + self.sup_dice_weight * self.dice_loss(out1_lab, target_lab)
            loss_sup2 = self.sup_ce_weight * F.cross_entropy(out2_lab, target_lab) + self.sup_dice_weight * self.dice_loss(out2_lab, target_lab)
            loss_sup = 0.5 * (loss_sup1 + loss_sup2)

            # 2. Deep Supervision loss on multi-scale side outputs (finest -> coarsest)
            loss_ds: Union[torch.Tensor, float] = 0.0
            stage_weights = self._resolve_ds_stage_weights(len(deep_outs))
            target_shape = target_lab.shape[1:]
            for ds_out, w in zip(deep_outs, stage_weights):
                if w <= 0.0:
                    continue
                ds_lab = ds_out[:self.labeled_bs]
                if ds_lab.shape[2:] != target_shape:
                    if self.ds_downsample_target:
                        stage_target = downsample_target_for_ds(target_lab, ds_lab.shape[2:])
                    else:
                        ds_lab = F.interpolate(ds_lab, size=target_shape, mode='trilinear', align_corners=False)
                        stage_target = target_lab
                else:
                    stage_target = target_lab
                l_ce = F.cross_entropy(ds_lab, stage_target)
                l_dice = self.dice_loss(ds_lab, stage_target)
                loss_ds = loss_ds + w * (self.sup_ce_weight * l_ce + self.sup_dice_weight * l_dice)

            # 3. Consistency loss between Decoder 1 and Decoder 2 across all samples
            loss_cons = self.consistency_loss(outputs1, outputs2)

            # Step total loss
            loss_step = loss_sup + loss_ds + consistency_weight * loss_cons

            yield loss_step, {
                'total': float(loss_step.detach()),
                'supervised': float(loss_sup.detach()),
                'deep_supervision': float(loss_ds.detach()) if isinstance(loss_ds, torch.Tensor) else float(loss_ds),
                'consistency': float(loss_cons.detach()),
                'weight': float(consistency_weight),
            }

    def train_step(self, sampled_batch, iter_num, mim_mask=None):
        """Loss-only API for inspection; optimization uses feedback passes in optimize_batch."""
        steps = list(self._feedback_steps(sampled_batch, iter_num, mim_mask))
        return sum(loss for loss, _ in steps) / len(steps), self._mean_logs(steps)

    @staticmethod
    def _mean_logs(steps):
        return {key: sum(log[key] for _, log in steps) / len(steps) for key in steps[0][1]}

    def _optimize_batch_accumulated(self, sampled_batch, iter_num, mim_mask=None):
        """
        Accumulate gradients across all feedback passes and apply a single optimizer step
        and single DDP gradient sync per iteration (aligned with momentum=0.99 & Poly LR).
        """
        clip_norm_cfg = getattr(self.cfg.method, 'clip_norm', None)
        clip_norm = DEFAULT_CLIP_NORM if clip_norm_cfg is None else float(clip_norm_cfg)

        self.optimizer.zero_grad(set_to_none=True)
        steps = []
        generator = self._feedback_steps(sampled_batch, iter_num, mim_mask)

        for step_idx in range(self.feedback_steps):
            is_last = (step_idx == self.feedback_steps - 1)
            sync_ctx = (
                self.model.no_sync()
                if (self.is_ddp and not is_last and hasattr(self.model, 'no_sync'))
                else nullcontext()
            )
            with sync_ctx:
                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    loss, logs = next(generator)
                loss_finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
                if self.is_ddp:
                    torch.distributed.all_reduce(loss_finite, op=torch.distributed.ReduceOp.MIN)
                if not loss_finite.item():
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f'Invalid training loss at iteration {iter_num}')

                scaled_loss = loss / float(self.feedback_steps)
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                steps.append((loss.detach(), logs))

        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            is_finite = torch.ones((), device=self.device, dtype=torch.int32)
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        is_finite.mul_(torch.isfinite(param.grad).all().to(torch.int32))
            if self.is_ddp:
                torch.distributed.all_reduce(is_finite, op=torch.distributed.ReduceOp.MIN)

            scale_before = self.scaler.get_scale()
            if is_finite.item() == 0:
                new_scale = scale_before * self.scaler.get_backoff_factor()
                self.scaler.update(new_scale=new_scale)
                logging.warning(
                    '[AMP] Iteration %d: Nonfinite gradient overflow detected across feedback passes; '
                    'optimizer step skipped, scale adjusted from %.1f to %.1f',
                    iter_num, scale_before, new_scale,
                )
                if new_scale < 1e-4:
                    raise FloatingPointError(f'GradScaler scale underflow at iteration {iter_num}: model or loss diverged')
            else:
                torch.nn.utils.clip_grad_norm_(get_raw_model(self.model).parameters(), max_norm=clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.after_optimizer_step(iter_num)
                if hasattr(self, 'successful_updates'):
                    self.successful_updates += 1
        else:
            torch.nn.utils.clip_grad_norm_(get_raw_model(self.model).parameters(), max_norm=clip_norm)
            finite = torch.ones((), device=self.device, dtype=torch.int32)
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        finite.mul_(torch.isfinite(param.grad).all().to(torch.int32))
            if self.is_ddp:
                torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f'Nonfinite gradients at iteration {iter_num}')
            self.optimizer.step()
            self.after_optimizer_step(iter_num)
            if hasattr(self, 'successful_updates'):
                self.successful_updates += 1

        mean_loss = sum(l for l, _ in steps) / len(steps)
        return mean_loss, self._mean_logs(steps)

    def optimize_batch(self, sampled_batch, iter_num, mim_mask=None, step_fn=None):
        """Run feedback passes with either sequential per-step updates or single accumulated update."""
        if step_fn is not None and step_fn != self.train_step:
            return super().optimize_batch(sampled_batch, iter_num, mim_mask, step_fn)
        if self.feedback_update_mode == 'accumulate':
            return self._optimize_batch_accumulated(sampled_batch, iter_num, mim_mask)

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
