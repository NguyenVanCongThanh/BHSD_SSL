"""
Mean Teacher Method for Semi-Supervised 3D Medical Segmentation (BHSD).
Reference: Tarvainen & Valpola, "Mean teachers are better role models:
Weight-averaged consistency targets improve semi-supervised deep learning results", NeurIPS 2017.

Adapted & Optimized for 3D Brain Hemorrhage Segmentation (Extreme class imbalance, 6 classes, DDP compatible).
"""

import math
from typing import Dict, Tuple, List, Optional, Union, Any
import os
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from core.distributed import is_main_process
from methods.base_method import BaseMethod
from utils.losses import mask_DiceLoss
from utils.ramps import sigmoid_rampup
from utils.deep_supervision import extract_multiscale_outputs, compute_ds_weights
from evaluation.sliding_window import evaluate_dataset_sliding_window


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


@METHODS.register("mean_teacher")
class MeanTeacherMethod(BaseMethod):
    """
    Enhanced Mean Teacher SSL Framework for 3D Multiclass Segmentation:
    - Teacher model updated via Exponential Moving Average (EMA).
    - Class-balanced & Confidence-thresholded consistency loss to handle severe voxel imbalance.
    - Asymmetric input perturbation on Student to eliminate confirmation bias.
    - Late-stage consistency rampdown to preserve rare class features (IVH).
    - Full DDP and sliding-window validation on both Student & Teacher models.
    """
    def __init__(self, model: nn.Module, cfg):
        super(MeanTeacherMethod, self).__init__(model, cfg)

        self.labeled_bs = int(getattr(self.cfg.data, 'labeled_bs', 2))
        self.consistency = float(getattr(self.cfg.method, 'consistency', 0.05))
        self.consistency_rampup = float(getattr(self.cfg.method, 'consistency_rampup', 4000.0))
        self.ema_decay = float(getattr(self.cfg.method, 'ema_decay', 0.999))
        self.consistency_type = str(getattr(self.cfg.method, 'consistency_type', 'hybrid')).lower()
        self.confidence_thresh = float(getattr(self.cfg.method, 'confidence_thresh', 0.0))

        # Asymmetric student perturbation to avoid confirmation bias
        self.perturb_student = bool(getattr(self.cfg.method, 'perturb_student', True))
        self.student_noise_std = float(getattr(self.cfg.method, 'student_noise_std', 0.02))
        self.student_intensity_scale = float(getattr(self.cfg.method, 'student_intensity_scale', 0.05))

        # Late-stage consistency rampdown (preserves rare classes such as IVH from drifting)
        max_iter = int(getattr(self.cfg.method, 'max_iteration', 12000))
        self.consistency_rampdown_start = int(getattr(self.cfg.method, 'consistency_rampdown_start', int(max_iter * 0.65)))
        self.consistency_min_factor = float(getattr(self.cfg.method, 'consistency_min_factor', 0.25))

        # EMA decay schedule: 'cosine' or 'step'
        self.ema_schedule = str(getattr(self.cfg.method, 'ema_schedule', 'cosine')).lower()

        # Deep supervision configuration
        self.ds_enabled = bool(
            getattr(self.cfg.model, 'deep_supervision', False) or
            getattr(self.cfg.method, 'deep_supervision', False)
        )
        ds_cfg = getattr(self.cfg.method, 'deep_supervision', {})
        if isinstance(ds_cfg, dict) or hasattr(ds_cfg, 'weights'):
            self.ds_weights = list(getattr(ds_cfg, 'weights', [0.533, 0.267, 0.133, 0.067]))
            self.ds_consistency = bool(getattr(ds_cfg, 'consistency_multiscale', True))
        else:
            self.ds_weights = [0.533, 0.267, 0.133, 0.067]
            self.ds_consistency = True

        # Background vs Foregrounds weight balance for consistency
        # Class 0: Background (98-99% volume) -> downweighted to 0.05
        # Class 1..5: Hemorrhage subtypes -> full weight 1.0
        if not self.is_binary and self.num_classes == 6:
            bg_weight = float(getattr(self.cfg.method, 'bg_consistency_weight', 0.05))
            self.class_weights = torch.tensor([bg_weight, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.device).view(1, -1, 1, 1, 1)
        else:
            self.class_weights = None

        # Build Teacher model (cloned from raw student backbone, detached from autograd)
        raw_net = self.get_raw_model()
        self.ema_model = copy.deepcopy(raw_net).to(self.device)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_model.eval()

        # Loss function for supervised branch (exclude background from Dice for multiclass)
        self.dice_loss = mask_DiceLoss(nclass=self.num_classes, include_background=self.is_binary)

    def get_raw_model(self) -> nn.Module:
        """Safely unwrap DistributedDataParallel or DataParallel wrapper."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def get_current_consistency_weight(self, iter_num: int, epoch: float) -> float:
        """
        Sigmoid rampup curve + smooth cosine rampdown schedule at late training stage.
        - Rampup: 0 -> consistency over consistency_rampup iters.
        - Plateau: consistency weight maintained.
        - Rampdown: from consistency_rampdown_start to max_iterations,
                    smoothly decays consistency to protect rare classes (IVH) from pseudo-label drift.
        """
        if self.consistency_rampup <= 0:
            base_w = self.consistency
        elif self.consistency_rampup > 100:
            base_w = self.consistency * sigmoid_rampup(iter_num, self.consistency_rampup)
        else:
            base_w = self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

        max_iterations = int(getattr(self.cfg.method, 'max_iteration', 12000))
        if self.consistency_rampdown_start > 0 and iter_num > self.consistency_rampdown_start:
            progress = (iter_num - self.consistency_rampdown_start) / max(max_iterations - self.consistency_rampdown_start, 1)
            progress = min(max(progress, 0.0), 1.0)
            decay_factor = self.consistency_min_factor + (1.0 - self.consistency_min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return base_w * decay_factor

        return base_w

    def compute_soft_dice_consistency(
        self,
        student_prob: torch.Tensor,
        teacher_prob: torch.Tensor,
        conf_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Continuous Probability-to-Probability Soft Dice Consistency on Foregrounds (classes 1..5).
        Completely ignores Background (class 0) to guarantee zero confirmation bias towards background.
        Optionally weights by teacher confidence mask to avoid hallucinated predictions.
        """
        fg_s = student_prob[:, 1:] if self.num_classes > 1 and not self.is_binary else student_prob
        fg_t = teacher_prob[:, 1:] if self.num_classes > 1 and not self.is_binary else teacher_prob

        if conf_mask is not None:
            fg_s = fg_s * conf_mask
            fg_t = fg_t * conf_mask

        # Spatial sum over 3D volume (D, H, W)
        dims = (2, 3, 4)
        intersection = torch.sum(fg_s * fg_t, dim=dims)
        cardinality = torch.sum(fg_s ** 2 + fg_t ** 2, dim=dims)

        smooth = 1e-5
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
        # Average over foreground classes and batch
        return 1.0 - torch.mean(dice_score)

    def compute_consistency_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss with class balancing and foreground-focused metrics.
        Supports:
        - 'hybrid': 50% Soft Dice on foreground + 50% Class-balanced Channel MSE.
        - 'dice': Continuous Soft Dice on foreground lesions.
        - 'weighted_mse': Class-balanced channel MSE (downweights background).
        """
        student_prob = F.softmax(student_logits, dim=1)
        teacher_prob = F.softmax(teacher_logits, dim=1)

        # Confidence thresholding (if enabled > 0.0)
        # Note: In CT 3D scans, background is 98-99% volume. Thresholding must never filter out delicate foreground lesions.
        if self.confidence_thresh > 0.0:
            if self.num_classes > 1 and not self.is_binary:
                fg_max_prob, _ = torch.max(teacher_prob[:, 1:], dim=1, keepdim=True)
                bg_prob = teacher_prob[:, :1]
                conf_mask = ((fg_max_prob >= min(self.confidence_thresh, 0.4)) | (bg_prob >= self.confidence_thresh)).float()
            else:
                max_prob, _ = torch.max(teacher_prob, dim=1, keepdim=True)
                conf_mask = (max_prob >= self.confidence_thresh).float()
        else:
            conf_mask = None

        # 1. Pure Continuous Soft Dice on Foreground Classes
        if self.consistency_type == 'dice':
            return self.compute_soft_dice_consistency(student_prob, teacher_prob, conf_mask=conf_mask)

        # 2. Hybrid Consistency: Soft Dice + Class-balanced Channel MSE
        elif self.consistency_type == 'hybrid':
            loss_dice = self.compute_soft_dice_consistency(student_prob, teacher_prob, conf_mask=conf_mask)
            sq_diff = (student_prob - teacher_prob) ** 2
            if conf_mask is not None:
                sq_diff = sq_diff * conf_mask
                mask_sum = conf_mask.sum(dim=(0, 2, 3, 4)) + 1e-6
                mse_per_class = sq_diff.sum(dim=(0, 2, 3, 4)) / mask_sum
            else:
                mse_per_class = sq_diff.mean(dim=(0, 2, 3, 4))

            if self.class_weights is not None:
                w = self.class_weights.view(-1)
                loss_mse = (mse_per_class * w).sum() / (w.sum() + 1e-6)
            else:
                loss_mse = mse_per_class.mean()
            return 0.5 * loss_mse + 0.5 * loss_dice

        # 3. Class-balanced Channel MSE
        elif self.consistency_type in ('weighted_mse', 'mse', 'class_balanced_mse'):
            sq_diff = (student_prob - teacher_prob) ** 2

            if conf_mask is not None:
                sq_diff = sq_diff * conf_mask
                mask_sum = conf_mask.sum(dim=(0, 2, 3, 4)) + 1e-6
                mse_per_class = sq_diff.sum(dim=(0, 2, 3, 4)) / mask_sum
            else:
                mse_per_class = sq_diff.mean(dim=(0, 2, 3, 4))

            if self.class_weights is not None:
                w = self.class_weights.view(-1)
                loss_cons = (mse_per_class * w).sum() / (w.sum() + 1e-6)
            else:
                loss_cons = mse_per_class.mean()
            return loss_cons

        elif self.consistency_type == 'kl':
            # KL divergence: Teacher -> Student
            if conf_mask is not None:
                kl = F.kl_div(F.log_softmax(student_logits, dim=1), teacher_prob, reduction='none')
                return (kl * conf_mask).sum() / (conf_mask.sum() + 1e-6)
            return F.kl_div(F.log_softmax(student_logits, dim=1), teacher_prob, reduction='batchmean')

        else:
            sq_diff = (student_prob - teacher_prob) ** 2
            if conf_mask is not None:
                return (sq_diff * conf_mask).sum() / (conf_mask.sum() + 1e-6)
            return sq_diff.mean()

    def train_step(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        volume_batch = sampled_batch['image'].to(self.device)
        label_batch = sampled_batch['label'].to(self.device)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # 1. Teacher forward on clean, unperturbed volume with AMP (FP16 Tensor Cores)
            with torch.no_grad():
                if self.ds_enabled and self.ds_consistency and getattr(self.ema_model, 'deep_supervision', False):
                    teacher_out = self.ema_model(volume_batch, return_all_scales=True)
                else:
                    teacher_out = self.ema_model(volume_batch)
                teacher_logits, teacher_aux = extract_multiscale_outputs(teacher_out)

            # 2. Asymmetric student perturbation (pure GPU tensors, zero CPU-GPU synchronization)
            if self.perturb_student:
                noise = torch.randn_like(volume_batch) * self.student_noise_std
                scale = 1.0 + (torch.rand((volume_batch.shape[0], 1, 1, 1, 1), device=self.device) * 2.0 - 1.0) * self.student_intensity_scale
                student_input = volume_batch * scale + noise
            else:
                student_input = volume_batch

            # 3. Forward student on perturbed input
            student_out = self.model(student_input)
            student_logits, student_aux = extract_multiscale_outputs(student_out)

            # Supervised loss on labeled samples (0..labeled_bs)
            lab_logits = student_logits[:self.labeled_bs]
            lab_targets = label_batch[:self.labeled_bs]
            loss_ce = F.cross_entropy(lab_logits, lab_targets)
            loss_dice = self.dice_loss(lab_logits, lab_targets)
            loss_sup_main = (loss_ce + loss_dice) / 2.0

            # Multi-scale deep supervision on labeled samples
            loss_ds_accum = 0.0
            if self.ds_enabled and len(student_aux) > 0:
                all_student_lab = [lab_logits] + [aux[:self.labeled_bs] for aux in student_aux]
                num_scales = len(all_student_lab)
                if len(self.ds_weights) >= num_scales:
                    norm_w = [w / sum(self.ds_weights[:num_scales]) for w in self.ds_weights[:num_scales]]
                else:
                    norm_w = compute_ds_weights(num_scales)

                loss_supervised = norm_w[0] * loss_sup_main
                target_shape = lab_targets.shape[1:]  # (D, H, W)
                for idx in range(1, num_scales):
                    aux_lab = all_student_lab[idx]
                    if aux_lab.shape[2:] != target_shape:
                        aux_lab = F.interpolate(aux_lab, size=target_shape, mode='trilinear', align_corners=False)
                    l_ce_aux = F.cross_entropy(aux_lab, lab_targets)
                    l_dice_aux = self.dice_loss(aux_lab, lab_targets)
                    l_stage = (l_ce_aux + l_dice_aux) / 2.0
                    loss_supervised = loss_supervised + norm_w[idx] * l_stage
                    loss_ds_accum = loss_ds_accum + norm_w[idx] * float(l_stage.item())
            else:
                loss_supervised = loss_sup_main

            # Consistency loss across full batch (labeled + unlabeled)
            loss_cons_main = self.compute_consistency_loss(student_logits, teacher_logits)

            if self.ds_enabled and self.ds_consistency and len(student_aux) > 0 and len(teacher_aux) == len(student_aux):
                all_student = [student_logits] + student_aux
                all_teacher = [teacher_logits] + teacher_aux
                num_scales = len(all_student)
                if len(self.ds_weights) >= num_scales:
                    norm_w = [w / sum(self.ds_weights[:num_scales]) for w in self.ds_weights[:num_scales]]
                else:
                    norm_w = compute_ds_weights(num_scales)

                loss_consistency = norm_w[0] * loss_cons_main
                for idx in range(1, num_scales):
                    l_stage_cons = self.compute_consistency_loss(all_student[idx], all_teacher[idx])
                    loss_consistency = loss_consistency + norm_w[idx] * l_stage_cons
            else:
                loss_consistency = loss_cons_main

            # Dynamic rampup + rampdown weight
            cur_epoch = iter_num / max(len(self.trainloader), 1)
            consistency_weight = self.get_current_consistency_weight(iter_num, cur_epoch)

            total_loss = loss_supervised + consistency_weight * loss_consistency

        # EMA update to Teacher with cosine decay schedule
        max_iter = int(getattr(self.cfg.method, 'max_iteration', 12000))
        raw_net = self.get_raw_model()
        update_ema_variables(raw_net, self.ema_model, self.ema_decay, global_step=iter_num, max_steps=max_iter, schedule=self.ema_schedule)

        loss_dict = {
            "total": float(total_loss.item()),
            "supervised": float(loss_supervised.item()),
            "consistency": float(loss_consistency.item()),
            "weight": float(consistency_weight)
        }
        if self.ds_enabled and len(student_aux) > 0:
            loss_dict["supervised_main"] = float(loss_sup_main.item())
            loss_dict["deep_supervision"] = float(loss_ds_accum)

        return total_loss, loss_dict

    def save_checkpoint(
        self,
        path: str,
        iter_num: int,
        best_dice: float,
        phase: str = 'train',
        artifact_name: Optional[str] = None,
        artifact_type: str = "checkpoint",
        aliases: Optional[List[str]] = None
    ):
        """Override to save both student weights and teacher (EMA) weights."""
        if not is_main_process():
            return

        raw_net = self.get_raw_model()
        state = {
            'net': raw_net.state_dict(),
            'ema_net': self.ema_model.state_dict(),
            'opt': self.optimizer.state_dict() if self.optimizer is not None else None,
            'iter_num': iter_num,
            'best_dice': best_dice,
            'phase': phase,
            'config': self.cfg.to_dict()
        }
        torch.save(state, str(path))

        if artifact_name and self.wandb_manager.enabled and self.wandb_manager.save_artifacts:
            self.wandb_manager.log_artifact(
                file_path=str(path),
                name=artifact_name,
                artifact_type=artifact_type,
                description=f"Mean Teacher checkpoint at iter {iter_num} with best_mDice: {best_dice:.4f}",
                aliases=aliases or ["latest"],
                metadata={'iter_num': iter_num, 'best_dice': best_dice, 'phase': phase}
            )

    def load_checkpoint(self, path: str) -> Tuple[int, float]:
        """Override to load student and teacher (EMA) weights if available."""
        state = torch.load(str(path), map_location=self.device)
        net_dict = state.get('net', state)
        raw_net = self.get_raw_model()
        clean_dict = {k[7:] if k.startswith('module.') else k: v for k, v in net_dict.items()}
        raw_net.load_state_dict(clean_dict, strict=False)

        # Restore Teacher weights
        if 'ema_net' in state and state['ema_net'] is not None:
            clean_ema = {k[7:] if k.startswith('module.') else k: v for k, v in state['ema_net'].items()}
            self.ema_model.load_state_dict(clean_ema, strict=False)
            logging.info("[Checkpoint] Successfully loaded Teacher (EMA) model weights.")
        else:
            # Fallback: clone from loaded student
            self.ema_model.load_state_dict(clean_dict, strict=False)
            logging.info("[Checkpoint] No 'ema_net' found, synchronized Teacher from Student weights.")

        if self.optimizer is not None and 'opt' in state and state['opt'] is not None:
            try:
                self.optimizer.load_state_dict(state['opt'])
            except Exception as e:
                logging.warning(f"[Checkpoint] Could not restore optimizer state: {e}")

        iter_num = state.get('iter_num', 0)
        best_dice = state.get('best_dice', 0.0)
        return iter_num, best_dice

    def validate(self, return_visuals: bool = False) -> Union[Tuple[float, List[float]], Tuple[float, List[float], List[Dict[str, Any]]]]:
        """
        Evaluate sliding window on Teacher (EMA) model as primary metric.
        Teacher typically achieves significantly higher and smoother mDice than Student.
        """
        stride_xy = int(getattr(self.cfg.logging, 'val_stride_xy', 32))
        stride_z = int(getattr(self.cfg.logging, 'val_stride_z', 8))
        self.ema_model.eval()
        num_vis = int(getattr(self.cfg.logging, 'num_vis_samples', 4)) if (return_visuals and self.rank == 0) else 0

        return evaluate_dataset_sliding_window(
            model=self.ema_model,
            val_dataset=self.db_val,
            num_classes=self.num_classes,
            is_binary=self.is_binary,
            patch_size=self.patch_size,
            stride_z=stride_z,
            stride_xy=stride_xy,
            use_amp=True,
            device=self.device,
            log_progress=(self.rank == 0),
            return_visual_samples=num_vis,
            rank=self.rank,
            world_size=self.world_size
        )
