"""
Mean Teacher Method for Semi-Supervised 3D Medical Segmentation (BHSD).
Reference: Tarvainen & Valpola, "Mean teachers are better role models:
Weight-averaged consistency targets improve semi-supervised deep learning results", NeurIPS 2017.

Adapted & Optimized for 3D Brain Hemorrhage Segmentation (Extreme class imbalance, 6 classes, DDP compatible).
Leverages centralized ConsistencyLoss, EMA utilities, and BaseMethod engine.
"""

from typing import Dict, Tuple, List, Optional, Union, Any
import math
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from methods.base_method import BaseMethod
from methods.ssl_utils import update_ema_variables
from losses import MaskDiceLoss, extract_multiscale_outputs, compute_ds_weights, ConsistencyLoss
from utils.ramps import sigmoid_rampup
from utils.checkpoint import get_raw_model
from evaluation.sliding_window import evaluate_dataset_sliding_window


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

        self.consistency = float(getattr(self.cfg.method, 'consistency', 0.05))
        self.consistency_rampup = float(getattr(self.cfg.method, 'consistency_rampup', 4000.0))
        self.ema_decay = float(getattr(self.cfg.method, 'ema_decay', 0.999))
        self.consistency_type = str(getattr(self.cfg.method, 'consistency_type', 'hybrid')).lower()
        self.confidence_thresh = float(getattr(self.cfg.method, 'confidence_thresh', 0.0))

        # Asymmetric student perturbation
        self.perturb_student = bool(getattr(self.cfg.method, 'perturb_student', True))
        self.student_noise_std = float(getattr(self.cfg.method, 'student_noise_std', 0.02))
        self.student_intensity_scale = float(getattr(self.cfg.method, 'student_intensity_scale', 0.05))

        # Late-stage consistency rampdown
        max_iter = int(getattr(self.cfg.method, 'max_iteration', 12000))
        self.consistency_rampdown_start = int(getattr(self.cfg.method, 'consistency_rampdown_start', int(max_iter * 0.65)))
        self.consistency_min_factor = float(getattr(self.cfg.method, 'consistency_min_factor', 0.25))
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

        # Class balancing: Background downweighted to 0.05, Foregrounds full weight 1.0
        if not self.is_binary and self.num_classes == 6:
            bg_weight = float(getattr(self.cfg.method, 'bg_consistency_weight', 0.05))
            self.class_weights = torch.tensor([bg_weight, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.device).view(1, -1, 1, 1, 1)
        else:
            self.class_weights = None

        # Build Teacher model (cloned from raw student backbone, detached from autograd)
        raw_net = get_raw_model(self.model)
        self.ema_model = copy.deepcopy(raw_net).to(self.device)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_model.eval()

        # Modular Loss functions
        self.dice_loss = MaskDiceLoss(nclass=self.num_classes, include_background=self.is_binary)
        self.consistency_loss = ConsistencyLoss(
            consistency_type=self.consistency_type,
            num_classes=self.num_classes,
            is_binary=self.is_binary,
            confidence_thresh=self.confidence_thresh,
            class_weights=self.class_weights
        )

    def get_current_consistency_weight(self, iter_num: int, epoch: float) -> float:
        """Sigmoid rampup curve + smooth cosine rampdown schedule at late training stage."""
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

    def train_step(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        volume_batch = sampled_batch['image'].to(self.device)
        label_batch = sampled_batch['label'].to(self.device)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # 1. Teacher forward on clean, unperturbed volume
            with torch.no_grad():
                if self.ds_enabled and self.ds_consistency and getattr(self.ema_model, 'deep_supervision', False):
                    teacher_out = self.ema_model(volume_batch, return_all_scales=True)
                else:
                    teacher_out = self.ema_model(volume_batch)
                teacher_logits, teacher_aux = extract_multiscale_outputs(teacher_out)

            # 2. Asymmetric student perturbation
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
                norm_w = compute_ds_weights(len(all_student_lab))
                loss_supervised = norm_w[0] * loss_sup_main
                target_shape = lab_targets.shape[1:]

                for idx in range(1, len(all_student_lab)):
                    aux_lab = all_student_lab[idx]
                    if aux_lab.shape[2:] != target_shape:
                        aux_lab = F.interpolate(aux_lab, size=target_shape, mode='trilinear', align_corners=False)
                    l_stage = (F.cross_entropy(aux_lab, lab_targets) + self.dice_loss(aux_lab, lab_targets)) / 2.0
                    loss_supervised = loss_supervised + norm_w[idx] * l_stage
                    loss_ds_accum = loss_ds_accum + norm_w[idx] * float(l_stage.item())
            else:
                loss_supervised = loss_sup_main

            # Consistency loss across full batch via centralized ConsistencyLoss
            loss_cons_main = self.consistency_loss(student_logits, teacher_logits)

            if self.ds_enabled and self.ds_consistency and len(student_aux) > 0 and len(teacher_aux) == len(student_aux):
                all_student = [student_logits] + student_aux
                all_teacher = [teacher_logits] + teacher_aux
                norm_w = compute_ds_weights(len(all_student))
                loss_consistency = norm_w[0] * loss_cons_main
                for idx in range(1, len(all_student)):
                    loss_consistency = loss_consistency + norm_w[idx] * self.consistency_loss(all_student[idx], all_teacher[idx])
            else:
                loss_consistency = loss_cons_main

            cur_epoch = iter_num / max(len(self.trainloader), 1)
            consistency_weight = self.get_current_consistency_weight(iter_num, cur_epoch)
            total_loss = loss_supervised + consistency_weight * loss_consistency

        # EMA update to Teacher with cosine decay schedule
        max_iter = int(getattr(self.cfg.method, 'max_iteration', 12000))
        raw_net = get_raw_model(self.model)
        raw_ema = get_raw_model(self.ema_model)
        update_ema_variables(raw_net, raw_ema, self.ema_decay, global_step=iter_num, max_steps=max_iter, schedule=self.ema_schedule)

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
        """Save student weights along with teacher (EMA) weights."""
        raw_ema = get_raw_model(self.ema_model)
        super().save_checkpoint(
            path=path,
            iter_num=iter_num,
            best_dice=best_dice,
            phase=phase,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            aliases=aliases,
            extra_state={'ema_net': raw_ema.state_dict()}
        )

    def load_checkpoint(self, path: str) -> Tuple[int, float]:
        """Load student weights and restore teacher (EMA) weights."""
        iter_num, best_dice = super().load_checkpoint(path, prefer_ema=False)
        from utils.checkpoint import load_checkpoint as _load_ckpt
        try:
            _load_ckpt(path, model=self.ema_model, device=self.device, prefer_ema=True)
            logging.info("[Checkpoint] Successfully loaded Teacher (EMA) model weights.")
        except Exception as e:
            logging.warning(f"[Checkpoint] Fallback synchronizing Teacher from Student: {e}")
            raw_net = get_raw_model(self.model)
            raw_ema = get_raw_model(self.ema_model)
            raw_ema.load_state_dict(raw_net.state_dict(), strict=False)
        return iter_num, best_dice

    def validate(self, return_visuals: bool = False) -> Union[Tuple[float, List[float]], Tuple[float, List[float], List[Dict[str, Any]]]]:
        """Evaluate sliding window on Teacher (EMA) model as primary metric."""
        stride_xy = int(getattr(self.cfg.logging, 'val_stride_xy', 32))
        stride_z = int(getattr(self.cfg.logging, 'val_stride_z', 8))
        self.ema_model.eval()
        num_vis = int(getattr(self.cfg.logging, 'num_vis_samples', 4)) if (return_visuals and self.rank == 0) else 0

        return evaluate_dataset_sliding_window(
            model=get_raw_model(self.ema_model),
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
