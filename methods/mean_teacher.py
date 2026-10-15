"""
Mean Teacher Method for Semi-Supervised 3D Medical Segmentation (BHSD).
Reference: Tarvainen & Valpola, "Mean teachers are better role models:
Weight-averaged consistency targets improve semi-supervised deep learning results", NeurIPS 2017.

Adapted & Optimized for 3D Brain Hemorrhage Segmentation (Extreme class imbalance, 6 classes, DDP compatible).
"""

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
from evaluation.sliding_window import evaluate_dataset_sliding_window


def update_ema_variables(model: nn.Module, ema_model: nn.Module, alpha: float, global_step: int = -1):
    """
    Update EMA teacher parameters from student model parameters.
    Supports dynamic alpha rampup early on if global_step is provided.
    """
    if global_step >= 0:
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
    - Full DDP and sliding-window validation on both Student & Teacher models.
    """
    def __init__(self, model: nn.Module, cfg):
        super(MeanTeacherMethod, self).__init__(model, cfg)

        self.labeled_bs = int(getattr(self.cfg.data, 'labeled_bs', 2))
        self.consistency = float(getattr(self.cfg.method, 'consistency', 1.0))
        self.consistency_rampup = float(getattr(self.cfg.method, 'consistency_rampup', 40.0))
        self.ema_decay = float(getattr(self.cfg.method, 'ema_decay', 0.99))
        self.consistency_type = str(getattr(self.cfg.method, 'consistency_type', 'weighted_mse')).lower()
        self.confidence_thresh = float(getattr(self.cfg.method, 'confidence_thresh', 0.0))

        # Background vs Foregrounds weight balance for consistency
        # Class 0: Background (98-99% volume) -> downweighted to 0.1
        # Class 1..5: Hemorrhage subtypes -> full weight 1.0
        if not self.is_binary and self.num_classes == 6:
            bg_weight = float(getattr(self.cfg.method, 'bg_consistency_weight', 0.1))
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

    def get_current_consistency_weight(self, epoch: float) -> float:
        """Sigmoid rampup curve across epochs."""
        if self.consistency_rampup <= 0:
            return self.consistency
        return self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

    def compute_consistency_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss with confidence thresholding and class balancing.
        """
        student_prob = F.softmax(student_logits, dim=1)
        teacher_prob = F.softmax(teacher_logits, dim=1)

        # 1. Confidence thresholding (filter out noisy teacher predictions)
        if self.confidence_thresh > 0.0:
            max_prob, _ = torch.max(teacher_prob, dim=1, keepdim=True)
            conf_mask = (max_prob >= self.confidence_thresh).float()
        else:
            conf_mask = None

        # 2. Consistency loss calculation
        if self.consistency_type == 'weighted_mse':
            sq_diff = (student_prob - teacher_prob) ** 2
            if self.class_weights is not None:
                sq_diff = sq_diff * self.class_weights

            if conf_mask is not None:
                loss_cons = (sq_diff * conf_mask).sum() / (conf_mask.sum() * student_prob.shape[1] + 1e-6)
            else:
                loss_cons = sq_diff.mean()

        elif self.consistency_type == 'dice':
            # Soft dice consistency
            pseudo_target = teacher_prob.argmax(dim=1)
            loss_cons = self.dice_loss(student_logits, pseudo_target, mask=conf_mask)

        elif self.consistency_type == 'kl':
            # KL divergence: Teacher -> Student
            loss_cons = F.kl_div(F.log_softmax(student_logits, dim=1), teacher_prob, reduction='batchmean')

        else:
            # Default unweighted MSE
            sq_diff = (student_prob - teacher_prob) ** 2
            if conf_mask is not None:
                loss_cons = (sq_diff * conf_mask).sum() / (conf_mask.sum() * student_prob.shape[1] + 1e-6)
            else:
                loss_cons = sq_diff.mean()

        return loss_cons

    def train_step(self, sampled_batch: Dict, mim_mask: torch.Tensor, iter_num: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        volume_batch = sampled_batch['image'].to(self.device)
        label_batch = sampled_batch['label'].to(self.device)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Forward student
            student_out = self.model(volume_batch)
            if isinstance(student_out, (tuple, list)):
                student_logits = student_out[1] if len(student_out) >= 2 and isinstance(student_out[1], torch.Tensor) and student_out[1].ndim >= 4 else student_out[0]
            else:
                student_logits = student_out

            # Teacher perturbation: gentle Gaussian intensity noise + clipping
            with torch.no_grad():
                noise = torch.clamp(torch.randn_like(volume_batch) * 0.05, -0.1, 0.1)
                teacher_input = volume_batch + noise
                teacher_out = self.ema_model(teacher_input)
                if isinstance(teacher_out, (tuple, list)):
                    teacher_logits = teacher_out[1] if len(teacher_out) >= 2 and isinstance(teacher_out[1], torch.Tensor) and teacher_out[1].ndim >= 4 else teacher_out[0]
                else:
                    teacher_logits = teacher_out

            # Supervised loss on labeled samples (0..labeled_bs)
            lab_logits = student_logits[:self.labeled_bs]
            lab_targets = label_batch[:self.labeled_bs]
            loss_ce = F.cross_entropy(lab_logits, lab_targets)
            loss_dice = self.dice_loss(lab_logits, lab_targets)
            loss_supervised = (loss_ce + loss_dice) / 2.0

            # Consistency loss across full batch (labeled + unlabeled)
            loss_consistency = self.compute_consistency_loss(student_logits, teacher_logits)

            # Dynamic rampup weight
            cur_epoch = iter_num / max(len(self.trainloader), 1)
            consistency_weight = self.get_current_consistency_weight(cur_epoch)

            total_loss = loss_supervised + consistency_weight * loss_consistency

        # EMA update to Teacher
        raw_net = self.get_raw_model()
        update_ema_variables(raw_net, self.ema_model, self.ema_decay, global_step=iter_num)

        loss_dict = {
            "total": float(total_loss.item()),
            "supervised": float(loss_supervised.item()),
            "consistency": float(loss_consistency.item()),
            "weight": float(consistency_weight)
        }
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
        num_vis = int(getattr(self.cfg.logging, 'num_vis_samples', 4)) if return_visuals else 0

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
            log_progress=True,
            return_visual_samples=num_vis
        )
