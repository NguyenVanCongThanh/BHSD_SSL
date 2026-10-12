"""
Mean Teacher Method for Semi-Supervised 3D Medical Segmentation.
Reference: Tarvainen & Valpola, "Mean teachers are better role models", NeurIPS 2017.
"""

from typing import Dict, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from methods.base_method import BaseMethod
from utils.losses import mask_DiceLoss
from utils.ramps import sigmoid_rampup


def update_ema_variables(model: nn.Module, ema_model: nn.Module, alpha: float):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


@METHODS.register("mean_teacher")
class MeanTeacherMethod(BaseMethod):
    """
    Mean Teacher SSL Framework with EMA consistency regularization and sigmoid rampup.
    """
    def __init__(self, model: nn.Module, cfg):
        super(MeanTeacherMethod, self).__init__(model, cfg)

        self.labeled_bs = int(getattr(self.cfg.data, 'labeled_bs', 2))
        self.consistency = float(getattr(self.cfg.method, 'consistency', 1.0))
        self.consistency_rampup = float(getattr(self.cfg.method, 'consistency_rampup', 40.0))
        self.ema_decay = float(getattr(self.cfg.method, 'ema_decay', 0.99))

        # Create EMA Teacher model
        raw_net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.ema_model = copy.deepcopy(raw_net).to(self.device)
        for param in self.ema_model.parameters():
            param.detach_()

        self.dice_loss = mask_DiceLoss(nclass=self.num_classes)

    def get_current_consistency_weight(self, epoch: float) -> float:
        return self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

    def train_step(self, sampled_batch: Dict, mim_mask: torch.Tensor, iter_num: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        volume_batch = sampled_batch['image'].to(self.device)
        label_batch = sampled_batch['label'].to(self.device)

        # Forward student
        student_out = self.model(volume_batch)
        if isinstance(student_out, (tuple, list)):
            student_logits = student_out[1] if len(student_out) >= 2 and isinstance(student_out[1], torch.Tensor) and student_out[1].ndim >= 4 else student_out[0]
        else:
            student_logits = student_out

        # Forward teacher with slight Gaussian noise perturbation
        with torch.no_grad():
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            teacher_input = volume_batch + noise
            teacher_out = self.ema_model(teacher_input)
            if isinstance(teacher_out, (tuple, list)):
                teacher_logits = teacher_out[1] if len(teacher_out) >= 2 and isinstance(teacher_out[1], torch.Tensor) and teacher_out[1].ndim >= 4 else teacher_out[0]
            else:
                teacher_logits = teacher_out

        # Supervised loss on labeled partition
        lab_logits = student_logits[:self.labeled_bs]
        lab_targets = label_batch[:self.labeled_bs]
        loss_ce = F.cross_entropy(lab_logits, lab_targets)
        loss_dice = self.dice_loss(lab_logits, lab_targets)
        loss_supervised = (loss_ce + loss_dice) / 2.0

        # Consistency loss across full batch (MSE between student & teacher probabilities)
        student_prob = F.softmax(student_logits, dim=1)
        teacher_prob = F.softmax(teacher_logits, dim=1)
        loss_consistency = F.mse_loss(student_prob, teacher_prob)

        cur_epoch = iter_num / max(len(self.trainloader), 1)
        consistency_weight = self.get_current_consistency_weight(cur_epoch)

        total_loss = loss_supervised + consistency_weight * loss_consistency

        # Update teacher weights with EMA
        raw_net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        update_ema_variables(raw_net, self.ema_model, self.ema_decay)

        loss_dict = {
            "total": float(total_loss.item()),
            "supervised": float(loss_supervised.item()),
            "consistency": float(loss_consistency.item()),
            "weight": float(consistency_weight)
        }
        return total_loss, loss_dict
