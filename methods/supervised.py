"""
Supervised Baseline Method (Lower Bound).
Trains network purely on available labeled samples using combined Cross-Entropy + Dice Loss.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from methods.base_method import BaseMethod
from methods.ssl_utils import extract_logits
from losses import MaskDiceLoss


@METHODS.register("supervised")
class SupervisedMethod(BaseMethod):
    """
    Supervised Baseline Engine.
    """
    def __init__(self, model: nn.Module, cfg):
        super(SupervisedMethod, self).__init__(model, cfg)
        self.dice_loss = MaskDiceLoss(nclass=self.num_classes)

    def train_step(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        volume_batch = sampled_batch['image'][:self.labeled_bs].to(self.device)
        label_batch = sampled_batch['label'][:self.labeled_bs].to(self.device)

        logits = extract_logits(self.model(volume_batch))

        loss_ce = F.cross_entropy(logits, label_batch)
        loss_dice = self.dice_loss(logits, label_batch)
        total_loss = (loss_ce + loss_dice) / 2.0

        loss_dict = {
            "total": float(total_loss.item()),
            "ce": float(loss_ce.item()),
            "dice": float(loss_dice.item())
        }
        return total_loss, loss_dict
