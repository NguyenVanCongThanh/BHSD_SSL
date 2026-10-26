"""
Bidirectional Copy-Paste (BCP) Method for Semi-Supervised 3D Medical Segmentation.
Reference: Bai et al., "Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation", CVPR 2023.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from methods.base_method import BaseMethod
from methods.ssl_utils import extract_logits, split_ssl_batch
from losses import MaskDiceLoss
from data.masking import context_mask


@METHODS.register("bcp")
class BCPMethod(BaseMethod):
    """
    Bidirectional Copy-Paste (BCP) framework:
    Paste labeled foreground into unlabeled background and vice versa.
    """
    def __init__(self, model: nn.Module, cfg):
        super(BCPMethod, self).__init__(model, cfg)

        self.mask_ratio = float(getattr(self.cfg.method, 'mask_ratio', 2 / 3))
        self.u_weight = float(getattr(self.cfg.method, 'u_weight', 0.5))
        self.dice_loss = MaskDiceLoss(nclass=self.num_classes)

    def train_step(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = split_ssl_batch(sampled_batch, self.labeled_bs, device=self.device)
        img_a, img_b = batch.img_a, batch.img_b
        lab_a, lab_b = batch.lab_a, batch.lab_b
        unimg = batch.unlabeled_volume
        unimg_a, unimg_b = batch.unimg_a, batch.unimg_b
        sub_bs = batch.sub_bs

        # Pseudo-labeling unlabeled images
        with torch.no_grad():
            un_logits = extract_logits(self.model(unimg))
            un_prob = F.softmax(un_logits, dim=1)
            plab = torch.argmax(un_prob, dim=1).long()
            plab_a, plab_b = plab[:sub_bs], plab[sub_bs:]

            img_mask, loss_mask = context_mask(img_a, self.mask_ratio)

        # Mix 1: labeled + unlabeled
        mix_img_1 = img_a * img_mask + unimg_a * (1 - img_mask)
        mix_lab_1 = lab_a * img_mask + plab_a * (1 - img_mask)

        # Mix 2: unlabeled + labeled
        mix_img_2 = unimg_b * img_mask + img_b * (1 - img_mask)
        mix_lab_2 = plab_b * img_mask + lab_b * (1 - img_mask)

        mix_imgs = torch.cat((mix_img_1, mix_img_2), dim=0)
        mix_labs = torch.cat((mix_lab_1, mix_lab_2), dim=0)

        mix_logits = extract_logits(self.model(mix_imgs))

        loss_ce = F.cross_entropy(mix_logits, mix_labs)
        loss_dice = self.dice_loss(mix_logits, mix_labs)
        total_loss = (loss_ce + loss_dice) / 2.0

        loss_dict = {
            "total": float(total_loss.item()),
            "ce": float(loss_ce.item()),
            "dice": float(loss_dice.item())
        }
        return total_loss, loss_dict
