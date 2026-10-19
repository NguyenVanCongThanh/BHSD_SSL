"""
Cross Pseudo Supervision (CPS) Method for Semi-Supervised 3D Medical Segmentation.
Reference: Chen et al., "Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision", CVPR 2021.
"""

from typing import Dict, Tuple, Optional
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from core.registry import METHODS
from methods.base_method import BaseMethod
from losses import MaskDiceLoss, extract_multiscale_outputs, compute_ds_weights
from utils.ramps import sigmoid_rampup


@METHODS.register("cps")
class CPSMethod(BaseMethod):
    """
    Cross Pseudo Supervision: Two networks with different initializations cross-supervise each other on unlabeled data.
    """
    def __init__(self, model: nn.Module, cfg):
        super(CPSMethod, self).__init__(model, cfg)

        self.labeled_bs = int(getattr(self.cfg.data, 'labeled_bs', 2))
        self.cps_weight = float(getattr(self.cfg.method, 'cps_weight', 1.5))
        self.consistency_rampup = float(getattr(self.cfg.method, 'consistency_rampup', 40.0))

        # Build second network model2 with independent weights
        raw_net1 = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.model2 = copy.deepcopy(raw_net1).to(self.device)

        # Re-initialize weights of model2
        for m in self.model2.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        base_lr = float(getattr(self.cfg.method, 'base_lr', 0.01))
        momentum = float(getattr(self.cfg.method, 'momentum', 0.9))
        weight_decay = float(getattr(self.cfg.method, 'weight_decay', 0.0001))
        self.optimizer2 = optim.SGD(
            self.model2.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay
        )

        self.dice_loss = MaskDiceLoss(nclass=self.num_classes)

        # Deep supervision configuration
        self.ds_enabled = bool(
            getattr(self.cfg.model, 'deep_supervision', False) or
            getattr(self.cfg.method, 'deep_supervision', False)
        )
        ds_cfg = getattr(self.cfg.method, 'deep_supervision', {})
        if isinstance(ds_cfg, dict) or hasattr(ds_cfg, 'weights'):
            self.ds_weights = list(getattr(ds_cfg, 'weights', [0.533, 0.267, 0.133, 0.067]))
        else:
            self.ds_weights = [0.533, 0.267, 0.133, 0.067]

    def _compute_multi_scale_sup_loss(self, main_lab: torch.Tensor, aux_labs: list, targets: torch.Tensor) -> Tuple[torch.Tensor, float]:
        l_ce = F.cross_entropy(main_lab, targets)
        l_dice = self.dice_loss(main_lab, targets)
        l_main = (l_ce + l_dice) / 2.0

        if not self.ds_enabled or len(aux_labs) == 0:
            return l_main, 0.0

        all_labs = [main_lab] + aux_labs
        num_scales = len(all_labs)
        if len(self.ds_weights) >= num_scales:
            norm_w = [w / sum(self.ds_weights[:num_scales]) for w in self.ds_weights[:num_scales]]
        else:
            norm_w = compute_ds_weights(num_scales)

        total_sup = norm_w[0] * l_main
        ds_accum = 0.0
        target_shape = targets.shape[1:]
        for idx in range(1, num_scales):
            aux = all_labs[idx]
            if aux.shape[2:] != target_shape:
                aux = F.interpolate(aux, size=target_shape, mode='trilinear', align_corners=False)
            stage_loss = (F.cross_entropy(aux, targets) + self.dice_loss(aux, targets)) / 2.0
            total_sup = total_sup + norm_w[idx] * stage_loss
            ds_accum = ds_accum + norm_w[idx] * float(stage_loss.item())

        return total_sup, ds_accum

    def train_step(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        volume_batch = sampled_batch['image'].to(self.device)
        label_batch = sampled_batch['label'].to(self.device)

        # Forward network 1 and network 2
        logits1, aux1 = extract_multiscale_outputs(self.model(volume_batch))
        logits2, aux2 = extract_multiscale_outputs(self.model2(volume_batch))

        # 1. Supervised loss on labeled partition with optional deep supervision
        lab_logits1 = logits1[:self.labeled_bs]
        lab_logits2 = logits2[:self.labeled_bs]
        lab_targets = label_batch[:self.labeled_bs]

        aux1_lab = [a[:self.labeled_bs] for a in aux1]
        aux2_lab = [a[:self.labeled_bs] for a in aux2]

        loss_sup1, ds_accum1 = self._compute_multi_scale_sup_loss(lab_logits1, aux1_lab, lab_targets)
        loss_sup2, ds_accum2 = self._compute_multi_scale_sup_loss(lab_logits2, aux2_lab, lab_targets)

        # 2. Cross pseudo-labels from argmax predictions
        with torch.no_grad():
            prob1 = F.softmax(logits1, dim=1)
            prob2 = F.softmax(logits2, dim=1)
            pl1 = torch.argmax(prob1, dim=1).long()
            pl2 = torch.argmax(prob2, dim=1).long()

        # 3. Cross supervision loss
        loss_cps1 = F.cross_entropy(logits1, pl2)
        loss_cps2 = F.cross_entropy(logits2, pl1)

        cur_epoch = iter_num / max(len(self.trainloader), 1)
        cur_weight = self.cps_weight * sigmoid_rampup(cur_epoch, self.consistency_rampup)

        total_loss1 = loss_sup1 + cur_weight * loss_cps1
        total_loss2 = loss_sup2 + cur_weight * loss_cps2
        total_loss = total_loss1 + total_loss2

        # Step optimizer 2
        self.optimizer2.zero_grad()
        total_loss2.backward(retain_graph=True)
        self.optimizer2.step()

        loss_dict = {
            "total": float(total_loss.item()),
            "sup1": float(loss_sup1.item()),
            "sup2": float(loss_sup2.item()),
            "cps1": float(loss_cps1.item()),
            "cps2": float(loss_cps2.item()),
            "weight": float(cur_weight)
        }
        if self.ds_enabled and len(aux1) > 0:
            loss_dict["ds1"] = float(ds_accum1)
            loss_dict["ds2"] = float(ds_accum2)

        return total_loss1, loss_dict
