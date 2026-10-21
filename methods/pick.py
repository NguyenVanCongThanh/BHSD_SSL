"""
PICK (Pseudo-label Informed CutMix / Context Masking Knowledge) Algorithm for BHSD.
Unified implementation for both Binary (Hemorrhage) and Multi-class (6 classes).
Seamlessly manages both Pre-training and Self-training phases using the centralized BaseMethod engine.
"""

from typing import Dict, Any, Optional, Tuple
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from core.distributed import is_main_process
from methods.base_method import BaseMethod
from methods.ssl_utils import split_ssl_batch, extract_logits
from losses import MaskDiceLoss
from data.masking import context_mask


@METHODS.register("pick")
class PICKMethod(BaseMethod):
    """
    PICK Semi-Supervised Learning Framework:
    - Pre-training phase: CutMix on labeled data + lesion-informed MIM reconstruction + Aux branch.
    - Self-training phase: Bidirectional CutMix (labeled <-> unlabeled) + Unlabeled MIM + Aux branch.
    """
    def __init__(self, model: nn.Module, cfg):
        super(PICKMethod, self).__init__(model, cfg)

        self.pre_max_iteration = int(getattr(self.cfg.method, 'pre_max_iteration', 4000))
        self.self_max_iteration = int(getattr(self.cfg.method, 'self_max_iteration', 12000))
        self.lambda_mim = float(getattr(self.cfg.method, 'lambda_mim', getattr(self.cfg.method, 'lambda_', 0.2)))
        self.mask_ratio = float(getattr(self.cfg.method, 'mask_ratio', 2 / 3))
        self.phase = str(getattr(self.cfg.experiment, 'phase', 'all')).lower()
        self.pretrain_checkpoint = getattr(self.cfg.experiment, 'pretrain_checkpoint', '')

        self.labeled_bs = int(getattr(self.cfg.data, 'labeled_bs', 2))
        self.val_interval = int(getattr(self.cfg.logging, 'val_interval', 500))
        self.save_interval = int(getattr(self.cfg.logging, 'save_interval', 50))

        # Snapshot paths for pre-train and self-train
        self.pre_snapshot_path = os.path.join(self.snapshot_path, "pre_train")
        self.self_snapshot_path = os.path.join(self.snapshot_path, "self_train")
        os.makedirs(self.pre_snapshot_path, exist_ok=True)
        os.makedirs(self.self_snapshot_path, exist_ok=True)

        # Loss function (exclude background from Dice for multi-class)
        self.dice_loss = MaskDiceLoss(nclass=self.num_classes, include_background=False)

    def _ensure_mim_mask(self, batch_data: Any, bs: int) -> torch.Tensor:
        """Extract or generate MIM mask for volumetric input."""
        if isinstance(batch_data, (tuple, list)) and len(batch_data) > 1 and batch_data[1] is not None:
            return batch_data[1]
        if not hasattr(self, '_mask_generator'):
            from data.masking import MaskGeneratorBHSD
            self._mask_generator = MaskGeneratorBHSD(input_size=self.patch_size)
        return torch.from_numpy(np.stack([self._mask_generator() for _ in range(bs)]))

    def pretrain_step(
        self,
        sampled_batch: Dict[str, Any],
        iter_num: int,
        mim_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Optimization step for Giai đoạn 1: Pre-training (Labeled CutMix + Lesion-informed MIM + Aux)."""
        volume_batch = sampled_batch['image'][:self.labeled_bs].to(self.device)
        label_batch = sampled_batch['label'][:self.labeled_bs].to(self.device)

        if mim_mask is None:
            mim_mask = self._ensure_mim_mask(sampled_batch, volume_batch.shape[0])
        mim_mask = mim_mask[:self.labeled_bs].unsqueeze(1).to(device=self.device, dtype=torch.float32)

        img_a, img_b = volume_batch, torch.flip(volume_batch, dims=[0])
        lab_a, lab_b = label_batch, torch.flip(label_batch, dims=[0])

        with torch.no_grad():
            img_mask, _ = context_mask(img_a, self.mask_ratio)

        cutmix_batch = img_a * img_mask + img_b * (1 - img_mask)
        cutmix_label = lab_a * img_mask + lab_b * (1 - img_mask)
        num_cutmix = cutmix_batch.shape[0]

        # Forward main decoder
        _, main_outputs, _ = self.model(torch.cat((cutmix_batch, volume_batch), dim=0))

        main_prob = F.softmax(main_outputs.detach(), dim=1)
        threshold = 0.5 if iter_num > 2000 else 0.2
        fg_prob = main_prob[num_cutmix:, 1:, :, :, :].sum(dim=1, keepdim=True)
        main_ps_lab = (fg_prob > threshold).float()

        mask_region = torch.clamp(mim_mask + main_ps_lab, 0.0, 1.0) if iter_num > 2000 else mim_mask
        mim_batch = volume_batch * (1 - mask_region)
        mim_outputs = self.model(mim_batch, mode='mim')

        re_batch = volume_batch * (1 - mask_region) + mim_outputs.detach() * mask_region
        aux_outputs = self.model(re_batch, mode='aux')

        combined_label = torch.cat((cutmix_label, label_batch), dim=0)
        loss_main_ce = F.cross_entropy(main_outputs, combined_label)
        loss_main_dice = self.dice_loss(main_outputs, combined_label)
        loss_main = (loss_main_ce + loss_main_dice) / 2

        loss_mim_raw = F.l1_loss(volume_batch, mim_outputs, reduction='none')
        loss_mim = self.lambda_mim * (loss_mim_raw * mask_region).sum() / (mask_region.sum() + 1e-5)

        loss_aux_ce = F.cross_entropy(aux_outputs, label_batch)
        loss_aux_dice = self.dice_loss(aux_outputs, label_batch)
        loss_aux = (loss_aux_ce + loss_aux_dice) / 2

        total_loss = loss_main + loss_mim + loss_aux
        loss_dict = {
            'loss_total': float(total_loss.item()),
            'loss_main': float(loss_main.item()),
            'loss_mim': float(loss_mim.item()),
            'loss_aux': float(loss_aux.item())
        }
        return total_loss, loss_dict

    def selftrain_step(
        self,
        sampled_batch: Dict[str, Any],
        iter_num: int,
        mim_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Optimization step for Giai đoạn 2: Self-training (Bidirectional CutMix + Unlabeled MIM + Aux)."""
        batch = split_ssl_batch(sampled_batch, self.labeled_bs, device=self.device)
        img_a, img_b = batch.img_a, batch.img_b
        lab_a, lab_b = batch.lab_a, batch.lab_b
        unimg = batch.unlabeled_volume
        laimg = batch.labeled_volume
        unimg_a, unimg_b = batch.unimg_a, batch.unimg_b
        sub_bs = batch.sub_bs

        raw_net = self.model.module if hasattr(self.model, 'module') else self.model
        with torch.no_grad():
            _, un_main_outputs, _ = raw_net(unimg)
            un_prob = F.softmax(un_main_outputs, dim=1)
            fg_un_prob = un_prob[:, 1:, :, :, :].sum(dim=1, keepdim=True)
            un_ps_lab = (fg_un_prob > 0.5).float()
            img_mask, _ = context_mask(img_a, self.mask_ratio)

        cutmix_batch_f = img_a * img_mask + unimg_a * (1 - img_mask)
        cutmix_batch_b = unimg_b * img_mask + img_b * (1 - img_mask)

        # Forward CutMix through model
        _, main_outputs, _ = self.model(torch.cat((cutmix_batch_f, cutmix_batch_b), dim=0))
        out_f = main_outputs[:sub_bs]
        out_b = main_outputs[sub_bs:]

        # Masked supervised loss on labeled regions
        mask_f = img_mask.float()
        loss_ce_f = (F.cross_entropy(out_f, lab_a, reduction='none') * mask_f).sum() / (mask_f.sum() + 1e-6)
        loss_dice_f = self.dice_loss(out_f, lab_a, mask=mask_f)

        mask_b = (1.0 - img_mask).float()
        loss_ce_b = (F.cross_entropy(out_b, lab_b, reduction='none') * mask_b).sum() / (mask_b.sum() + 1e-6)
        loss_dice_b = self.dice_loss(out_b, lab_b, mask=mask_b)

        loss_main = (loss_ce_f + loss_dice_f + loss_ce_b + loss_dice_b) / 4

        mask_region = un_ps_lab
        mim_batch = unimg * (1 - mask_region)
        mim_outputs = self.model(mim_batch, mode='mim')

        re_batch = unimg * (1 - mask_region) + mim_outputs.detach() * mask_region
        re_batch = torch.flip(re_batch, dims=[2])
        aux_outputs = self.model(torch.cat((re_batch, laimg), dim=0), mode='aux')

        loss_mim_raw = F.l1_loss(unimg, mim_outputs, reduction='none')
        loss_mim = self.lambda_mim * (loss_mim_raw * mask_region).sum() / (mask_region.sum() + 1e-5)

        loss_aux_ce = F.cross_entropy(aux_outputs[unimg.shape[0]:], batch.labeled_label)
        loss_aux_dice = self.dice_loss(aux_outputs[unimg.shape[0]:], batch.labeled_label)
        loss_aux = (loss_aux_ce + loss_aux_dice) / 2

        total_loss = loss_main + loss_mim + loss_aux
        loss_dict = {
            'loss_total': float(total_loss.item()),
            'loss_main': float(loss_main.item()),
            'loss_mim': float(loss_mim.item()),
            'loss_aux': float(loss_aux.item())
        }
        return total_loss, loss_dict

    def train_step(
        self,
        sampled_batch: Dict[str, Any],
        iter_num: int,
        mim_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Standard train_step delegates to selftrain_step."""
        return self.selftrain_step(sampled_batch, iter_num, mim_mask)

    def load_pretrain_weights_for_selftrain(self):
        """Restore or load pre-trained weights before starting self-training phase."""
        pretrain_candidate = None
        if self.pretrain_checkpoint and os.path.exists(self.pretrain_checkpoint):
            pretrain_candidate = self.pretrain_checkpoint
        else:
            best_pre = os.path.join(self.pre_snapshot_path, "best_model.pth")
            legacy_pre = os.path.join(self.pre_snapshot_path, "VNet_best_model.pth")
            target_pre = best_pre if os.path.exists(best_pre) else legacy_pre

            # Auto-restore pre-train weights from W&B if missing locally!
            if not os.path.exists(target_pre) and self.wandb_manager.enabled and is_main_process():
                logging.info("[RESUME] Pre-train weights not found locally. Attempting restore from W&B Cloud...")
                self.wandb_manager.restore_artifact(
                    artifact_name=f"{self.exp_name}_pretrain_best",
                    dest_dir=self.pre_snapshot_path,
                    tag="latest"
                )

            if self.is_ddp:
                torch.distributed.barrier()

            if os.path.exists(best_pre):
                pretrain_candidate = best_pre
            elif os.path.exists(legacy_pre):
                pretrain_candidate = legacy_pre
            elif os.path.exists(os.path.join(self.pre_snapshot_path, "checkpoint_latest.pth")):
                pretrain_candidate = os.path.join(self.pre_snapshot_path, "checkpoint_latest.pth")

        if pretrain_candidate and os.path.exists(pretrain_candidate):
            self.load_checkpoint(pretrain_candidate)
            logging.info(f"[INFO] Loaded pre-trained weights from {pretrain_candidate}")
        else:
            logging.warning("[WARN] No pre-trained checkpoint found! Starting self-training from scratch.")

    def run_pre_train(self) -> Tuple[int, float]:
        """Execute Phase 1: Pre-training via unified training loop."""
        logging.info("=== [PICK] STARTING PRE-TRAINING PHASE ===")
        return self.run_training_loop(
            phase='pre_train',
            max_iterations=self.pre_max_iteration,
            val_interval=self.val_interval,
            save_interval=self.save_interval,
            step_fn=self.pretrain_step,
            snapshot_dir=self.pre_snapshot_path,
            artifact_prefix=f"{self.exp_name}_pretrain",
            best_model_name="best_model.pth",
            legacy_best_model_name="VNet_best_model.pth",
            close_wandb_on_finish=False
        )

    def run_self_train(self) -> Tuple[int, float]:
        """Execute Phase 2: Self-training via unified training loop."""
        logging.info("=== [PICK] STARTING SELF-TRAINING PHASE ===")
        resume = bool(getattr(self.cfg.experiment, 'resume', False))
        latest_pth = os.path.join(self.self_snapshot_path, "checkpoint_latest.pth")
        
        # If not resuming existing self-training checkpoint, load pre-trained weights
        if not (resume and os.path.exists(latest_pth)):
            self.load_pretrain_weights_for_selftrain()

        return self.run_training_loop(
            phase='self_train',
            max_iterations=self.self_max_iteration,
            val_interval=self.val_interval,
            save_interval=self.save_interval,
            step_fn=self.selftrain_step,
            snapshot_dir=self.self_snapshot_path,
            artifact_prefix=f"{self.exp_name}_self",
            start_step_offset=self.pre_max_iteration,
            best_model_name="best_model.pth",
            legacy_best_model_name="VNet_best_model.pth",
            close_wandb_on_finish=False
        )

    def train(self):
        """Execute requested phase: pre_train, self_train, or all."""
        try:
            if self.phase in ('pre_train', 'all'):
                self.run_pre_train()
            if self.phase in ('self_train', 'all'):
                self.run_self_train()
        finally:
            if self.writer is not None:
                self.writer.close()
            self.wandb_manager.finish()
