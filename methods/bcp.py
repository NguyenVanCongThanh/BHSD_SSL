"""
Bidirectional Copy-Paste (BCP) Method for Semi-Supervised 3D Medical Segmentation.
Reference: Bai et al., "Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation", CVPR 2023.

Full Two-Stage Pipeline matching CVPR 2023:
- Stage 1 (Pre-training): Copy-Paste data augmentation on labeled partition to establish strong initialization.
- Stage 2 (Self-training): Bidirectional Copy-Paste between labeled and unlabeled volumes using an EMA Teacher
  for stable pseudo-supervision, with adaptive weighting between ground-truth and pseudo-labeled regions.
"""

from typing import Dict, Tuple, Optional, Any
import os
import copy
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import METHODS
from core.distributed import is_main_process
from data.constants import DEFAULT_SUPERVISED_CE_WEIGHT, DEFAULT_SUPERVISED_DICE_WEIGHT
from methods.base_method import BaseMethod
from methods.ssl_utils import (
    extract_logits,
    split_ssl_batch,
    compute_masked_ce_loss,
    update_ema_variables,
)
from losses import MaskDiceLoss
from data.masking import context_mask
from utils.checkpoint import get_raw_model


@METHODS.register("bcp")
class BCPMethod(BaseMethod):
    """
    Bidirectional Copy-Paste (BCP) Framework (CVPR 2023):
    - Two-stage lifecycle: Pre-training (Labeled Copy-Paste) -> Self-training (Bidirectional EMA Teacher).
    - Labeled foreground into Unlabeled background and Unlabeled foreground into Labeled background.
    - Preserves exact parameter naming and capability contracts across 3D backbones.
    """
    REQUIRED_MODEL_CAPABILITIES = {"segmentation"}

    def __init__(self, model: nn.Module, cfg):
        super(BCPMethod, self).__init__(model, cfg)

        self.mask_ratio = float(getattr(self.cfg.method, 'mask_ratio', 2 / 3))
        self.u_weight = float(getattr(self.cfg.method, 'u_weight', 0.5))
        self.ema_decay = float(getattr(self.cfg.method, 'ema_decay', 0.99))
        self.pre_max_iteration = int(getattr(self.cfg.method, 'pre_max_iteration', 2000))
        self.self_max_iteration = int(getattr(self.cfg.method, 'self_max_iteration', 10000))
        self.phase = str(getattr(self.cfg.experiment, 'phase', 'all')).lower()
        self.pretrain_checkpoint = getattr(self.cfg.experiment, 'pretrain_checkpoint', '')

        self.val_interval = int(getattr(self.cfg.logging, 'val_interval', 500))
        self.save_interval = int(getattr(self.cfg.logging, 'save_interval', 50))

        # Snapshot paths for 2-stage execution
        self.pre_snapshot_path = os.path.join(self.snapshot_path, "pre_train")
        self.self_snapshot_path = os.path.join(self.snapshot_path, "self_train")
        os.makedirs(self.pre_snapshot_path, exist_ok=True)
        os.makedirs(self.self_snapshot_path, exist_ok=True)

        self.dice_loss = MaskDiceLoss(
            nclass=self.num_classes,
            batch_dice=self.batch_dice,
            ddp=self.is_ddp
        )
        sup_ce_cfg = getattr(self.cfg.method, 'sup_ce_weight', None)
        sup_dice_cfg = getattr(self.cfg.method, 'sup_dice_weight', None)
        self.sup_ce_weight = DEFAULT_SUPERVISED_CE_WEIGHT if sup_ce_cfg is None else float(sup_ce_cfg)
        self.sup_dice_weight = DEFAULT_SUPERVISED_DICE_WEIGHT if sup_dice_cfg is None else float(sup_dice_cfg)

        # EMA Teacher model initialized as shadow copy
        raw_net = get_raw_model(self.model)
        self.ema_model = copy.deepcopy(raw_net).to(self.device)
        for p in self.ema_model.parameters():
            p.detach_()
        self.ema_model.eval()

    def pretrain_step(
        self,
        sampled_batch: Dict[str, Any],
        iter_num: int,
        mim_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Stage 1: Supervised Pre-training with Copy-Paste Augmentation on Labeled Partition."""
        volume_batch = sampled_batch['image'][:self.labeled_bs].to(self.device)
        label_batch = sampled_batch['label'][:self.labeled_bs].to(self.device)

        sub_bs = max(int(self.labeled_bs // 2), 1)
        img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:2 * sub_bs]
        lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:2 * sub_bs]

        with torch.no_grad():
            img_mask, _ = context_mask(img_a, self.mask_ratio)

        # Copy-paste between labeled pairs
        mix_img = img_a * img_mask + img_b * (1.0 - img_mask)
        mix_lab = lab_a * img_mask.long() + lab_b * (1 - img_mask.long())

        # Combine with original labeled volume for stability
        input_tensor = torch.cat((mix_img, volume_batch), dim=0)
        target_tensor = torch.cat((mix_lab, label_batch), dim=0)

        logits = extract_logits(self.model(input_tensor))
        loss_ce = F.cross_entropy(logits, target_tensor.squeeze(1).long() if target_tensor.dim() > 4 else target_tensor.long())
        loss_dice = self.dice_loss(logits, target_tensor)
        total_loss = self.sup_ce_weight * loss_ce + self.sup_dice_weight * loss_dice

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_dice": float(loss_dice.item()),
        }
        return total_loss, loss_dict

    def selftrain_step(
        self,
        sampled_batch: Dict[str, Any],
        iter_num: int,
        mim_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage 2: Self-training with Bidirectional Copy-Paste (CVPR 2023).
        EMA Teacher generates pseudo-labels on unlabeled volumes.
        Xin and Xout are generated by cross copy-pasting labeled and unlabeled data.
        """
        batch = split_ssl_batch(sampled_batch, self.labeled_bs, device=self.device)
        img_a, img_b = batch.img_a, batch.img_b
        lab_a, lab_b = batch.lab_a, batch.lab_b
        unimg = batch.unlabeled_volume
        unimg_a, unimg_b = batch.unimg_a, batch.unimg_b
        sub_bs = batch.sub_bs

        # 1. EMA Teacher predicts pseudo-labels on unlabeled data
        with torch.no_grad():
            un_logits = extract_logits(self.ema_model(unimg))
            un_prob = F.softmax(un_logits, dim=1)
            plab = torch.argmax(un_prob, dim=1).long()
            plab_a = plab[:sub_bs]
            plab_b = plab[sub_bs:2 * sub_bs] if plab.shape[0] >= 2 * sub_bs else plab[:sub_bs]

            img_mask, _ = context_mask(img_a, self.mask_ratio)

        # 2. Bidirectional Copy-Paste synthesis
        # Mix 1 (Xin): labeled background + unlabeled foreground
        mix_img_in = img_a * img_mask + unimg_a * (1.0 - img_mask)
        mix_lab_in = lab_a * img_mask.long() + plab_a * (1 - img_mask.long())

        # Mix 2 (Xout): unlabeled background + labeled foreground
        mix_img_out = unimg_b * img_mask + img_b * (1.0 - img_mask)
        mix_lab_out = plab_b * img_mask.long() + lab_b * (1 - img_mask.long())

        mix_imgs = torch.cat((mix_img_in, mix_img_out), dim=0)
        mix_labs = torch.cat((mix_lab_in, mix_lab_out), dim=0)

        # 3. Forward student on mixed volumes
        student_logits = extract_logits(self.model(mix_imgs))

        # Spatial weighting: Ground-truth voxels get weight 1.0; pseudo-labeled voxels get weight u_weight
        w_in_single = img_mask.float() + self.u_weight * (1.0 - img_mask.float())
        w_out_single = (1.0 - img_mask.float()) + self.u_weight * img_mask.float()
        w_in = w_in_single.unsqueeze(0).expand(sub_bs, -1, -1, -1)
        w_out = w_out_single.unsqueeze(0).expand(sub_bs, -1, -1, -1)
        weights = torch.cat([w_in, w_out], dim=0)

        loss_ce = compute_masked_ce_loss(student_logits, mix_labs, weights)
        loss_dice = self.dice_loss(student_logits, mix_labs, mask=weights)
        total_loss = self.sup_ce_weight * loss_ce + self.sup_dice_weight * loss_dice

        # 4. Update EMA Teacher parameters
        raw_net = get_raw_model(self.model)
        update_ema_variables(raw_net, self.ema_model, self.ema_decay, iter_num, max_steps=self.self_max_iteration)

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_dice": float(loss_dice.item()),
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
        """Restore pre-trained weights from Stage 1 before initiating Stage 2."""
        pretrain_candidate = None
        if self.pretrain_checkpoint and os.path.isfile(self.pretrain_checkpoint):
            pretrain_candidate = self.pretrain_checkpoint
        else:
            best_pre = os.path.join(self.pre_snapshot_path, "best_model.pth")
            target_latest = os.path.join(self.pre_snapshot_path, "checkpoint_latest.pth")

            if not os.path.exists(best_pre) and self.wandb_manager.enabled and is_main_process():
                logging.info("[RESUME] Pre-train weights not found locally. Attempting restore from W&B Cloud...")
                self.wandb_manager.restore_artifact(
                    artifact_name=f"{self.exp_name}_pretrain_fold{self.fold}_best",
                    dest_dir=self.pre_snapshot_path,
                    tag="latest"
                )

            if self.is_ddp:
                torch.distributed.barrier()

            if os.path.exists(best_pre):
                pretrain_candidate = best_pre
            elif os.path.exists(target_latest):
                pretrain_candidate = target_latest

        if pretrain_candidate and os.path.exists(pretrain_candidate):
            logging.info(f"[BCP] Loading Stage 1 pre-trained checkpoint: {pretrain_candidate}")
            ckpt = torch.load(pretrain_candidate, map_location=self.device)
            raw_net = get_raw_model(self.model)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            raw_net.load_state_dict(state_dict, strict=False)

            # Sync weights into EMA Teacher
            self.ema_model.load_state_dict(raw_net.state_dict())
            logging.info("[BCP] Pre-trained weights successfully transferred to Student and EMA Teacher.")
        else:
            logging.warning("[BCP] No pre-trained checkpoint found. Initializing Stage 2 directly with random weights.")
            raw_net = get_raw_model(self.model)
            self.ema_model.load_state_dict(raw_net.state_dict())

    def run_pre_train(self) -> Tuple[int, float]:
        """Execute Stage 1: Pre-training via unified training loop."""
        logging.info("=== [BCP] STARTING STAGE 1: COPY-PASTE PRE-TRAINING PHASE ===")
        return self.run_training_loop(
            phase='pre_train',
            max_iterations=self.pre_max_iteration,
            val_interval=self.val_interval,
            save_interval=self.save_interval,
            step_fn=self.pretrain_step,
            snapshot_dir=self.pre_snapshot_path,
            artifact_prefix=f"{self.exp_name}_pretrain",
            best_model_name="best_model.pth",
            legacy_best_model_name=None,
            close_wandb_on_finish=False
        )

    def run_self_train(self) -> Tuple[int, float]:
        """Execute Stage 2: Self-training via unified training loop."""
        logging.info("=== [BCP] STARTING STAGE 2: BIDIRECTIONAL SELF-TRAINING PHASE ===")
        resume = bool(getattr(self.cfg.experiment, 'resume', False))
        policy = getattr(self.cfg.experiment, 'resume_policy', 'latest')
        target_name = "checkpoint_epoch_latest.pth" if policy == 'epoch_boundary' else "checkpoint_latest.pth"
        latest_pth = os.path.join(self.self_snapshot_path, target_name)
        alt_pth = os.path.join(self.self_snapshot_path, "checkpoint_latest.pth")

        has_self_train_ckpt = os.path.exists(latest_pth) or os.path.exists(alt_pth)
        if resume and not has_self_train_ckpt and self.wandb_manager is not None and self.wandb_manager.enabled:
            restored = self.wandb_manager.restore_checkpoint(target_name, self.self_snapshot_path)
            if restored and os.path.exists(restored):
                has_self_train_ckpt = True

        if not (resume and has_self_train_ckpt):
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
            legacy_best_model_name=None,
            close_wandb_on_finish=False,
            allow_new_phase=True
        )

    def train(self):
        """Execute requested lifecycle: pre_train, self_train, or both."""
        try:
            if self.phase in ('pre_train', 'all'):
                self.run_pre_train()
            if self.phase in ('self_train', 'all'):
                self.run_self_train()
        finally:
            if self.writer is not None:
                self.writer.close()
            self.wandb_manager.finish(exit_code=1 if sys.exc_info()[0] else 0)
