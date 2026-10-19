"""
PICK (Pseudo-label Informed CutMix / Context Masking Knowledge) Algorithm for BHSD.
Unified implementation for both Binary (Hemorrhage) and Multi-class (6 classes).
Seamlessly manages both Pre-training and Self-training phases with sliding-window validation
and full W&B Cloud Artifacts + Auto-Restore integration.
"""

from typing import Dict, Any, Optional, Tuple, List
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from core.registry import METHODS
from core.distributed import is_main_process
from methods.base_method import BaseMethod
from losses import MaskDiceLoss
from data.masking import context_mask
from evaluation.metrics import format_dice_summary


@METHODS.register("pick")
class PICKMethod(BaseMethod):
    """
    PICK Semi-Supervised Learning Framework:
    - Pre-training phase: CutMix on labeled data + lesion-informed MIM reconstruction + Aux branch.
    - Self-training phase: Bidirectional CutMix (labeled <-> unlabeled) + Unlabeled MIM + Aux branch.
    """
    def __init__(self, model: nn.Module, cfg):
        super(PICKMethod, self).__init__(model, cfg)

        self.base_lr = float(getattr(self.cfg.method, 'base_lr', 0.01))
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

    def _extract_batch_and_mim(self, batch_data: Any) -> Tuple[Dict[str, Any], torch.Tensor]:
        if isinstance(batch_data, (tuple, list)):
            return batch_data[0], batch_data[1]
        sampled_batch = batch_data
        bs = sampled_batch['image'].shape[0]
        if not hasattr(self, '_mask_generator'):
            from data.masking import MaskGeneratorBHSD
            self._mask_generator = MaskGeneratorBHSD(input_size=self.patch_size)
        mim_mask = torch.from_numpy(np.stack([self._mask_generator() for _ in range(bs)]))
        return sampled_batch, mim_mask

    def train_step(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None):
        """Not directly used by PICK as it splits into run_pre_train and run_self_train phases."""
        pass

    def run_pre_train(self):
        logging.info("=== [PICK] STARTING PRE-TRAINING PHASE ===")
        self.model.train()
        optimizer = self.optimizer
        iter_num = 0
        best_dice = 0.0

        writer = SummaryWriter(os.path.join(self.pre_snapshot_path, 'log')) if is_main_process() else None
        latest_pth = os.path.join(self.pre_snapshot_path, "checkpoint_latest.pth")

        resume = bool(getattr(self.cfg.experiment, 'resume', False))
        if resume and not os.path.exists(latest_pth) and self.wandb_manager.enabled and is_main_process():
            logging.info("[RESUME] Local pre-train checkpoint not found. Attempting auto-restore from W&B Cloud...")
            self.wandb_manager.restore_artifact(
                artifact_name=f"{self.exp_name}_pretrain_latest",
                dest_dir=self.pre_snapshot_path,
                tag="latest"
            )

        if self.is_ddp:
            torch.distributed.barrier()

        if resume and os.path.exists(latest_pth):
            iter_num, best_dice = self.load_checkpoint(latest_pth)
            logging.info(f"[RESUME] Resumed pre-training from iter {iter_num} | best_mDice: {best_dice:.4f}")
            if iter_num >= self.pre_max_iteration:
                logging.info(f"[RESUME] Pre-training already completed ({iter_num}/{self.pre_max_iteration}). Skipping.")
                if writer is not None:
                    writer.close()
                return

        remaining_iters = max(self.pre_max_iteration - iter_num, 0)
        max_epoch = remaining_iters // max(len(self.trainloader), 1) + 2
        iter_t0 = time.time()

        for epoch in range(max_epoch):
            if self.is_ddp and hasattr(self, 'batch_sampler') and hasattr(self.batch_sampler, 'set_epoch'):
                self.batch_sampler.set_epoch(epoch)
            elif self.is_ddp and hasattr(self, 'train_sampler') and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)

            for _, batch_data in enumerate(self.trainloader):
                sampled_batch, mim_mask = self._extract_batch_and_mim(batch_data)
                volume_batch = sampled_batch['image'][:self.labeled_bs].to(self.device)
                label_batch = sampled_batch['label'][:self.labeled_bs].to(self.device)
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
                # Foreground lesion probability sum (works for both binary class 1 and multi-class classes 1..5)
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

                loss_mim = F.l1_loss(volume_batch, mim_outputs, reduction='none')
                loss_mim = self.lambda_mim * (loss_mim * mask_region).sum() / (mask_region.sum() + 1e-5)

                loss_aux_ce = F.cross_entropy(aux_outputs, label_batch)
                loss_aux_dice = self.dice_loss(aux_outputs, label_batch)
                loss_aux = (loss_aux_ce + loss_aux_dice) / 2

                loss = loss_main + loss_mim + loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Polynomial learning rate decay (nnUNet standard)
                lr_ = self.base_lr * (1.0 - iter_num / self.pre_max_iteration) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num += 1

                if is_main_process():
                    if writer is not None:
                        writer.add_scalar('loss/total', loss.item(), iter_num)
                        writer.add_scalar('loss/main', loss_main.item(), iter_num)
                        writer.add_scalar('loss/mim', loss_mim.item(), iter_num)
                        writer.add_scalar('loss/aux', loss_aux.item(), iter_num)
                        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter_num)

                    # W&B Logging
                    self.wandb_manager.log_metrics({
                        'train/loss_total': loss.item(),
                        'train/loss_main': loss_main.item(),
                        'train/loss_mim': loss_mim.item(),
                        'train/loss_aux': loss_aux.item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'phase': 'pre_train',
                        'pre_train/iter': iter_num,
                    }, step=iter_num)

                if is_main_process() and (iter_num % 10 == 0 or iter_num == 1):
                    elapsed_10 = time.time() - iter_t0
                    sec_per_it = elapsed_10 / 10 if iter_num > 1 else elapsed_10
                    rem_steps = max(self.pre_max_iteration - iter_num, 0)
                    eta_sec = rem_steps * sec_per_it
                    eta_str = time.strftime('%Hh%Mm%Ss', time.gmtime(eta_sec)) if eta_sec < 86400 else f"{eta_sec/3600:.1f}h"
                    vram_used = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
                    pct = (iter_num / self.pre_max_iteration) * 100
                    logging.info(
                        f"[Pre-train] Iter {iter_num}/{self.pre_max_iteration} ({pct:.1f}%) | "
                        f"{sec_per_it:.2f}s/it | ETA: {eta_str} | "
                        f"Loss: {loss.item():.4f} (Main: {loss_main.item():.4f}, MIM: {loss_mim.item():.4f}, Aux: {loss_aux.item():.4f}) | "
                        f"VRAM: {vram_used:.1f}GB"
                    )
                    iter_t0 = time.time()

                if iter_num % self.save_interval == 0 and iter_num % self.val_interval != 0:
                    if is_main_process():
                        self.save_checkpoint(
                            path=latest_pth,
                            iter_num=iter_num,
                            best_dice=best_dice,
                            phase='pre_train'
                        )

                if iter_num % self.val_interval == 0:
                    if is_main_process():
                        logging.info(f"\n>>> [Pre-train Validation] Periodic evaluation at Iter {iter_num}...")
                        val_visualize = bool(getattr(self.cfg.logging, 'val_visualize', True))
                        should_vis = self.wandb_manager.enabled and val_visualize
                        if should_vis:
                            val_mdice, per_class, visual_samples = self.validate(return_visuals=True)
                        else:
                            val_mdice, per_class = self.validate(return_visuals=False)
                            visual_samples = []
                        summary_str = format_dice_summary(per_class, self.class_names[1:])
                        logging.info(f"<<< [Pre-train Validation] Iter {iter_num} | mDice: {val_mdice:.4f} (Best: {best_dice:.4f}) | {summary_str}")

                        if writer is not None:
                            writer.add_scalar('val/mDice', val_mdice, iter_num)
                            for c, d in enumerate(per_class):
                                writer.add_scalar(f'val/dice_{self.class_names[c+1]}', d, iter_num)

                        if val_mdice > best_dice:
                            best_dice = val_mdice
                            best_model_path = os.path.join(self.pre_snapshot_path, "VNet_best_model.pth")
                            self.save_checkpoint(
                                path=best_model_path,
                                iter_num=iter_num,
                                best_dice=best_dice,
                                phase='pre_train',
                                artifact_name=f"{self.exp_name}_pretrain_best",
                                artifact_type="model",
                                aliases=["latest", "best"]
                            )
                            logging.info(f"  [★ NEW BEST MODEL] Saved pre-train best model: mDice {best_dice:.4f}")
                            self.wandb_manager.update_summary({
                                'pretrain/best_mDice': best_dice,
                                'pretrain/best_iter': iter_num
                            })

                        # Save and upload latest checkpoint
                        self.save_checkpoint(
                            path=latest_pth,
                            iter_num=iter_num,
                            best_dice=best_dice,
                            phase='pre_train',
                            artifact_name=f"{self.exp_name}_pretrain_latest",
                            artifact_type="checkpoint",
                            aliases=["latest"]
                        )

                        val_metrics = {
                            'val/mDice': val_mdice,
                            'val/best_mDice': best_dice,
                            'pre_train/val_mDice': val_mdice
                        }
                        for c, d in enumerate(per_class):
                            val_metrics[f'val/dice_{self.class_names[c+1]}'] = d

                        if should_vis and visual_samples:
                            wandb_images = self.build_visual_samples(visual_samples)
                            if wandb_images:
                                val_metrics['val/predictions'] = wandb_images

                        self.wandb_manager.log_metrics(val_metrics, step=iter_num)

                    if self.is_ddp:
                        torch.distributed.barrier()

                    self.model.train()
                    iter_t0 = time.time()

                if iter_num >= self.pre_max_iteration:
                    break
            if iter_num >= self.pre_max_iteration:
                break

        if is_main_process():
            self.save_checkpoint(latest_pth, iter_num, best_dice, 'pre_train')
            best_pth = os.path.join(self.pre_snapshot_path, "VNet_best_model.pth")
            if not os.path.exists(best_pth):
                self.save_checkpoint(best_pth, iter_num, best_dice, 'pre_train')
            if writer is not None:
                writer.close()
            logging.info(f"=== [PICK] PRE-TRAINING FINISHED | Best mDice: {best_dice:.4f} ===")

    def run_self_train(self):
        logging.info("=== [PICK] STARTING SELF-TRAINING PHASE ===")
        self.model.train()
        optimizer = self.optimizer
        iter_num = 0
        best_dice = 0.0

        writer = SummaryWriter(os.path.join(self.self_snapshot_path, 'log')) if is_main_process() else None
        latest_pth = os.path.join(self.self_snapshot_path, "checkpoint_latest.pth")

        resume = bool(getattr(self.cfg.experiment, 'resume', False))
        if resume and not os.path.exists(latest_pth) and self.wandb_manager.enabled and is_main_process():
            logging.info("[RESUME] Local self-train checkpoint not found. Attempting auto-restore from W&B Cloud...")
            self.wandb_manager.restore_artifact(
                artifact_name=f"{self.exp_name}_self_latest",
                dest_dir=self.self_snapshot_path,
                tag="latest"
            )

        if self.is_ddp:
            torch.distributed.barrier()

        if resume and os.path.exists(latest_pth):
            iter_num, best_dice = self.load_checkpoint(latest_pth)
            logging.info(f"[RESUME] Resumed self-training from iter {iter_num} | best_mDice: {best_dice:.4f}")
            if iter_num >= self.self_max_iteration:
                logging.info(f"[RESUME] Self-training already completed ({iter_num}/{self.self_max_iteration}). Skipping.")
                if writer is not None:
                    writer.close()
                return
        else:
            # Load pre-trained weights
            pretrain_candidate = None
            if self.pretrain_checkpoint and os.path.exists(self.pretrain_checkpoint):
                pretrain_candidate = self.pretrain_checkpoint
            else:
                best_pre = os.path.join(self.pre_snapshot_path, "VNet_best_model.pth")
                # Auto-restore pre-train weights from W&B if missing locally!
                if not os.path.exists(best_pre) and self.wandb_manager.enabled and is_main_process():
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
                elif os.path.exists(os.path.join(self.pre_snapshot_path, "checkpoint_latest.pth")):
                    pretrain_candidate = os.path.join(self.pre_snapshot_path, "checkpoint_latest.pth")

            if pretrain_candidate and os.path.exists(pretrain_candidate):
                self.load_checkpoint(pretrain_candidate)
                logging.info(f"[INFO] Loaded pre-trained weights from {pretrain_candidate}")
            else:
                logging.warning("[WARN] No pre-trained checkpoint found! Starting self-training from scratch.")

        sub_bs = max(int(self.labeled_bs / 2), 1)
        remaining_iters = max(self.self_max_iteration - iter_num, 0)
        max_epoch = remaining_iters // max(len(self.trainloader), 1) + 2
        iter_t0 = time.time()

        for epoch in range(max_epoch):
            if self.is_ddp and hasattr(self, 'batch_sampler') and hasattr(self.batch_sampler, 'set_epoch'):
                self.batch_sampler.set_epoch(epoch)
            elif self.is_ddp and hasattr(self, 'train_sampler') and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)

            for _, batch_data in enumerate(self.trainloader):
                sampled_batch, mim_mask = self._extract_batch_and_mim(batch_data)
                volume_batch = sampled_batch['image'].to(self.device)
                label_batch = sampled_batch['label'].to(self.device)

                img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:self.labeled_bs]
                lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:self.labeled_bs]
                unimg = volume_batch[self.labeled_bs:]
                laimg = volume_batch[:self.labeled_bs]

                unimg_a, unimg_b = unimg[:sub_bs], unimg[sub_bs:]

                raw_net = self.model.module if hasattr(self.model, 'module') else self.model
                with torch.no_grad():
                    _, un_main_outputs, _ = raw_net(unimg)
                    un_prob = F.softmax(un_main_outputs, dim=1)
                    fg_un_prob = un_prob[:, 1:, :, :, :].sum(dim=1, keepdim=True)
                    un_ps_lab = (fg_un_prob > 0.5).float()

                with torch.no_grad():
                    img_mask, _ = context_mask(img_a, self.mask_ratio)

                cutmix_batch_f = img_a * img_mask + unimg_a * (1 - img_mask)
                cutmix_batch_b = unimg_b * img_mask + img_b * (1 - img_mask)

                # Forward CutMix through model without in-place tensor modification
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

                loss_mim = F.l1_loss(unimg, mim_outputs, reduction='none')
                loss_mim = self.lambda_mim * (loss_mim * mask_region).sum() / (mask_region.sum() + 1e-5)

                loss_aux_ce = F.cross_entropy(aux_outputs[unimg.shape[0]:], label_batch[:self.labeled_bs])
                loss_aux_dice = self.dice_loss(aux_outputs[unimg.shape[0]:], label_batch[:self.labeled_bs])
                loss_aux = (loss_aux_ce + loss_aux_dice) / 2

                loss = loss_main + loss_mim + loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Polynomial learning rate decay (nnUNet standard)
                lr_ = self.base_lr * (1.0 - iter_num / self.self_max_iteration) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num += 1

                if is_main_process():
                    if writer is not None:
                        writer.add_scalar('loss/total', loss.item(), iter_num)
                        writer.add_scalar('loss/main', loss_main.item(), iter_num)
                        writer.add_scalar('loss/mim', loss_mim.item(), iter_num)
                        writer.add_scalar('loss/aux', loss_aux.item(), iter_num)
                        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter_num)

                    global_step = self.pre_max_iteration + iter_num
                    self.wandb_manager.log_metrics({
                        'train/loss_total': loss.item(),
                        'train/loss_main': loss_main.item(),
                        'train/loss_mim': loss_mim.item(),
                        'train/loss_aux': loss_aux.item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'phase': 'self_train',
                        'self_train/iter': iter_num,
                    }, step=global_step)

                if is_main_process() and (iter_num % 10 == 0 or iter_num == 1):
                    elapsed_10 = time.time() - iter_t0
                    sec_per_it = elapsed_10 / 10 if iter_num > 1 else elapsed_10
                    rem_steps = max(self.self_max_iteration - iter_num, 0)
                    eta_sec = rem_steps * sec_per_it
                    eta_str = time.strftime('%Hh%Mm%Ss', time.gmtime(eta_sec)) if eta_sec < 86400 else f"{eta_sec/3600:.1f}h"
                    vram_used = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
                    pct = (iter_num / self.self_max_iteration) * 100
                    logging.info(
                        f"[Self-train] Iter {iter_num}/{self.self_max_iteration} ({pct:.1f}%) | "
                        f"{sec_per_it:.2f}s/it | ETA: {eta_str} | "
                        f"Loss: {loss.item():.4f} (Main: {loss_main.item():.4f}, MIM: {loss_mim.item():.4f}, Aux: {loss_aux.item():.4f}) | "
                        f"VRAM: {vram_used:.1f}GB"
                    )
                    iter_t0 = time.time()

                if iter_num % self.save_interval == 0 and iter_num % self.val_interval != 0:
                    if is_main_process():
                        self.save_checkpoint(
                            path=latest_pth,
                            iter_num=iter_num,
                            best_dice=best_dice,
                            phase='self_train'
                        )

                if iter_num % self.val_interval == 0:
                    if is_main_process():
                        logging.info(f"\n>>> [Self-train Validation] Periodic evaluation at Iter {iter_num}...")
                        val_visualize = bool(getattr(self.cfg.logging, 'val_visualize', True))
                        should_vis = self.wandb_manager.enabled and val_visualize
                        if should_vis:
                            val_mdice, per_class, visual_samples = self.validate(return_visuals=True)
                        else:
                            val_mdice, per_class = self.validate(return_visuals=False)
                            visual_samples = []
                        summary_str = format_dice_summary(per_class, self.class_names[1:])
                        logging.info(f"<<< [Self-train Validation] Iter {iter_num} | mDice: {val_mdice:.4f} (Best: {best_dice:.4f}) | {summary_str}")

                        if writer is not None:
                            writer.add_scalar('val/mDice', val_mdice, iter_num)
                            for c, d in enumerate(per_class):
                                writer.add_scalar(f'val/dice_{self.class_names[c+1]}', d, iter_num)

                        if val_mdice > best_dice:
                            best_dice = val_mdice
                            best_model_path = os.path.join(self.self_snapshot_path, "VNet_best_model.pth")
                            self.save_checkpoint(
                                path=best_model_path,
                                iter_num=iter_num,
                                best_dice=best_dice,
                                phase='self_train',
                                artifact_name=f"{self.exp_name}_best_model",
                                artifact_type="model",
                                aliases=["latest", "best"]
                            )
                            logging.info(f"  [★ NEW BEST MODEL] Saved self-train best model: mDice {best_dice:.4f}")
                            self.wandb_manager.update_summary({
                                'self_train/best_mDice': best_dice,
                                'self_train/best_iter': iter_num,
                                'best_mDice': best_dice
                            })

                        # Save and upload latest checkpoint
                        self.save_checkpoint(
                            path=latest_pth,
                            iter_num=iter_num,
                            best_dice=best_dice,
                            phase='self_train',
                            artifact_name=f"{self.exp_name}_self_latest",
                            artifact_type="checkpoint",
                            aliases=["latest"]
                        )

                        val_metrics = {
                            'val/mDice': val_mdice,
                            'val/best_mDice': best_dice,
                            'self_train/val_mDice': val_mdice
                        }
                        for c, d in enumerate(per_class):
                            val_metrics[f'val/dice_{self.class_names[c+1]}'] = d

                        if should_vis and visual_samples:
                            wandb_images = self.build_visual_samples(visual_samples)
                            if wandb_images:
                                val_metrics['val/predictions'] = wandb_images

                        self.wandb_manager.log_metrics(val_metrics, step=global_step)

                    if self.is_ddp:
                        torch.distributed.barrier()

                    self.model.train()
                    iter_t0 = time.time()

                if iter_num >= self.self_max_iteration:
                    break
            if iter_num >= self.self_max_iteration:
                break

        if is_main_process():
            self.save_checkpoint(latest_pth, iter_num, best_dice, 'self_train')
            best_pth = os.path.join(self.self_snapshot_path, "VNet_best_model.pth")
            if not os.path.exists(best_pth):
                self.save_checkpoint(best_pth, iter_num, best_dice, 'self_train')
            if writer is not None:
                writer.close()
            self.wandb_manager.update_summary({
                'final_mDice': best_dice,
                'completed_iters': iter_num
            })
            logging.info(f"=== [PICK] SELF-TRAINING FINISHED | Best mDice: {best_dice:.4f} ===")

    def train(self):
        """Execute requested phase: pre_train, self_train, or all."""
        try:
            if self.phase in ('pre_train', 'all'):
                self.run_pre_train()
            if self.phase in ('self_train', 'all'):
                self.run_self_train()
        finally:
            self.wandb_manager.finish()
