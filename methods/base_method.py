"""
Abstract Base Method / Trainer for Semi-Supervised and Supervised Learning Algorithms.
Provides unified training loops, validation engine, checkpointing, and logging (W&B / TensorBoard).
Integrated with WandbManager for cloud artifacts and auto-resume.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import abc
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from core.config_parser import Config
from core.distributed import (
    is_dist_avail_and_initialized,
    get_rank,
    get_world_size,
    is_main_process,
    reduce_tensor
)
from core.registry import DATASETS
from core.optimizers import build_optimizer, calculate_lr
from data.dataset_bhsd import BHSD_MULTICLASS_NAMES, BHSD_BINARY_NAMES
from data.samplers import TwoStreamBatchSampler, DistributedTwoStreamBatchSampler
from data.transforms3d import (
    RandomRotFlip,
    RandomCrop,
    ToTensor,
    RandomScaling3D,
    GaussianBlur3D,
    GammaTransform,
)
from evaluation.sliding_window import evaluate_dataset_sliding_window
from evaluation.metrics import format_dice_summary
from utils.wandb_manager import WandbManager


class BaseMethod(abc.ABC):
    """
    Abstract Base Class for training engines.
    Subclasses only need to implement `train_step()` and specific loss computations.
    """
    def __init__(self, model: nn.Module, cfg: Config):
        self.model = model
        self.cfg = cfg
        self.is_ddp = is_dist_avail_and_initialized()
        self.rank = get_rank()
        self.world_size = get_world_size()

        if self.is_ddp:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Environment & deterministic setup
        self._setup_environment()

        # Task & Dataset specifications
        self.dataset_name = str(getattr(self.cfg.data, 'name', 'bhsd')).lower()
        self.task = str(getattr(self.cfg.data, 'task', getattr(self.cfg.experiment, 'task', 'multiclass'))).lower()
        self.is_binary = (self.task == 'binary')
        self.num_classes = int(getattr(self.cfg.data, 'num_classes', getattr(self.cfg.experiment, 'num_classes', 2 if self.is_binary else 6)))

        cfg_class_names = getattr(self.cfg.data, 'class_names', None)
        if cfg_class_names:
            self.class_names = list(cfg_class_names)
        else:
            self.class_names = BHSD_BINARY_NAMES if self.is_binary else BHSD_MULTICLASS_NAMES

        self.patch_size = tuple(getattr(self.cfg.data, 'patch_size', [32, 160, 160]))
        self.fold = int(getattr(self.cfg.experiment, 'fold', 0))
        self.exp_name = getattr(self.cfg.experiment, 'name', 'experiment')

        # Snapshot path
        exp_dir = getattr(self.cfg.experiment, 'exp_dir', '../experiments')
        self.snapshot_path = os.path.join(exp_dir, self.exp_name, f"fold_{self.fold}")
        if is_main_process():
            os.makedirs(self.snapshot_path, exist_ok=True)

        # Multi-GPU support (DDP vs DP Default)
        if self.is_ddp:
            # Distributed Data Parallel mode
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.model = self.model.to(self.device)
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True
            )
            # Enable static graph for multi-branch architectures (PICK: main + mim + aux decoders)
            # This prevents "marked as ready twice" error when calling forward multiple times per iteration
            if hasattr(self.model, '_set_static_graph'):
                self.model._set_static_graph()
            logging.info(f"[Model] DDP initialized on GPU {local_rank} (Rank {self.rank}/{self.world_size}) with static graph enabled")
        elif torch.cuda.is_available():
            # DataParallel default (preserving backward compatibility)
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model).to(self.device)
                logging.info(f"[Model] Multi-GPU enabled with {torch.cuda.device_count()} GPUs (DataParallel)")
            else:
                self.model = self.model.to(self.device)
                logging.info(f"[Model] Single-GPU on {torch.cuda.get_device_name(0)}")
        else:
            self.model = self.model.to(self.device)
            logging.warning("[Model] No GPU detected, running on CPU!")

        # Mixed Precision (AMP) & GradScaler setup
        self.use_amp = bool(getattr(self.cfg.experiment, 'amp', True)) and torch.cuda.is_available()
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp and is_main_process():
            logging.info("[AMP] Mixed precision (FP16) training enabled with torch.cuda.amp.GradScaler.")

        # Setup components
        self.setup_dataloaders()
        self.setup_optimizer()
        self.setup_logging()

    def _setup_environment(self):
        deterministic = getattr(self.cfg.experiment, 'deterministic', True)
        seed = int(getattr(self.cfg.experiment, 'seed', 1337))
        from utils.reproducibility import set_seed
        set_seed(seed=seed, deterministic=deterministic, rank=self.rank if self.is_ddp else 0)

    def setup_dataloaders(self):
        root_path = getattr(self.cfg.data, 'root_path', '/home/u001015/notebook/preprocessed_bhsd_pick/')
        
        # Check if extended nnUNet-style data augmentation is enabled in config
        use_extended_da = getattr(self.cfg.data, 'extended_aug', True)
        aug_list = [
            RandomRotFlip(),
        ]
        if use_extended_da:
            aug_list.extend([
                RandomScaling3D(scale_range=(0.85, 1.15), prob=0.2),
                GammaTransform(gamma_range=(0.7, 1.5), prob=0.2),
                GaussianBlur3D(sigma_range=(0.5, 1.0), prob=0.15),
            ])
        aug_list.extend([
            RandomCrop(self.patch_size),
            ToTensor()
        ])
        train_transforms = transforms.Compose(aug_list)

        dataset_name = self.dataset_name
        if not DATASETS.contains(dataset_name):
            logging.warning(f"[Data] Dataset '{dataset_name}' not found in registry. Falling back to 'bhsd'.")
            dataset_name = 'bhsd'

        self.db_train = DATASETS.build(
            dataset_name,
            base_dir=root_path,
            split='train',
            binary=self.is_binary,
            patch_size=self.patch_size,
            fold=self.fold,
            transform=train_transforms,
            num_classes=self.num_classes,
            class_names=self.class_names
        )

        self.db_val = DATASETS.build(
            dataset_name,
            base_dir=root_path,
            split='val',
            binary=self.is_binary,
            patch_size=self.patch_size,
            fold=self.fold,
            num_classes=self.num_classes,
            class_names=self.class_names
        )

        batch_size = int(getattr(self.cfg.data, 'batch_size', 4))
        labeled_bs = int(getattr(self.cfg.data, 'labeled_bs', 2))
        labelnum = min(int(getattr(self.cfg.data, 'labelnum', 15)), len(self.db_train))
        max_samples_cfg = int(getattr(self.cfg.data, 'max_samples', -1))
        max_samples = len(self.db_train) if max_samples_cfg <= 0 else min(max_samples_cfg, len(self.db_train))

        labeled_idxs = list(range(labelnum))
        unlabeled_idxs = list(range(labelnum, max_samples))

        unlabeled_bs = batch_size - labeled_bs
        if self.is_ddp:
            if unlabeled_bs > 0 and len(unlabeled_idxs) > 0:
                self.batch_sampler = DistributedTwoStreamBatchSampler(
                    labeled_idxs,
                    unlabeled_idxs,
                    batch_size=batch_size,
                    secondary_batch_size=unlabeled_bs,
                    num_replicas=self.world_size,
                    rank=self.rank
                )
                self.trainloader = DataLoader(
                    self.db_train, batch_sampler=self.batch_sampler, num_workers=2, pin_memory=True
                )
            else:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.db_train, num_replicas=self.world_size, rank=self.rank, shuffle=True
                )
                self.trainloader = DataLoader(
                    self.db_train, batch_size=batch_size, sampler=self.train_sampler, num_workers=2, pin_memory=True
                )
        else:
            if unlabeled_bs > 0 and len(unlabeled_idxs) > 0:
                batch_sampler = TwoStreamBatchSampler(
                    labeled_idxs, unlabeled_idxs, batch_size, unlabeled_bs
                )
                self.trainloader = DataLoader(
                    self.db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=False
                )
            else:
                self.trainloader = DataLoader(
                    self.db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False
                )

    def setup_optimizer(self):
        self.optimizer = build_optimizer(self.model, self.cfg)

    def update_learning_rate(self, iter_num: int, max_iterations: int) -> float:
        """Update learning rate per iteration with support for modular schedulers."""
        scheduler_type = str(getattr(self.cfg.method, 'lr_scheduler', 'poly')).lower()
        base_lr = float(getattr(self.cfg.method, 'base_lr', 0.01))
        poly_power = float(getattr(self.cfg.method, 'poly_power', 0.9))
        warmup_iters = int(getattr(self.cfg.method, 'warmup_iters', 0))

        current_lr = calculate_lr(
            iter_num=iter_num,
            max_iterations=max_iterations,
            base_lr=base_lr,
            scheduler_type=scheduler_type,
            poly_power=poly_power,
            warmup_iters=warmup_iters
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        return current_lr

    def setup_logging(self):
        log_dir = os.path.join(self.snapshot_path, 'log')
        if is_main_process():
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            self.wandb_manager = WandbManager(cfg=self.cfg, fold=self.fold)
            self.use_wandb = self.wandb_manager.enabled
        else:
            self.writer = None
            self.wandb_manager = WandbManager(cfg=self.cfg, fold=self.fold)
            self.wandb_manager.enabled = False
            self.use_wandb = False

    def save_checkpoint(
        self,
        path: str,
        iter_num: int,
        best_dice: float,
        phase: str = 'train',
        artifact_name: Optional[str] = None,
        artifact_type: str = "checkpoint",
        aliases: Optional[List[str]] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ):
        if not is_main_process():
            return

        from utils.checkpoint import save_checkpoint as _save_ckpt
        cfg_dict = self.cfg.to_dict() if hasattr(self.cfg, 'to_dict') else None
        merged_extra = {'config': cfg_dict}
        if extra_state:
            merged_extra.update(extra_state)

        _save_ckpt(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler if hasattr(self, 'scaler') else None,
            iter_num=iter_num,
            best_dice=best_dice,
            phase=phase,
            extra_state=merged_extra
        )

        if artifact_name and self.wandb_manager.enabled and self.wandb_manager.save_artifacts:
            self.wandb_manager.log_artifact(
                file_path=str(path),
                name=artifact_name,
                artifact_type=artifact_type,
                description=f"Checkpoint at iter {iter_num} ({phase}) with best_mDice: {best_dice:.4f}",
                aliases=aliases or ["latest"],
                metadata={'iter_num': iter_num, 'best_dice': best_dice, 'phase': phase}
            )

    def load_checkpoint(self, path: str, prefer_ema: bool = False) -> Tuple[int, float]:
        from utils.checkpoint import load_checkpoint as _load_ckpt
        ckpt_data = _load_ckpt(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler if hasattr(self, 'scaler') else None,
            device=self.device,
            prefer_ema=prefer_ema
        )
        return ckpt_data.get('iter_num', 0), ckpt_data.get('best_dice', 0.0)

    def build_visual_samples(self, visual_records: List[Dict[str, Any]]) -> List[Any]:
        """Build interactive segmentation masks for W&B."""
        if not self.wandb_manager.enabled or not visual_records:
            return []
        from utils.visualizer import create_wandb_segmentation_samples
        class_labels = {i: name for i, name in enumerate(self.class_names)}
        return create_wandb_segmentation_samples(visual_records, class_labels)

    def log_visual_samples(self, visual_records: List[Dict[str, Any]], step: int):
        """Build and log interactive segmentation masks to W&B."""
        wandb_images = self.build_visual_samples(visual_records)
        if wandb_images:
            self.wandb_manager.log_visual_predictions(wandb_images, step=step)

    def validate(self, return_visuals: bool = False) -> Union[Tuple[float, List[float]], Tuple[float, List[float], List[Dict[str, Any]]]]:
        stride_xy = int(getattr(self.cfg.logging, 'val_stride_xy', 32))
        stride_z = int(getattr(self.cfg.logging, 'val_stride_z', 8))
        raw_net = self.model.module if hasattr(self.model, 'module') else self.model
        num_vis = int(getattr(self.cfg.logging, 'num_vis_samples', 4)) if (return_visuals and self.rank == 0) else 0
        return evaluate_dataset_sliding_window(
            model=raw_net,
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

    @abc.abstractmethod
    def train_step(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute a single optimization step.
        Returns total_loss and a dictionary of scalar loss components for logging.
        """
        pass

    def _log_iteration_progress(
        self,
        iter_num: int,
        max_iters: int,
        loss_dict: Dict[str, float],
        iter_t0: float,
        phase: str
    ) -> float:
        """Log iteration progress to console (Rank 0 only). Returns refreshed timer."""
        elapsed_10 = time.time() - iter_t0
        sec_per_it = elapsed_10 / 10 if iter_num > 1 else elapsed_10
        rem_steps = max(max_iters - iter_num, 0)
        eta_sec = rem_steps * sec_per_it
        eta_str = time.strftime('%Hh%Mm%Ss', time.gmtime(eta_sec)) if eta_sec < 86400 else f"{eta_sec/3600:.1f}h"
        vram_used = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
        pct = (iter_num / max_iters) * 100

        losses_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
        logging.info(
            f"[{phase}] Iter {iter_num}/{max_iters} ({pct:.1f}%) | "
            f"{sec_per_it:.2f}s/it | ETA: {eta_str} | "
            f"{losses_str} | VRAM: {vram_used:.1f}GB"
        )
        return time.time()

    def _handle_periodic_validation(
        self,
        snap_dir: str,
        phase: str,
        iter_num: int,
        global_step: int,
        best_dice: float,
        best_model_name: str,
        legacy_best_model_name: Optional[str],
        art_prefix: str,
        latest_pth: str,
        should_vis: bool
    ) -> float:
        """Run validation across all DDP ranks, log metrics, and update checkpoints."""
        if is_main_process():
            num_gpus_str = f" across {self.world_size} GPU(s)" if self.world_size > 1 else ""
            msg = f"\n>>> [{phase} Validation] Periodic evaluation at Iter {iter_num}{num_gpus_str}..."
            logging.info(msg)
            print(msg, flush=True)

        if should_vis:
            val_mdice, per_class, visual_samples = self.validate(return_visuals=True)
        else:
            val_res = self.validate(return_visuals=False)
            if isinstance(val_res, tuple) and len(val_res) == 3:
                val_mdice, per_class, visual_samples = val_res
            else:
                val_mdice, per_class = val_res
                visual_samples = []

        if is_main_process():
            class_names = self.class_names[1:]
            summary_str = format_dice_summary(per_class, class_names)
            end_msg = f"<<< [{phase} Validation] Iter {iter_num} | mDice: {val_mdice:.4f} (Best: {best_dice:.4f}) | {summary_str}"
            logging.info(end_msg)
            print(end_msg, flush=True)

            if self.writer is not None:
                self.writer.add_scalar('val/mDice', val_mdice, iter_num)
                for c, d in enumerate(per_class):
                    self.writer.add_scalar(f'val/dice_{class_names[c]}', d, iter_num)

            if val_mdice > best_dice:
                best_dice = val_mdice
                best_pth = os.path.join(snap_dir, best_model_name)
                self.save_checkpoint(
                    path=best_pth,
                    iter_num=iter_num,
                    best_dice=best_dice,
                    phase=phase,
                    artifact_name=f"{art_prefix}_best",
                    artifact_type="model",
                    aliases=["latest", "best"]
                )
                if legacy_best_model_name and legacy_best_model_name != best_model_name:
                    legacy_pth = os.path.join(snap_dir, legacy_best_model_name)
                    self.save_checkpoint(
                        path=legacy_pth,
                        iter_num=iter_num,
                        best_dice=best_dice,
                        phase=phase
                    )
                logging.info(f"  [★ NEW BEST MODEL] Saved best model with mDice: {best_dice:.4f}")
                self.wandb_manager.update_summary({
                    f'{phase}/best_mDice': best_dice,
                    f'{phase}/best_iter': iter_num,
                    'best_mDice': best_dice
                })

            # Save and upload latest checkpoint
            self.save_checkpoint(
                path=latest_pth,
                iter_num=iter_num,
                best_dice=best_dice,
                phase=phase,
                artifact_name=f"{art_prefix}_latest",
                artifact_type="checkpoint",
                aliases=["latest"]
            )

            # Atomic logging: bundle metrics + visual predictions in the SAME payload
            val_log = {
                'val/mDice': val_mdice,
                'val/best_mDice': best_dice,
                f'{phase}/val_mDice': val_mdice
            }
            for c, d in enumerate(per_class):
                val_log[f'val/dice_{class_names[c]}'] = d

            if should_vis and visual_samples:
                wandb_images = self.build_visual_samples(visual_samples)
                if wandb_images:
                    val_log['val/predictions'] = wandb_images

            self.wandb_manager.log_metrics(val_log, step=global_step)

        if self.is_ddp:
            torch.distributed.barrier()

        self.model.train()
        return best_dice

    def run_training_loop(
        self,
        phase: str = 'train',
        max_iterations: Optional[int] = None,
        val_interval: Optional[int] = None,
        save_interval: Optional[int] = None,
        step_fn: Optional[Any] = None,
        snapshot_dir: Optional[str] = None,
        artifact_prefix: Optional[str] = None,
        start_step_offset: int = 0,
        best_model_name: str = "best_model.pth",
        legacy_best_model_name: Optional[str] = "VNet_best_model.pth",
        close_wandb_on_finish: bool = True
    ) -> Tuple[int, float]:
        """
        Universal, extensible training engine supporting single-phase and multi-phase algorithms.
        Handles DDP synchronization, learning rate schedules, AMP FP16 GradScaler,
        TensorBoard & W&B metrics logging, periodic validation, and checkpoints.
        """
        max_iters = int(max_iterations if max_iterations is not None else getattr(self.cfg.method, 'max_iteration', 10000))
        val_int = int(val_interval if val_interval is not None else getattr(self.cfg.logging, 'val_interval', 500))
        save_int = int(save_interval if save_interval is not None else getattr(self.cfg.logging, 'save_interval', 50))
        snap_dir = snapshot_dir or self.snapshot_path
        os.makedirs(snap_dir, exist_ok=True)

        step_callback = step_fn or self.train_step
        art_prefix = artifact_prefix or self.exp_name
        resume = bool(getattr(self.cfg.experiment, 'resume', False))

        self.model.train()
        iter_num = 0
        best_dice = 0.0

        latest_pth = os.path.join(snap_dir, "checkpoint_latest.pth")

        # Cloud auto-restore if local checkpoint is missing
        if resume and not os.path.exists(latest_pth) and self.wandb_manager.enabled and is_main_process():
            logging.info(f"[RESUME] Local checkpoint not found in {snap_dir}. Attempting auto-restore from W&B Cloud Storage...")
            self.wandb_manager.restore_artifact(
                artifact_name=f"{art_prefix}_latest",
                dest_dir=snap_dir,
                tag="latest"
            )

        if self.is_ddp:
            torch.distributed.barrier()

        if resume and os.path.exists(latest_pth):
            iter_num, best_dice = self.load_checkpoint(latest_pth)
            logging.info(f"[RESUME] Resumed phase '{phase}' from iter {iter_num} | best_mDice: {best_dice:.4f}")

        if iter_num >= max_iters:
            logging.info(f"Phase '{phase}' already completed ({iter_num}/{max_iters}). Skipping.")
            if close_wandb_on_finish:
                self.wandb_manager.finish()
            return iter_num, best_dice

        remaining_iters = max(max_iters - iter_num, 0)
        max_epoch = remaining_iters // max(len(self.trainloader), 1) + 2
        iter_t0 = time.time()

        for epoch in range(max_epoch):
            if self.is_ddp and hasattr(self, 'train_sampler') and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
            elif self.is_ddp and hasattr(self, 'batch_sampler') and hasattr(self.batch_sampler, 'set_epoch'):
                self.batch_sampler.set_epoch(epoch)

            for _, batch_data in enumerate(self.trainloader):
                cur_lr = self.update_learning_rate(iter_num, max_iters)
                if isinstance(batch_data, (tuple, list)):
                    sampled_batch = batch_data[0]
                    mim_mask = batch_data[1] if len(batch_data) > 1 else None
                else:
                    sampled_batch = batch_data
                    mim_mask = None

                try:
                    loss, loss_dict = step_callback(sampled_batch, iter_num, mim_mask)
                except TypeError:
                    try:
                        loss, loss_dict = step_callback(sampled_batch, iter_num)
                    except TypeError:
                        loss, loss_dict = step_callback(sampled_batch)

                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                iter_num += 1
                global_step = start_step_offset + iter_num

                # TensorBoard & W&B logging (Rank 0 only)
                if is_main_process():
                    if self.writer is not None:
                        for k, v in loss_dict.items():
                            self.writer.add_scalar(f"loss/{k}", v, iter_num)
                        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], iter_num)

                    wandb_dict = {f"train/{k}": v for k, v in loss_dict.items()}
                    wandb_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                    wandb_dict['phase'] = phase
                    wandb_dict[f'{phase}/iter'] = iter_num
                    self.wandb_manager.log_metrics(wandb_dict, step=global_step)

                # Console progress log (Rank 0 only)
                if is_main_process() and (iter_num % 10 == 0 or iter_num == 1):
                    iter_t0 = self._log_iteration_progress(iter_num, max_iters, loss_dict, iter_t0, phase)

                # Periodic Checkpoint Saving between validation intervals
                if iter_num % save_int == 0 and iter_num % val_int != 0:
                    if is_main_process():
                        self.save_checkpoint(
                            path=latest_pth,
                            iter_num=iter_num,
                            best_dice=best_dice,
                            phase=phase
                        )

                # Periodic Validation (All DDP Ranks evaluate in parallel)
                if iter_num % val_int == 0:
                    val_visualize = bool(getattr(self.cfg.logging, 'val_visualize', True))
                    should_vis = self.wandb_manager.enabled and val_visualize and is_main_process()
                    best_dice = self._handle_periodic_validation(
                        snap_dir=snap_dir,
                        phase=phase,
                        iter_num=iter_num,
                        global_step=global_step,
                        best_dice=best_dice,
                        best_model_name=best_model_name,
                        legacy_best_model_name=legacy_best_model_name,
                        art_prefix=art_prefix,
                        latest_pth=latest_pth,
                        should_vis=should_vis
                    )
                    iter_t0 = time.time()

                if iter_num >= max_iters:
                    break
            if iter_num >= max_iters:
                break

        if is_main_process():
            self.save_checkpoint(latest_pth, iter_num, best_dice, phase)
            best_pth = os.path.join(snap_dir, best_model_name)
            if not os.path.exists(best_pth):
                self.save_checkpoint(best_pth, iter_num, best_dice, phase)
            if legacy_best_model_name:
                legacy_pth = os.path.join(snap_dir, legacy_best_model_name)
                if not os.path.exists(legacy_pth):
                    self.save_checkpoint(legacy_pth, iter_num, best_dice, phase)
            logging.info(f"=== [{phase.upper()}] FINISHED | Best mDice: {best_dice:.4f} ===")

        if close_wandb_on_finish:
            if self.writer is not None:
                self.writer.close()
            self.wandb_manager.update_summary({
                'final_mDice': best_dice,
                'completed_iters': iter_num
            })
            self.wandb_manager.finish()

        return iter_num, best_dice


    def train(self):
        """Default standard training loop executing single-phase training."""
        max_iterations = int(getattr(self.cfg.method, 'max_iteration', 10000))
        val_interval = int(getattr(self.cfg.logging, 'val_interval', 500))
        save_interval = int(getattr(self.cfg.logging, 'save_interval', 50))
        self.run_training_loop(
            phase='train',
            max_iterations=max_iterations,
            val_interval=val_interval,
            save_interval=save_interval,
            step_fn=self.train_step,
            snapshot_dir=self.snapshot_path,
            artifact_prefix=self.exp_name,
            close_wandb_on_finish=True
        )

