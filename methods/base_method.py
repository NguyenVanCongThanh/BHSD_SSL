"""
Abstract Base Method / Trainer for Semi-Supervised and Supervised Learning Algorithms.
Provides unified training loops, validation engine, checkpointing, and logging (W&B / TensorBoard).
Integrated with WandbManager for cloud artifacts and auto-resume.
"""

from typing import Dict, Any, Optional, Tuple, List
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
from data.dataset_bhsd import BHSDDataset
from data.samplers import TwoStreamBatchSampler, DistributedTwoStreamBatchSampler
from data.transforms3d import RandomRotFlip, RandomCrop, ToTensor
from evaluation.sliding_window import evaluate_dataset_sliding_window
from evaluation.metrics import CLASS_NAMES_MULTICLASS, CLASS_NAMES_BINARY, format_dice_summary
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

        # Task specifications
        self.task = str(getattr(self.cfg.experiment, 'task', 'multiclass')).lower()
        self.is_binary = (self.task == 'binary')
        self.num_classes = 2 if self.is_binary else int(getattr(self.cfg.experiment, 'num_classes', 6))
        self.class_names = CLASS_NAMES_BINARY if self.is_binary else CLASS_NAMES_MULTICLASS
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
                find_unused_parameters=True
            )
            logging.info(f"[Model] DDP initialized on GPU {local_rank} (Rank {self.rank}/{self.world_size})")
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

        # Setup components
        self.setup_dataloaders()
        self.setup_optimizer()
        self.setup_logging()

    def _setup_environment(self):
        deterministic = getattr(self.cfg.experiment, 'deterministic', True)
        seed = int(getattr(self.cfg.experiment, 'seed', 1337))
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            import random
            random.seed(seed)
            np.random.seed(seed)

    def setup_dataloaders(self):
        root_path = getattr(self.cfg.data, 'root_path', '/home/u001015/dataset/BHSD_h5/')
        train_transforms = transforms.Compose([
            RandomRotFlip(),
            RandomCrop(self.patch_size),
            ToTensor()
        ])

        self.db_train = BHSDDataset(
            base_dir=root_path,
            split='train',
            binary=self.is_binary,
            patch_size=self.patch_size,
            fold=self.fold,
            transform=train_transforms
        )

        self.db_val = BHSDDataset(
            base_dir=root_path,
            split='val',
            binary=self.is_binary,
            patch_size=self.patch_size,
            fold=self.fold
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
        base_lr = float(getattr(self.cfg.method, 'base_lr', 0.01))
        momentum = float(getattr(self.cfg.method, 'momentum', 0.9))
        weight_decay = float(getattr(self.cfg.method, 'weight_decay', 0.0001))
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay
        )

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
        aliases: Optional[List[str]] = None
    ):
        if not is_main_process():
            return

        raw_net = self.model.module if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else self.model
        state = {
            'net': raw_net.state_dict(),
            'opt': self.optimizer.state_dict() if self.optimizer is not None else None,
            'iter_num': iter_num,
            'best_dice': best_dice,
            'phase': phase,
            'config': self.cfg.to_dict()
        }
        torch.save(state, str(path))

        # Upload to W&B Artifacts if enabled
        if artifact_name and self.wandb_manager.enabled and self.wandb_manager.save_artifacts:
            self.wandb_manager.log_artifact(
                file_path=str(path),
                name=artifact_name,
                artifact_type=artifact_type,
                description=f"Checkpoint at iter {iter_num} ({phase}) with best_mDice: {best_dice:.4f}",
                aliases=aliases or ["latest"],
                metadata={'iter_num': iter_num, 'best_dice': best_dice, 'phase': phase}
            )

    def load_checkpoint(self, path: str) -> Tuple[int, float]:
        state = torch.load(str(path), map_location=self.device)
        net_dict = state.get('net', state)
        raw_net = self.model.module if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else self.model
        clean_dict = {k[7:] if k.startswith('module.') else k: v for k, v in net_dict.items()}
        raw_net.load_state_dict(clean_dict, strict=False)

        if self.optimizer is not None and 'opt' in state and state['opt'] is not None:
            try:
                self.optimizer.load_state_dict(state['opt'])
            except Exception as e:
                logging.warning(f"[Checkpoint] Could not restore optimizer state: {e}")

        iter_num = state.get('iter_num', 0)
        best_dice = state.get('best_dice', 0.0)
        return iter_num, best_dice

    def validate(self) -> Tuple[float, List[float]]:
        stride_xy = int(getattr(self.cfg.logging, 'val_stride_xy', 32))
        stride_z = int(getattr(self.cfg.logging, 'val_stride_z', 8))
        raw_net = self.model.module if hasattr(self.model, 'module') else self.model
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
            log_progress=True
        )

    @abc.abstractmethod
    def train_step(self, sampled_batch: Dict, mim_mask: torch.Tensor, iter_num: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute a single optimization step.
        Returns total_loss and a dictionary of scalar loss components for logging.
        """
        pass

    def train(self):
        """Default standard training loop."""
        max_iterations = int(getattr(self.cfg.method, 'max_iteration', 10000))
        val_interval = int(getattr(self.cfg.logging, 'val_interval', 500))
        resume = bool(getattr(self.cfg.experiment, 'resume', False))

        self.model.train()
        iter_num = 0
        best_dice = 0.0

        latest_pth = os.path.join(self.snapshot_path, "checkpoint_latest.pth")
        
        # Cloud auto-restore if local checkpoint is missing
        if resume and not os.path.exists(latest_pth) and self.wandb_manager.enabled:
            logging.info("[RESUME] Local checkpoint not found. Attempting auto-restore from W&B Cloud Storage...")
            self.wandb_manager.restore_artifact(
                artifact_name=f"{self.exp_name}_latest",
                dest_dir=self.snapshot_path,
                tag="latest"
            )

        if resume and os.path.exists(latest_pth):
            iter_num, best_dice = self.load_checkpoint(latest_pth)
            logging.info(f"[RESUME] Resumed training from iter {iter_num} | best_mDice: {best_dice:.4f}")

        if iter_num >= max_iterations:
            logging.info(f"Training already completed ({iter_num}/{max_iterations}). Exiting.")
            self.wandb_manager.finish()
            return

        remaining_iters = max(max_iterations - iter_num, 0)
        max_epoch = remaining_iters // max(len(self.trainloader), 1) + 2
        iter_t0 = time.time()

        for epoch in range(max_epoch):
            if self.is_ddp and hasattr(self, 'train_sampler') and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
            elif self.is_ddp and hasattr(self, 'batch_sampler') and hasattr(self.batch_sampler, 'set_epoch'):
                self.batch_sampler.set_epoch(epoch)

            for _, (sampled_batch, mim_mask) in enumerate(self.trainloader):
                loss, loss_dict = self.train_step(sampled_batch, mim_mask, iter_num)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                iter_num += 1

                # TensorBoard & W&B logging (Rank 0 only)
                if is_main_process():
                    if self.writer is not None:
                        for k, v in loss_dict.items():
                            self.writer.add_scalar(f"loss/{k}", v, iter_num)
                        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], iter_num)

                    wandb_dict = {f"train/{k}": v for k, v in loss_dict.items()}
                    wandb_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                    self.wandb_manager.log_metrics(wandb_dict, step=iter_num)

                # Console progress log (Rank 0 only)
                if is_main_process() and (iter_num % 10 == 0 or iter_num == 1):
                    elapsed_10 = time.time() - iter_t0
                    sec_per_it = elapsed_10 / 10 if iter_num > 1 else elapsed_10
                    rem_steps = max(max_iterations - iter_num, 0)
                    eta_sec = rem_steps * sec_per_it
                    eta_str = time.strftime('%Hh%Mm%Ss', time.gmtime(eta_sec)) if eta_sec < 86400 else f"{eta_sec/3600:.1f}h"
                    vram_used = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
                    pct = (iter_num / max_iterations) * 100

                    losses_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                    logging.info(
                        f"Iter {iter_num}/{max_iterations} ({pct:.1f}%) | "
                        f"{sec_per_it:.2f}s/it | ETA: {eta_str} | "
                        f"{losses_str} | VRAM: {vram_used:.1f}GB"
                    )
                    iter_t0 = time.time()

                # Periodic Validation (Rank 0 only)
                if iter_num % val_interval == 0:
                    if is_main_process():
                        logging.info(f"\n>>> [Validation] Periodic evaluation at Iter {iter_num}...")
                        val_mdice, per_class = self.validate()
                        class_names = self.class_names[1:]
                        summary_str = format_dice_summary(per_class, class_names)
                        logging.info(f"<<< [Validation] Iter {iter_num} | mDice: {val_mdice:.4f} (Best: {best_dice:.4f}) | {summary_str}")

                        if self.writer is not None:
                            self.writer.add_scalar('val/mDice', val_mdice, iter_num)
                            for c, d in enumerate(per_class):
                                self.writer.add_scalar(f'val/dice_{class_names[c]}', d, iter_num)

                        if val_mdice > best_dice:
                            best_dice = val_mdice
                            best_pth = os.path.join(self.snapshot_path, "best_model.pth")
                            self.save_checkpoint(
                                path=best_pth,
                                iter_num=iter_num,
                                best_dice=best_dice,
                                phase='train',
                                artifact_name=f"{self.exp_name}_best_model",
                                artifact_type="model",
                                aliases=["latest", "best"]
                            )
                            logging.info(f"  [★ NEW BEST MODEL] Saved best model with mDice: {best_dice:.4f}")
                            self.wandb_manager.update_summary({
                                'best_mDice': best_dice,
                                'best_iter': iter_num
                            })

                        # Save and upload latest checkpoint
                        self.save_checkpoint(
                            path=latest_pth,
                            iter_num=iter_num,
                            best_dice=best_dice,
                            phase='train',
                            artifact_name=f"{self.exp_name}_latest",
                            artifact_type="checkpoint",
                            aliases=["latest"]
                        )

                        val_log = {'val/mDice': val_mdice, 'val/best_mDice': best_dice}
                        for c, d in enumerate(per_class):
                            val_log[f'val/dice_{class_names[c]}'] = d
                        self.wandb_manager.log_metrics(val_log, step=iter_num)

                    if self.is_ddp:
                        torch.distributed.barrier()

                    self.model.train()
                    iter_t0 = time.time()

                if iter_num >= max_iterations:
                    break
            if iter_num >= max_iterations:
                break

        self.save_checkpoint(latest_pth, iter_num, best_dice, 'train')
        self.writer.close()
        self.wandb_manager.update_summary({
            'final_mDice': best_dice,
            'completed_iters': iter_num
        })
        self.wandb_manager.finish()
        logging.info(f"Training finished | Best mDice: {best_dice:.4f}")
