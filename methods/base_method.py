"""
Abstract Base Method / Trainer for Semi-Supervised and Supervised Learning Algorithms.
Provides unified training loops, validation engine, checkpointing, and logging (W&B / TensorBoard).
Integrated with WandbManager for cloud artifacts and auto-resume.
"""

from typing import Dict, Any, Optional, Tuple, List
import abc
import inspect
import os
import sys
import time
from pathlib import Path
import logging
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from core.config_parser import Config
from core.batching import resolve_batch_layout
from core.distributed import (
    is_dist_avail_and_initialized,
    get_rank,
    get_world_size,
    is_main_process
)
from core.optimizers import build_optimizer, calculate_lr
from data.dataset_bhsd import BHSD_MULTICLASS_NAMES, BHSD_BINARY_NAMES
from data.dataloader_factory import build_bhsd_dataloaders
from utils.wandb_manager import WandbManager
from training import GradientHandler, ValidationRunner


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
        self.batch_layout = resolve_batch_layout(
            int(getattr(cfg.data, 'batch_size', 4)),
            int(getattr(cfg.data, 'labeled_bs', 2)),
            self.world_size if self.is_ddp else 1,
            str(getattr(cfg.method, 'name', 'pick')),
        )
        self.batch_size = self.batch_layout.local_batch_size
        self.labeled_bs = self.batch_layout.local_labeled_bs
        self.unlabeled_bs = self.batch_layout.local_unlabeled_bs
        logging.info(
            '[Batch] global=%d (%dL+%dU), world_size=%d, per_rank=%d (%dL+%dU)',
            self.batch_layout.global_batch_size, self.batch_layout.global_labeled_bs,
            self.batch_layout.global_unlabeled_bs, self.batch_layout.world_size,
            self.batch_size, self.labeled_bs, self.unlabeled_bs,
        )

        if self.is_ddp:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
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
                device_ids=[local_rank] if self.device.type == "cuda" else None,
                output_device=local_rank if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=True
            )
            # PICK has repeated forwards through shared encoder branches.
            if str(getattr(cfg.method, 'name', '')).lower() == 'pick':
                self.model._set_static_graph()
            logging.info('[Model] DDP initialized on %s (rank %s/%s)', self.device, self.rank, self.world_size)
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
        self.successful_updates = 0

    def _setup_environment(self):
        deterministic = getattr(self.cfg.experiment, 'deterministic', True)
        seed = int(getattr(self.cfg.experiment, 'seed', 1337))
        from utils.reproducibility import set_seed
        set_seed(seed=seed, deterministic=deterministic, rank=self.rank if self.is_ddp else 0)

    def setup_dataloaders(self):
        bundle = build_bhsd_dataloaders(
            cfg=self.cfg,
            batch_layout=self.batch_layout,
            dataset_name=self.dataset_name,
            fold=self.fold,
            is_ddp=self.is_ddp,
            rank=self.rank,
            world_size=self.world_size,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
            is_binary=self.is_binary,
            class_names=self.class_names,
        )
        self.db_train = bundle.db_train
        self.db_val = bundle.db_val
        self.trainloader = bundle.trainloader
        self.train_sampler = bundle.train_sampler
        self.batch_sampler = bundle.batch_sampler

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
            self._record_resolved_config()
            self.wandb_manager = WandbManager(cfg=self.cfg, fold=self.fold)
            self.wandb_manager.update_summary({'batch_layout': self.batch_layout.to_dict()})
            self.use_wandb = self.wandb_manager.enabled
        else:
            self.writer = None
            self.wandb_manager = WandbManager(cfg=self.cfg, fold=self.fold)
            self.wandb_manager.enabled = False
            self.use_wandb = False

    def validation_options(self):
        return {
            'patch_size': self.patch_size,
            'stride_xy': int(getattr(self.cfg.logging, 'val_stride_xy', 32)),
            'stride_z': int(getattr(self.cfg.logging, 'val_stride_z', 8)),
            'use_amp': bool(getattr(self.cfg.logging, 'val_amp', self.use_amp)),
            'use_gaussian': bool(getattr(self.cfg.logging, 'val_gaussian', True)),
            'use_tta': bool(getattr(self.cfg.logging, 'val_tta', False)),
            'compute_surface': bool(getattr(self.cfg.logging, 'val_surface_metrics', False)),
        }

    def _record_resolved_config(self):
        from utils.system_info import record_resolved_config
        repo_root = Path(__file__).resolve().parents[1]
        record_resolved_config(
            snapshot_path=self.snapshot_path,
            cfg=self.cfg,
            batch_layout=self.batch_layout,
            validation_options=self.validation_options(),
            db_train=self.db_train,
            db_val=self.db_val,
            repo_root=repo_root
        )

    def cloud_latest_due(self, iteration):
        interval = getattr(self.cfg.logging, 'artifact_interval', None) or getattr(self.cfg.logging, 'val_interval', 500)
        interval = int(interval)
        if interval < 1:
            raise ValueError('artifact_interval must be positive')
        return iteration % interval == 0 or iteration == getattr(self, '_phase_max_iters', None)

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
        from utils.wandb_manager import redact_config
        from training.checkpoint_metadata import build_checkpoint_metadata
        cfg_dict = redact_config(self.cfg.to_dict()) if hasattr(self.cfg, 'to_dict') else None
        wandb_ident = self.wandb_manager.identity() if getattr(self.wandb_manager, 'run_id', None) else None
        merged_extra = build_checkpoint_metadata(
            config_dict=cfg_dict,
            wandb_identity=wandb_ident,
            validation_options=self.validation_options(),
            successful_updates=getattr(self, 'successful_updates', 0),
            validation_details=getattr(self, 'validation_details', None),
            extra_state=extra_state
        )

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
        from utils.checkpoint import load_checkpoint as _load_ckpt, validate_checkpoint_context
        checkpoint_state = torch.load(path, map_location=self.device, weights_only=True)
        validate_checkpoint_context(checkpoint_state, fold=self.fold, task=self.task,
                                    num_classes=self.num_classes, dataset_name=self.dataset_name)
        ckpt_data = _load_ckpt(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler if hasattr(self, 'scaler') else None,
            device=self.device,
            prefer_ema=prefer_ema,
            strict=True,
            checkpoint_state=checkpoint_state,
            restore_rng=True
        )
        self.validation_details = ckpt_data.get('validation_metrics') or {}
        if 'successful_updates' in ckpt_data:
            self.successful_updates = ckpt_data['successful_updates']
        elif 'successful_updates' in ckpt_data.get('extra_state', {}):
            self.successful_updates = ckpt_data['extra_state']['successful_updates']
        return ckpt_data.get('iter_num', 0), ckpt_data.get('best_dice', 0.0)

    def build_visual_samples(self, visual_records: List[Dict[str, Any]]) -> List[Any]:
        """Build interactive segmentation masks for W&B."""
        if not self.wandb_manager.enabled or not visual_records:
            return []
        from utils.visualizer import create_wandb_segmentation_samples
        class_labels = {i: name for i, name in enumerate(self.class_names)}
        return create_wandb_segmentation_samples(visual_records, class_labels,
            getattr(self.cfg.logging, 'visualization', {}))

    def build_visual_payload(self, records):
        if not self.wandb_manager.enabled or not records:
            return {}
        from utils.visualizer import create_wandb_visual_payload
        return create_wandb_visual_payload(records,
            {i: name for i, name in enumerate(self.class_names)},
            getattr(self.cfg.logging, 'visualization', {}))

    def log_visual_samples(self, visual_records, step):
        self.wandb_manager.log_metrics(self.build_visual_payload(visual_records), step=step)

    def validate(self, return_visuals=False):
        raw_net = self.model.module if hasattr(self.model, 'module') else self.model
        return self.evaluate_validation_model(raw_net, return_visuals, 'net')

    def evaluate_validation_model(self, model, return_visuals=False, weights_source='net'):
        runner = ValidationRunner(
            val_dataset=self.db_val,
            num_classes=self.num_classes,
            is_binary=self.is_binary,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
            options=self.validation_options(),
            visual_settings=dict(getattr(self.cfg.logging, 'visualization', {})),
            num_vis_samples=int(getattr(self.cfg.logging, 'num_vis_samples', 4))
        )
        res = runner.run(model, return_visuals=return_visuals, weights_source=weights_source)
        self.validation_details = runner.last_details
        return res

    @abc.abstractmethod
    def train_step(self, sampled_batch: Dict, iter_num: int, mim_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute a single optimization step.
        Returns total_loss and a dictionary of scalar loss components for logging.
        """
        pass

    def training_optimizers(self):
        """All optimizers participating in one coordinated training update."""
        return [self.optimizer]

    def training_models(self):
        """All models participating in one coordinated training update (e.g. model1 and model2 for CPS)."""
        models = [self.model]
        if hasattr(self, 'model2') and isinstance(self.model2, nn.Module):
            models.append(self.model2)
        return models

    def after_optimizer_step(self, iter_num):
        """Update derived model state only after a successful optimizer step."""

    def apply_gradients(self, loss, iter_num):
        handler = GradientHandler(
            optimizers=self.training_optimizers(),
            models=self.training_models(),
            scaler=self.scaler if hasattr(self, 'scaler') else None,
            use_amp=self.use_amp,
            is_ddp=self.is_ddp,
            device=self.device,
            clip_norm=12.0,
            after_step_fn=self.after_optimizer_step
        )
        success = handler.apply_gradients(loss, iter_num)
        if success and hasattr(self, 'successful_updates'):
            self.successful_updates += 1
        return success

    def optimize_batch(self, sampled_batch, iter_num, mim_mask=None, step_fn=None):
        callback = step_fn or self.train_step
        # Inspect once per call; never catch exceptions thrown inside user code.
        args = (sampled_batch, iter_num, mim_mask)
        signature = inspect.signature(callback)
        for count in (3, 2, 1):
            try:
                signature.bind(*args[:count])
            except TypeError:
                continue
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                loss, loss_dict = callback(*args[:count])
            self.apply_gradients(loss, iter_num)
            return loss, loss_dict
        raise TypeError('Training callback must accept batch[, iteration[, mask]]')

    def _log_iteration_progress(
        self,
        iter_num: int,
        max_iters: int,
        loss_dict: Dict[str, float],
        iter_t0: float,
        phase: str
    ) -> float:
        """Log iteration progress to console (Rank 0 only). Returns refreshed timer."""
        from utils.logger import format_iteration_progress_string
        elapsed_10 = time.time() - iter_t0
        sec_per_it = elapsed_10 / 10 if iter_num > 1 else elapsed_10
        vram_used = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
        msg = format_iteration_progress_string(
            iter_num=iter_num,
            max_iters=max_iters,
            sec_per_it=sec_per_it,
            loss_dict=loss_dict,
            phase=phase,
            vram_used=vram_used,
        )
        logging.info(msg)
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
        from training.validation_coordinator import (
            ValidationCoordinator,
            ValidationRequest,
            ValidationRuntime,
            ValidationCallbacks,
        )
        request = ValidationRequest(
            snap_dir=snap_dir,
            phase=phase,
            iter_num=iter_num,
            global_step=global_step,
            best_dice=best_dice,
            best_model_name=best_model_name,
            legacy_best_model_name=legacy_best_model_name,
            art_prefix=art_prefix,
            latest_pth=latest_pth,
            should_vis=should_vis,
        )
        runtime = ValidationRuntime(
            device=self.device,
            is_ddp=self.is_ddp,
            rank=self.rank,
            world_size=self.world_size,
            class_names=self.class_names,
        )

        def publish_validation_state(dice, details):
            self.last_validation_dice = dice
            self.validation_details = details

        callbacks = ValidationCallbacks(
            validate=self.validate,
            save_checkpoint=self.save_checkpoint,
            cloud_latest_due=self.cloud_latest_due,
            build_visual_payload=self.build_visual_payload,
            get_validation_details=lambda: getattr(self, 'validation_details', {}),
            resume_training_mode=self.model.train,
            publish_validation_state=publish_validation_state,
        )
        coordinator = ValidationCoordinator()
        outcome = coordinator.handle(
            request=request,
            runtime=runtime,
            callbacks=callbacks,
            writer=self.writer,
            wandb_manager=self.wandb_manager,
        )
        self.last_validation_dice = outcome.last_validation_dice
        self.validation_details = outcome.validation_details
        return outcome.best_dice

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
        close_wandb_on_finish: bool = True,
        allow_new_phase: bool = False
    ) -> Tuple[int, float]:
        """
        Universal, extensible training engine supporting single-phase and multi-phase algorithms.
        Handles DDP synchronization, learning rate schedules, AMP FP16 GradScaler,
        TensorBoard & W&B metrics logging, periodic validation, and checkpoints.
        """
        max_iters = int(max_iterations if max_iterations is not None else getattr(self.cfg.method, 'max_iteration', 10000))
        val_int = int(val_interval if val_interval is not None else getattr(self.cfg.logging, 'val_interval', 500))
        save_int = int(save_interval if save_interval is not None else getattr(self.cfg.logging, 'save_interval', 50))
        if max_iters <= 0 or val_int <= 0 or save_int <= 0:
            raise ValueError('Iteration count and validation/save intervals must be positive')
        if len(self.trainloader) == 0:
            raise ValueError('Training dataloader has no batches')
        self._phase_max_iters = max_iters
        snap_dir = snapshot_dir or self.snapshot_path
        os.makedirs(snap_dir, exist_ok=True)

        step_callback = step_fn or self.train_step
        art_prefix = f'{artifact_prefix or self.exp_name}_fold{self.fold}'
        resume = bool(getattr(self.cfg.experiment, 'resume', False))

        self.model.train()
        self.validation_details = {}
        iter_num = 0
        best_dice = float("-inf")

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

        if not resume and any(os.path.exists(os.path.join(snap_dir, name)) for name in
                              ('checkpoint_latest.pth', best_model_name, legacy_best_model_name or best_model_name)):
            raise FileExistsError(f'Existing checkpoints in {snap_dir}; use resume or a new experiment directory')

        if resume and not os.path.exists(latest_pth):
            has_best = any(os.path.isfile(os.path.join(snap_dir, name)) for name in
                           (best_model_name, legacy_best_model_name or best_model_name))
            if has_best or not allow_new_phase:
                raise FileNotFoundError(f'Cannot resume {phase}: latest checkpoint missing in {snap_dir}')

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

                loss, loss_dict = self.optimize_batch(
                    sampled_batch, iter_num, mim_mask, step_fn=step_callback)

                iter_num += 1
                global_step = start_step_offset + iter_num

                # TensorBoard & W&B logging (Rank 0 only)
                if is_main_process():
                    if self.writer is not None:
                        for k, v in loss_dict.items():
                            self.writer.add_scalar(f"loss/{k}", v, global_step)
                        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], global_step)

                    wandb_dict = {f"train/{k}": v for k, v in loss_dict.items()}
                    wandb_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                    wandb_dict['phase'] = phase
                    wandb_dict[f'{phase}/iter'] = iter_num
                    log_interval = int(getattr(self.cfg.logging, 'log_interval', 10))
                    if log_interval < 1:
                        raise ValueError('log_interval must be positive')
                    if iter_num == 1 or iter_num % log_interval == 0:
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
                            phase=phase,
                            artifact_name=f"{art_prefix}_latest" if self.cloud_latest_due(iter_num) else None,
                            artifact_type="checkpoint"
                        )

                # Periodic Validation (All DDP Ranks evaluate in parallel)
                if iter_num % val_int == 0:
                    val_visualize = bool(getattr(self.cfg.logging, 'val_visualize', True))
                    visual_every = int(getattr(self.cfg.logging, 'visual_every_n_validations', 1))
                    if visual_every < 1:
                        raise ValueError('visual_every_n_validations must be positive')
                    should_vis = self.wandb_manager.enabled and val_visualize and is_main_process() and (iter_num // val_int) % visual_every == 0
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

        if iter_num % val_int:
            best_dice = self._handle_periodic_validation(
                snap_dir, phase, iter_num, start_step_offset + iter_num,
                best_dice, best_model_name, legacy_best_model_name,
                art_prefix, latest_pth, False)

        if is_main_process():
            # Final validation already saved latest and any selected best.
            # Do not fabricate a best checkpoint from unvalidated final weights.
            logging.info(f"=== [{phase.upper()}] FINISHED | Best mDice: {best_dice:.4f} ===")

        if close_wandb_on_finish:
            if self.writer is not None:
                self.writer.close()
            self.wandb_manager.update_summary({
                'best_mDice': best_dice,
                'last_validation_mDice': getattr(self, 'last_validation_dice', None),
                'completed_iters': iter_num
            })
            self.wandb_manager.finish()

        return iter_num, best_dice


    def train(self):
        """Default standard training loop executing single-phase training."""
        max_iterations = int(getattr(self.cfg.method, 'max_iteration', 10000))
        val_interval = int(getattr(self.cfg.logging, 'val_interval', 500))
        save_interval = int(getattr(self.cfg.logging, 'save_interval', 50))
        try:
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
        except BaseException as e:
            import traceback
            logging.error(f"[FATAL TRAINING ERROR]: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            if self.writer is not None:
                self.writer.close()
            self.wandb_manager.finish(exit_code=1)
            raise
