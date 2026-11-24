"""
Training Checkpoint & Cloud Sync Manager for MedSSL-BHSD.
Extracts checkpoint persistence, resume validation, and cloud upload throttling
out of BaseMethod into a focused, testable coordinator.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from core.distributed import is_main_process
from training.checkpoint_metadata import build_checkpoint_metadata


@dataclass
class CheckpointLoadResult:
    """Structured state returned when restoring a training checkpoint."""

    iter_num: int
    best_dice: float
    checkpoint_state: Dict[str, Any]
    validation_details: Dict[str, Any]
    successful_updates: Optional[int]
    completed_epochs: Optional[int]


class CheckpointManager:
    """Coordinates local checkpoint I/O, provenance metadata, and cloud upload policy."""

    @staticmethod
    def cloud_latest_due(cfg: Any, iteration: int, phase_max_iters: Optional[int] = None) -> bool:
        if getattr(cfg.experiment, 'resume_policy', 'latest') != 'latest':
            return False
        interval = getattr(cfg.logging, 'artifact_interval', None) or 5000
        interval = int(interval)
        if interval < 1:
            raise ValueError('artifact_interval must be positive')
        return iteration % interval == 0 or iteration == phase_max_iters

    @staticmethod
    def cloud_epoch_due(cfg: Any, completed_epochs: int) -> bool:
        if getattr(cfg.experiment, 'resume_policy', 'latest') != 'epoch_boundary':
            return False
        epoch_art_int = getattr(cfg.logging, 'epoch_artifact_interval', None)
        interval = 10 if epoch_art_int is None else int(epoch_art_int)
        if interval < 1:
            raise ValueError('epoch_artifact_interval must be positive')
        return completed_epochs % interval == 0

    @staticmethod
    def save(
        *,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scaler: Optional[Any],
        cfg: Any,
        wandb_manager: Any,
        validation_options: Dict[str, Any],
        successful_updates: int,
        validation_details: Optional[Dict[str, Any]],
        iter_num: int,
        best_dice: float,
        phase: str = 'train',
        artifact_name: Optional[str] = None,
        artifact_type: str = 'checkpoint',
        aliases: Optional[List[str]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not is_main_process():
            return

        from utils.checkpoint import save_checkpoint as _save_ckpt
        from utils.wandb_manager import redact_config

        cfg_dict = redact_config(cfg.to_dict()) if hasattr(cfg, 'to_dict') else None
        wandb_ident = wandb_manager.identity() if getattr(wandb_manager, 'run_id', None) else None
        merged_extra = build_checkpoint_metadata(
            config_dict=cfg_dict,
            wandb_identity=wandb_ident,
            validation_options=validation_options,
            successful_updates=successful_updates,
            validation_details=validation_details,
            extra_state=extra_state,
        )

        _save_ckpt(
            path=path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            iter_num=iter_num,
            best_dice=best_dice,
            phase=phase,
            extra_state=merged_extra,
        )

        if artifact_name and wandb_manager.enabled and wandb_manager.save_artifacts:
            wandb_manager.log_artifact(
                file_path=str(path),
                name=artifact_name,
                artifact_type=artifact_type,
                description=f"Checkpoint at iter {iter_num} ({phase}) with best_mDice: {best_dice:.4f}",
                aliases=aliases or ["latest"],
                metadata={'iter_num': iter_num, 'best_dice': best_dice, 'phase': phase},
            )

    @staticmethod
    def load(
        *,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scaler: Optional[Any],
        device: torch.device,
        fold: int,
        task: str,
        num_classes: int,
        dataset_name: str,
        prefer_ema: bool = False,
    ) -> CheckpointLoadResult:
        from utils.checkpoint import load_checkpoint as _load_ckpt, validate_checkpoint_context

        checkpoint_state = torch.load(path, map_location=device, weights_only=True)
        validate_checkpoint_context(
            checkpoint_state,
            fold=fold,
            task=task,
            num_classes=num_classes,
            dataset_name=dataset_name,
        )
        ckpt_data = _load_ckpt(
            path=path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            prefer_ema=prefer_ema,
            strict=True,
            checkpoint_state=checkpoint_state,
            restore_rng=True,
        )

        successful_updates = None
        if 'successful_updates' in ckpt_data:
            successful_updates = ckpt_data['successful_updates']
        elif isinstance(ckpt_data.get('extra_state'), dict) and 'successful_updates' in ckpt_data['extra_state']:
            successful_updates = ckpt_data['extra_state']['successful_updates']

        completed_epochs = ckpt_data.get('completed_epochs')
        if completed_epochs is None and isinstance(ckpt_data.get('extra_state'), dict):
            completed_epochs = ckpt_data['extra_state'].get('completed_epochs')

        return CheckpointLoadResult(
            iter_num=ckpt_data.get('iter_num', 0),
            best_dice=ckpt_data.get('best_dice', 0.0),
            checkpoint_state=checkpoint_state,
            validation_details=ckpt_data.get('validation_metrics') or {},
            successful_updates=successful_updates,
            completed_epochs=completed_epochs,
        )
