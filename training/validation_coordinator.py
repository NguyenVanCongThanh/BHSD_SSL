"""
Periodic Validation Coordinator for 3D Medical Image Segmentation.
Orchestrates distributed validation execution, finite checks, metrics publishing
to TensorBoard / W&B, checkpoint saving, atomic JSON dumps, and DDP score broadcast.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Sequence, Callable
import os
import json
import logging
from pathlib import Path
import numpy as np
import torch

from core.distributed import is_main_process
from evaluation.metrics import format_dice_summary


@dataclass
class ValidationRequest:
    snap_dir: str
    phase: str
    iter_num: int
    global_step: int
    best_dice: float
    best_model_name: str
    legacy_best_model_name: Optional[str]
    art_prefix: str
    latest_pth: str
    should_vis: bool


@dataclass
class ValidationRuntime:
    device: torch.device
    is_ddp: bool
    rank: int
    world_size: int
    class_names: Sequence[str]


@dataclass
class ValidationCallbacks:
    validate: Callable[..., Any]
    save_checkpoint: Callable[..., Any]
    cloud_latest_due: Callable[[int], bool]
    build_visual_payload: Callable[[Any], Dict[str, Any]]
    get_validation_details: Callable[[], Optional[Dict[str, Any]]]
    resume_training_mode: Callable[[], None]
    publish_validation_state: Callable[[float, Dict[str, Any]], None]


@dataclass
class ValidationOutcome:
    best_dice: float
    last_validation_dice: float
    validation_details: Dict[str, Any]


class ValidationCoordinator:
    """Coordinates validation workflow steps and persistence across single and distributed processes."""

    def handle(
        self,
        *,
        request: ValidationRequest,
        runtime: ValidationRuntime,
        callbacks: ValidationCallbacks,
        writer: Optional[Any],
        wandb_manager: Any,
    ) -> ValidationOutcome:
        best_dice = request.best_dice

        if is_main_process():
            num_gpus_str = f" across {runtime.world_size} GPU(s)" if runtime.world_size > 1 else ""
            msg = f"\n>>> [{request.phase} Validation] Periodic evaluation at Iter {request.iter_num}{num_gpus_str}..."
            logging.info(msg)
            print(msg, flush=True)

        if request.should_vis:
            val_mdice, per_class, visual_samples = callbacks.validate(return_visuals=True)
        else:
            val_res = callbacks.validate(return_visuals=False)
            if isinstance(val_res, tuple) and len(val_res) == 3:
                val_mdice, per_class, visual_samples = val_res
            else:
                val_mdice, per_class = val_res
                visual_samples = []

        if not np.isfinite(val_mdice) or not np.isfinite(per_class).all():
            raise FloatingPointError('Validation produced nonfinite metrics')

        last_validation_dice = float(val_mdice)
        details = callbacks.get_validation_details()
        if details is None:
            details = {}
        details.update(iteration=request.iter_num, global_step=request.global_step, phase=request.phase)
        # Publish before persistence so subclass hooks observe this evaluation.
        callbacks.publish_validation_state(last_validation_dice, details)

        if is_main_process():
            class_names = runtime.class_names[1:]
            summary_str = format_dice_summary(per_class, class_names)
            end_msg = (
                f"<<< [{request.phase} Validation] Iter {request.iter_num} | "
                f"mDice: {val_mdice:.4f} (Best: {best_dice:.4f}) | {summary_str}"
            )
            logging.info(end_msg)
            print(end_msg, flush=True)

            if writer is not None:
                writer.add_scalar('val/mDice', val_mdice, request.global_step)
                for c, d in enumerate(per_class):
                    writer.add_scalar(f'val/dice_{class_names[c]}', d, request.global_step)

            if val_mdice > best_dice:
                best_dice = val_mdice
                best_pth = os.path.join(request.snap_dir, request.best_model_name)
                callbacks.save_checkpoint(
                    path=best_pth,
                    iter_num=request.iter_num,
                    best_dice=best_dice,
                    phase=request.phase,
                    artifact_name=f"{request.art_prefix}_best",
                    artifact_type="model",
                    aliases=["latest", "best"],
                    extra_state={
                        "checkpoint_kind": "best",
                        "selection_metric": "val_mDice",
                        "selected_iter": request.iter_num
                    }
                )
                if request.legacy_best_model_name and request.legacy_best_model_name != request.best_model_name:
                    legacy_pth = os.path.join(request.snap_dir, request.legacy_best_model_name)
                    callbacks.save_checkpoint(
                        path=legacy_pth,
                        iter_num=request.iter_num,
                        best_dice=best_dice,
                        phase=request.phase,
                        extra_state={
                            "checkpoint_kind": "best",
                            "selection_metric": "val_mDice",
                            "selected_iter": request.iter_num
                        }
                    )
                logging.info(f"  [★ NEW BEST MODEL] Saved best model with mDice: {best_dice:.4f}")
                wandb_manager.update_summary({
                    f'{request.phase}/best_mDice': best_dice,
                    f'{request.phase}/best_iter': request.iter_num,
                    'best_mDice': best_dice
                })

            # Save latest checkpoint
            callbacks.save_checkpoint(
                path=request.latest_pth,
                iter_num=request.iter_num,
                best_dice=best_dice,
                phase=request.phase,
                artifact_name=f"{request.art_prefix}_latest" if callbacks.cloud_latest_due(request.iter_num) else None,
                artifact_type="checkpoint",
                aliases=["latest"]
            )

            # Send metrics, visual predictions and the table in one validation payload
            val_log = {
                'val/mDice': val_mdice,
                'val/best_mDice': best_dice,
                f'{request.phase}/val_mDice': val_mdice
            }
            for c, d in enumerate(per_class):
                val_log[f'val/dice_{class_names[c]}'] = d

            if request.should_vis and visual_samples:
                val_log.update(callbacks.build_visual_payload(visual_samples))

            macro_pos = details.get('macro_positive_dice') if details else None
            if macro_pos is not None:
                val_log['val/macro_positive_dice'] = float(macro_pos)
                if writer is not None:
                    writer.add_scalar('val/macro_positive_dice', float(macro_pos), request.global_step)

            if wandb_manager.enabled and details and details.get('per_class'):
                eval_table = wandb_manager.build_evaluation_table(details, runtime.class_names)
                if eval_table is not None:
                    val_log['val/evaluation_table'] = eval_table

            if details:
                for row in details.get('per_class', []):
                    name = runtime.class_names[row['class_id']]
                    for key, value in row.items():
                        if key != 'class_id' and value is not None:
                            val_log[f'val/{name}_{key}'] = value
                            if writer is not None:
                                writer.add_scalar(f'val/{name}_{key}', value, request.global_step)

                output = dict(details, global_step=request.global_step, phase=request.phase)
                path = Path(request.snap_dir) / 'validation_latest.json'
                tmp = path.with_suffix('.tmp')
                tmp.write_text(json.dumps(output, indent=2, allow_nan=False))
                tmp.replace(path)

            wandb_manager.log_metrics(val_log, step=request.global_step)

        if runtime.is_ddp:
            score = torch.tensor(best_dice, device=runtime.device, dtype=torch.float64)
            torch.distributed.broadcast(score, src=0)
            best_dice = score.item()
            torch.distributed.barrier()

        callbacks.resume_training_mode()
        return ValidationOutcome(
            best_dice=best_dice,
            last_validation_dice=last_validation_dice,
            validation_details=details or {}
        )

