"""
Weights & Biases (W&B) Cloud Experiment & Artifact Manager.
Provides robust logging, cloud model checkpointing, auto-restore for preemptible instances,
and summary metric tracking for MedSSL-BHSD.
"""

from typing import Dict, Any, Optional, List, Union
import os
import logging
import json
from pathlib import Path


class WandbManager:
    """
    Centralized W&B Manager with graceful fallbacks and cloud artifact synchronization.
    """
    def __init__(
        self,
        cfg,
        run_name: Optional[str] = None,
        fold: Optional[int] = None,
        default_project: str = "PICK_BHSD"
    ):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg.logging, 'use_wandb', False))
        self.save_artifacts = bool(getattr(cfg.logging, 'save_wandb_model', False))
        self.project = getattr(cfg.logging, 'wandb_project', default_project) or default_project
        self.entity = getattr(cfg.logging, 'wandb_entity', None)
        self.run_name = run_name or getattr(cfg.logging, 'wandb_run_name', None)
        self.fold = fold if fold is not None else getattr(cfg.experiment, 'fold', 0)
        self.run = None
        self.run_id = None
        self.mode = os.environ.get('WANDB_MODE', getattr(cfg.logging, 'wandb_mode', 'online'))
        if self.mode not in ('online', 'offline', 'disabled'):
            raise ValueError('wandb_mode must be online, offline or disabled')
        self.identity_path = Path(getattr(cfg.experiment, 'exp_dir', '../experiments')) / getattr(cfg.experiment, 'name', 'exp') / f'fold_{self.fold}' / 'wandb_run.json'
        self.log_failures = 0

        from core.distributed import is_main_process
        if self.enabled and self.mode != "disabled" and is_main_process():
            self._init_wandb()
        else:
            self.enabled = False

    def _init_wandb(self):
        try:
            import wandb
            import re

            if not self.run_name:
                exp_name = getattr(self.cfg.experiment, 'name', 'exp')
                method_name = getattr(self.cfg.method, 'name', 'method')
                self.run_name = f"{exp_name}_{method_name}_fold{self.fold}"

            is_resume = bool(getattr(self.cfg.experiment, 'resume', False))
            identity = self._resume_identity() if is_resume else None
            custom_id = getattr(self.cfg.logging, 'wandb_id', None)
            if identity and custom_id and identity['id'] != custom_id:
                raise ValueError('wandb_id conflicts with saved run identity')
            run_id = custom_id or (identity['id'] if identity else None)
            if is_resume and not run_id:
                raise ValueError('W&B resume needs persisted run identity or explicit logging.wandb_id')
            config_dict = self.cfg.to_dict() if hasattr(self.cfg, 'to_dict') else dict(self.cfg)
            init_kwargs = {
                'project': self.project, 'entity': self.entity, 'name': self.run_name,
                'config': redact_config(config_dict), 'mode': self.mode,
                'reinit': 'create_new',
            }
            if run_id:
                init_kwargs['id'] = str(run_id)
            if self.mode == 'online':
                init_kwargs['resume'] = 'must' if is_resume else 'never'
            elif is_resume:
                logging.warning('W&B offline resume preserves run ID; remote history must be reconciled on sync')
            self.run = wandb.init(**init_kwargs)
            self.run.define_metric('global_step')
            for prefix in ('train/*', 'val/*', 'pre_train/*', 'self_train/*'):
                self.run.define_metric(prefix, step_metric='global_step')
            self.run.define_metric('val/mDice', step_metric='global_step', summary='max')
            self.run.define_metric('val/macro_positive_dice', step_metric='global_step', summary='max')
            self.run.define_metric('train/total', step_metric='global_step', summary='min')
            # Hide raw count metrics from auto-generated default panels to prevent chart clutter
            for pattern in ('*_tp_voxels', '*_fp_voxels', '*_fn_voxels', '*_positive_cases', '*_negative_cases', '*_surface_valid_pairs'):
                self.run.define_metric(f'val/{pattern}', step_metric='global_step', hidden=True)
            self.run_id = self.run.id
            self.entity = getattr(self.run, 'entity', None) or self.entity
            self.identity_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.identity_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.identity(), indent=2))
            temporary.replace(self.identity_path)
            logging.info(
                f"[W&B CLOUD] Initialized Run '{self.run_name}' (ID: {self.run_id}, mode={self.mode}) "
                f"in project '{self.project}'{f' (entity: {self.entity})' if self.entity else ''}"
            )
        except Exception as e:
            if self.run is not None:
                self.run.finish(exit_code=1)
            if bool(getattr(self.cfg.experiment, 'resume', False)) or getattr(self.cfg.logging, 'wandb_required', False):
                raise
            logging.warning(f"[W&B CLOUD] Failed to initialize W&B: {e}. Degrading to local-only logging.")
            self.enabled = False
            self.run = None
            self.run_id = None

    def identity(self):
        return {'id': self.run_id, 'project': self.project, 'entity': self.entity, 'mode': self.mode}

    def _resume_identity(self):
        if self.identity_path.exists():
            identity = json.loads(self.identity_path.read_text())
        else:
            import torch
            phase = getattr(self.cfg.experiment, 'phase', 'all')
            directories = ['self_train', '', 'pre_train'] if phase == 'all' else [phase, '']
            identity = None
            for directory in directories:
                path = self.identity_path.parent / directory / 'checkpoint_latest.pth'
                if path.is_file():
                    identity = torch.load(path, map_location='cpu', weights_only=True).get('wandb_run')
                    if identity:
                        break
        if identity:
            if self.entity is None:
                self.entity = identity.get('entity')
            if identity.get('project') != self.project or (self.entity and identity.get('entity') != self.entity):
                raise ValueError('W&B project/entity conflicts with saved identity')
        return identity

    def log_metrics(self, metrics, step=None):
        """Use an explicit training axis; SDK history steps remain append-only."""
        if not self.enabled or self.run is None:
            return
        payload = dict(metrics)
        if step is not None:
            payload['global_step'] = step
        try:
            self.run.log(payload, commit=True)
        except Exception as exc:
            self.log_failures += 1
            if self.log_failures == 1 or self.log_failures % 100 == 0:
                logging.warning('[W&B] %d log failures; latest: %s', self.log_failures, exc)
            if getattr(self.cfg.logging, 'wandb_required', False):
                raise

    def log_visual_predictions(self, images, step=None, key='val/predictions'):
        if images:
            self.log_metrics({key: images}, step)

    def log_artifact(
        self,
        file_path: str,
        name: str,
        artifact_type: str = "model",
        description: str = "",
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Upload checkpoint or dataset artifact to W&B Cloud Storage.
        """
        if not self.enabled or not self.save_artifacts or self.run is None:
            return False
        if not os.path.exists(file_path):
            logging.warning(f"[W&B ARTIFACT] File not found: {file_path}")
            return False

        try:
            import wandb
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                description=description,
                metadata=metadata or {}
            )
            artifact.add_file(file_path)
            aliases = aliases or ["latest"]
            self.run.log_artifact(artifact, aliases=aliases)
            logging.info(f"[W&B ARTIFACT] Queued '{name}' ({artifact_type}) with aliases {aliases}")
            return True
        except Exception as e:
            logging.warning(f"[W&B ARTIFACT] Failed to log artifact '{name}': {e}")
            return False

    def restore_artifact(
        self,
        artifact_name: str,
        dest_dir: str,
        tag: str = "latest"
    ) -> bool:
        """
        Download artifact from W&B Cloud Storage to local destination directory.
        Used for auto-resuming on pre-emptible GPU instances (e.g. Slurm preemption, Kaggle timeout).
        """
        try:
            import wandb
            api = wandb.Api()
            entity = self.entity
            if not entity:
                try:
                    entity = api.default_entity
                except Exception:
                    entity = None

            candidates = []
            if entity:
                candidates.append(f"{entity}/{self.project}/{artifact_name}:{tag}")
            candidates.append(f"{self.project}/{artifact_name}:{tag}")
            candidates.append(f"{artifact_name}:{tag}")

            for cand in candidates:
                try:
                    art = api.artifact(cand)
                    os.makedirs(dest_dir, exist_ok=True)
                    art.download(root=dest_dir)
                    logging.info(f"[W&B CLOUD] Successfully restored artifact '{cand}' to {dest_dir}!")
                    return True
                except Exception:
                    continue

            logging.warning(f"[W&B CLOUD] Artifact '{artifact_name}:{tag}' not found in W&B project '{self.project}'")
        except Exception as e:
            logging.warning(f"[W&B CLOUD] Failed to query W&B API for artifact '{artifact_name}': {e}")
        return False

    def update_summary(self, summary_dict: Dict[str, Any]):
        """Update top-level run summary (e.g. best_mDice, overall score)."""
        if not self.enabled or self.run is None:
            return
        try:
            for k, v in summary_dict.items():
                self.run.summary[k] = v
        except Exception as e:
            logging.debug(f"[W&B] Failed to update summary: {e}")

    def finish(self, exit_code=0):
        """Cleanly terminate W&B run to finalize sync."""
        if not self.enabled or self.run is None:
            return
        try:
            self.run.finish(exit_code=exit_code)
            logging.info(f"[W&B CLOUD] Finalized run '{self.run_name}'.")
        except Exception as e:
            logging.warning(f"[W&B] Error finalizing run: {e}")
        finally:
            self.run = None


def redact_config(value):
    """Prevent credentials from entering run config or local config snapshots."""
    if isinstance(value, dict):
        return {key: '[REDACTED]' if any(word in key.lower() for word in ('api_key', 'token', 'password', 'secret'))
                else redact_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_config(item) for item in value]
    return value
