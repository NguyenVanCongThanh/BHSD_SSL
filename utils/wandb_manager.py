"""
Weights & Biases (W&B) Cloud Experiment & Artifact Manager.
Provides robust logging, cloud model checkpointing, auto-restore for preemptible instances,
and summary metric tracking for MedSSL-BHSD.
"""

from typing import Dict, Any, Optional, List, Union, Sequence
import os
import logging
import json
import subprocess
import sys
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
        self._hidden_metrics = set()
        self._confirmed_uploads = 0
        self.artifact_keep_versions = int(getattr(cfg.logging, 'artifact_keep_versions', 2))
        self.cache_cleanup_interval = int(getattr(cfg.logging, 'artifact_cache_cleanup_interval', 5))
        if self.artifact_keep_versions < 0 or self.cache_cleanup_interval < 1:
            raise ValueError('artifact_keep_versions must be nonnegative and artifact_cache_cleanup_interval positive')

        from core.distributed import is_main_process
        if self.enabled and self.mode != "disabled" and is_main_process():
            self.check_fresh_run_directory(cfg, self.fold)
            self._init_wandb()
        else:
            self.enabled = False

    @staticmethod
    def check_fresh_run_directory(cfg, fold=None):
        """Reject checkpoint collisions before creating a run or replacing its identity."""
        if bool(getattr(cfg.experiment, 'resume', False)):
            return
        fold = fold if fold is not None else getattr(cfg.experiment, 'fold', 0)
        root = (Path(getattr(cfg.experiment, 'exp_dir', '../experiments'))
                / getattr(cfg.experiment, 'name', 'exp') / f'fold_{fold}')
        directories = [root]
        if getattr(cfg.method, 'name', '') == 'pick':
            phase = str(getattr(cfg.experiment, 'phase', 'all')).lower()
            directories.extend(root / name for name in ('pre_train', 'self_train')
                               if phase in ('all', name))
        for directory in directories:
            for name in ('checkpoint_latest.pth', 'best_model.pth', 'VNet_best_model.pth'):
                if (directory / name).exists():
                    raise FileExistsError(
                        f'Existing checkpoints in {directory}; use resume or a new experiment directory')

    def _init_wandb(self):
        try:
            import wandb

            if not self.run_name:
                exp_name = getattr(self.cfg.experiment, 'name', 'exp')
                method_name = getattr(self.cfg.method, 'name', 'method')
                self.run_name = f"{exp_name}_{method_name}_fold{self.fold}"

            is_resume = bool(getattr(self.cfg.experiment, 'resume', False))
            new_segment = bool(getattr(self.cfg.logging, 'wandb_resume_new_run', False))
            if new_segment and not is_resume:
                raise ValueError('wandb_resume_new_run requires training resume')
            # Training state resumes, but a new tracking segment must not inherit the source ID.
            identity = self._resume_identity() if is_resume and not new_segment else None
            custom_id = getattr(self.cfg.logging, 'wandb_id', None)
            if identity and custom_id and identity['id'] != custom_id:
                raise ValueError('wandb_id conflicts with saved run identity')
            run_id = custom_id or (identity['id'] if identity else None)
            if is_resume and not new_segment and not run_id:
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
                init_kwargs['resume'] = 'must' if is_resume and not new_segment else 'never'
            elif is_resume:
                logging.warning('W&B offline resume preserves run ID; remote history must be reconciled on sync')
            self.run = wandb.init(**init_kwargs)
            self.run.define_metric('global_step')
            for prefix in ('train/*', 'val/*', 'pre_train/*', 'self_train/*'):
                self.run.define_metric(prefix, step_metric='global_step')
            self.run.define_metric('val/mDice', step_metric='global_step', summary='max')
            self.run.define_metric('val/macro_positive_dice', step_metric='global_step', summary='max')
            self.run.define_metric('train/total', step_metric='global_step', summary='min')
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
        identity = json.loads(self.identity_path.read_text()) if self.identity_path.exists() else None
        import torch
        phase = getattr(self.cfg.experiment, 'phase', 'all')
        directories = ['self_train', '', 'pre_train'] if phase == 'all' else [phase, '']
        for directory in directories:
            path = self.identity_path.parent / directory / 'checkpoint_latest.pth'
            if path.is_file():
                checkpoint_identity = torch.load(path, map_location='cpu', weights_only=True).get('wandb_run')
                if checkpoint_identity:
                    if identity and any(identity.get(key) != checkpoint_identity.get(key)
                                        for key in ('id', 'project', 'entity')):
                        raise ValueError(f'W&B saved identity conflicts with checkpoint {path}')
                    identity = checkpoint_identity
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
            # W&B supports suffix globs only; register each actual per-class key.
            hidden_suffixes = ('_tp_voxels', '_fp_voxels', '_fn_voxels',
                               '_positive_cases', '_negative_cases', '_surface_valid_pairs')
            for key in payload:
                if key.startswith('val/') and key.endswith(hidden_suffixes) and key not in self._hidden_metrics:
                    self.run.define_metric(key, step_metric='global_step', hidden=True)
                    self._hidden_metrics.add(key)
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

    def build_evaluation_table(self, details: Dict[str, Any], class_names: Sequence[str]) -> Optional[Any]:
        """
        Build a wandb.Table summarizing per-class validation metrics.
        Returns Table instance or None if disabled, empty, or on failure. Does not send logs.
        """
        if not self.enabled or not details or not details.get('per_class'):
            return None
        try:
            import wandb
            columns = [
                "class_id", "class_name", "positive_cases", "negative_cases",
                "dice_case", "dice_positive", "hd95_mm", "asd_mm",
                "surface_valid_pairs", "fp_case_rate", "fp_ml",
                "precision_micro", "recall_micro", "tp_voxels", "fp_voxels", "fn_voxels"
            ]
            table_rows = []
            for row in details['per_class']:
                name = class_names[row['class_id']]
                table_rows.append([
                    row.get('class_id'),
                    name,
                    row.get('positive_cases'),
                    row.get('negative_cases'),
                    row.get('dice_case_mean'),
                    row.get('dice_positive_mean'),
                    row.get('hd95_mm_mean'),
                    row.get('asd_mm_mean'),
                    row.get('surface_valid_pairs'),
                    row.get('negative_fp_case_rate'),
                    row.get('negative_fp_ml_mean'),
                    row.get('precision_micro'),
                    row.get('recall_micro'),
                    row.get('tp_voxels'),
                    row.get('fp_voxels'),
                    row.get('fn_voxels')
                ])
            return wandb.Table(columns=columns, data=table_rows)
        except Exception as table_err:
            logging.warning(f"[W&B] Failed to build evaluation summary table: {table_err}")
            return None

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
            if self.mode == 'online':
                # Never prune using an upload that has merely been queued.
                try:
                    artifact.wait(timeout=60)
                except Exception as exc:
                    logging.warning('[W&B] Upload confirmation failed; skipping storage cleanup: %s', exc)
                else:
                    self._maintain_artifact_storage(artifact, name, artifact_type)
            return True
        except Exception as e:
            logging.warning(f"[W&B ARTIFACT] Failed to log artifact '{name}': {e}")
            return False

    def _maintain_artifact_storage(self, artifact, name, artifact_type):
        """Prune this run's old versions only after a replacement is committed."""
        try:
            if self.artifact_keep_versions:
                import wandb
                collection = f'{self.entity}/{self.project}/{name}'
                versions = list(wandb.Api().artifacts(artifact_type, collection))
                owned = []
                for version in versions:
                    owner = version.logged_by()
                    if owner is not None and owner.id == self.run_id:
                        owned.append(version)
                owned.sort(key=lambda v: int(v.version.lstrip('v')), reverse=True)
                # Eventual API consistency must not cause the prior latest to be deleted.
                if any(v.id == artifact.id for v in owned):
                    for old in owned[self.artifact_keep_versions:]:
                        if old.id == artifact.id or {'latest', 'best'} & set(old.aliases):
                            continue
                        old.delete(delete_aliases=True)
        except Exception as exc:
            logging.warning('[W&B] Could not prune artifact versions: %s', exc)
        self._confirmed_uploads += 1
        if self._confirmed_uploads % self.cache_cleanup_interval == 0:
            self._cleanup_artifact_cache()

    def _cleanup_artifact_cache(self):
        target = getattr(self.cfg.logging, 'artifact_cache_size', '1GB')
        if not target:
            return
        try:
            # Public CLI; leave upload staging/temp files untouched.
            result = subprocess.run(
                [sys.executable, '-m', 'wandb', 'artifact', 'cache', 'cleanup', str(target)],
                capture_output=True, text=True, timeout=30, check=True)
            logging.info('[W&B CACHE] %s', result.stdout.strip())
        except Exception as exc:
            logging.warning('[W&B CACHE] Cleanup failed: %s', exc)

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
            if self.mode == 'online':
                self._cleanup_artifact_cache()
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
