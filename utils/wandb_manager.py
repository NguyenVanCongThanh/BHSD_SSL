"""
Weights & Biases (W&B) Cloud Experiment & Artifact Manager.
Provides robust logging, cloud model checkpointing, auto-restore for preemptible instances,
and summary metric tracking for MedSSL-BHSD.
"""

from typing import Dict, Any, Optional, List, Union
import os
import logging


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

        from core.distributed import is_main_process
        if self.enabled and is_main_process():
            self._init_wandb()
        else:
            self.enabled = False

    def _init_wandb(self):
        try:
            import wandb
            import re

            # Ensure any lingering run in current process is finished cleanly
            if wandb.run is not None:
                try:
                    wandb.finish()
                except Exception:
                    pass

            if not self.run_name:
                exp_name = getattr(self.cfg.experiment, 'name', 'exp')
                method_name = getattr(self.cfg.method, 'name', 'method')
                self.run_name = f"{exp_name}_{method_name}_fold{self.fold}"

            is_resume = bool(getattr(self.cfg.experiment, 'resume', False))
            custom_id = getattr(self.cfg.logging, 'wandb_id', None)

            if custom_id:
                run_id = str(custom_id)
                resume_mode = "allow" if is_resume else None
            elif is_resume:
                clean_slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', self.run_name)
                run_id = clean_slug[:128]
                resume_mode = "allow"
            else:
                run_id = None
                resume_mode = None

            config_dict = self.cfg.to_dict() if hasattr(self.cfg, 'to_dict') else dict(self.cfg)
            init_kwargs = {
                'project': self.project,
                'entity': self.entity,
                'name': self.run_name,
                'config': config_dict,
                'reinit': 'finish_previous'
            }
            if run_id is not None:
                init_kwargs['id'] = run_id
            if resume_mode is not None:
                init_kwargs['resume'] = resume_mode

            try:
                self.run = wandb.init(**init_kwargs)
            except Exception as init_err:
                err_msg = str(init_err).lower()
                if any(k in err_msg for k in ["deleted", "in use", "in_use", "usageerror", "conflict"]):
                    logging.warning(
                        f"[W&B CLOUD] Run ID '{run_id}' xung đột hoặc đã bị xoá trên web ({init_err}). "
                        f"Tự động fallback sang Run ID mới để duy trì đồng bộ W&B Cloud..."
                    )
                    init_kwargs.pop('id', None)
                    init_kwargs.pop('resume', None)
                    self.run = wandb.init(**init_kwargs)
                else:
                    raise init_err

            self.run_id = self.run.id
            logging.info(
                f"[W&B CLOUD] Initialized Run '{self.run_name}' (ID: {self.run_id}, resume={resume_mode}) "
                f"in project '{self.project}'{f' (entity: {self.entity})' if self.entity else ''}"
            )
        except Exception as e:
            logging.warning(f"[W&B CLOUD] Failed to initialize W&B: {e}. Degrading to local-only logging.")
            self.enabled = False
            self.run = None
            self.run_id = None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log key-value metrics at specific global step."""
        if not self.enabled or self.run is None:
            return
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception as e:
            logging.debug(f"[W&B] Failed to log metrics at step {step}: {e}")

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
            wandb.log_artifact(artifact, aliases=aliases)
            logging.info(f"[W&B ARTIFACT] Uploaded '{name}' ({artifact_type}) with aliases {aliases}")
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
            import wandb
            for k, v in summary_dict.items():
                wandb.run.summary[k] = v
        except Exception as e:
            logging.debug(f"[W&B] Failed to update summary: {e}")

    def finish(self):
        """Cleanly terminate W&B run to finalize sync."""
        if not self.enabled or self.run is None:
            return
        try:
            import wandb
            wandb.finish()
            logging.info(f"[W&B CLOUD] Finalized run '{self.run_name}'.")
        except Exception as e:
            logging.debug(f"[W&B] Error finalizing run: {e}")
        finally:
            self.run = None
