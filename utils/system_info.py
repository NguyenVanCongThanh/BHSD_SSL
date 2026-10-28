"""
System, Runtime, and Repository Profiling Utilities.
Collects Git revision, software environment versions, dataset fingerprints,
and exports resolved configurations in a redacted, reproducible format.
"""

from typing import Dict, Any, Optional, Union
import json
import sys
import subprocess
from pathlib import Path
import torch

from evaluation.metrics import METRIC_POLICY
from utils.wandb_manager import redact_config


def collect_git_info(repo_root: Union[str, Path]) -> Dict[str, Optional[Any]]:
    """Capture git commit SHA and dirty working tree status."""
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(repo_root),
            text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=str(repo_root),
            text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {'source_revision': revision, 'source_dirty': dirty}


def build_resolved_config_dict(
    *,
    repo_root: Union[str, Path],
    batch_layout: Any,
    validation_options: Dict[str, Any],
    db_train: Any,
    db_val: Any,
    data_cfg: Any,
) -> Dict[str, Any]:
    """Compile comprehensive runtime reproducibility and dataset fingerprint metadata."""
    git_info = collect_git_info(repo_root)
    layout_dict = batch_layout.to_dict() if hasattr(batch_layout, 'to_dict') else dict(batch_layout)
    return {
        'source_revision': git_info['source_revision'],
        'source_dirty': git_info['source_dirty'],
        'python': sys.version.split()[0],
        'torch': str(torch.__version__),
        'batch_layout': layout_dict,
        'validation': dict(validation_options),
        'metric_policy': METRIC_POLICY,
        'split_fingerprints': {
            'train': getattr(db_train, 'split_fingerprint', None),
            'val': getattr(db_val, 'split_fingerprint', None),
        },
        'intensity_units': getattr(data_cfg, 'intensity_units', None),
        'normalization': getattr(data_cfg, 'normalization', None),
        'actual_labeled_cases': len(getattr(db_train, 'labeled_indices', []) or []),
    }


def record_resolved_config(
    *,
    snapshot_path: Union[str, Path],
    cfg: Any,
    batch_layout: Any,
    validation_options: Dict[str, Any],
    db_train: Any,
    db_val: Any,
    repo_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Compile resolved configuration, attach to cfg.resolved, and write resolved_config.json."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    resolved = build_resolved_config_dict(
        repo_root=repo_root,
        batch_layout=batch_layout,
        validation_options=validation_options,
        db_train=db_train,
        db_val=db_val,
        data_cfg=getattr(cfg, 'data', None),
    )
    cfg.resolved = resolved

    out_dir = Path(snapshot_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'resolved_config.json'
    cfg_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
    out_file.write_text(json.dumps(redact_config(cfg_dict), indent=2))
    return resolved

