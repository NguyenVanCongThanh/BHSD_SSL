"""
Hierarchical YAML Configuration Manager with Dot-Access and Composable Overrides.
Enables clean, reproducible, and parameterized experiments without messy argparse boilerplate.
Supports composable 3-tier configurations: Datasets + Models + Methods.
"""

from typing import Any, Dict, List, Optional, Union
import os
import sys
import copy
import yaml
import argparse


class Config(dict):
    """
    Dict subclass that enables attribute-style dot access (cfg.data.batch_size).
    Recursively wraps nested dictionaries.
    """
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, Config):
                self[key] = Config(value)
            elif isinstance(value, list):
                self[key] = [Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v for v in value]

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config object back to standard Python dict."""
        out = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [item.to_dict() if isinstance(item, Config) else item for item in v]
            else:
                out[k] = v
        return out

    def to_yaml(self) -> str:
        """Dump configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


def _deep_update(target: Dict, source: Dict) -> Dict:
    """Recursively update target dict with source dict."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _parse_cli_override_val(val_str: str) -> Any:
    """Parse string representation of value to Python native type."""
    if val_str.lower() in ('true', 'yes'):
        return True
    if val_str.lower() in ('false', 'no'):
        return False
    if val_str.lower() in ('none', 'null'):
        return None
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    # List support if formatted as [a,b,c]
    if val_str.startswith('[') and val_str.endswith(']'):
        parts = [p.strip() for p in val_str[1:-1].split(',') if p.strip()]
        return [_parse_cli_override_val(p) for p in parts]
    return val_str


# Mapping of known flat arguments to their target section
SECTION_MAPPING = {
    # Data section
    'dataset': 'data',
    'root_path': 'data',
    'patch_size': 'data',
    'batch_size': 'data',
    'labeled_bs': 'data',
    'labelnum': 'data',
    'max_samples': 'data',
    
    # Method section
    'method': 'method',
    'optimizer': 'method',
    'pre_max_iteration': 'method',
    'self_max_iteration': 'method',
    'max_iteration': 'method',
    'base_lr': 'method',
    'lr_scheduler': 'method',
    'poly_power': 'method',
    'consistency': 'method',
    'consistency_rampup': 'method',
    'lambda_': 'method',
    'lambda_mim': 'method',
    'mask_ratio': 'method',
    'cps_weight': 'method',
    'ema_decay': 'method',
    'u_weight': 'method',
    
    # Experiment section
    'exp': 'experiment',
    'name': 'experiment',
    'task': 'experiment',
    'num_classes': 'experiment',
    'fold': 'experiment',
    'seed': 'experiment',
    'deterministic': 'experiment',
    'exp_dir': 'experiment',
    'resume': 'experiment',
    'amp': 'experiment',
    'phase': 'experiment',
    'pretrain_checkpoint': 'experiment',

    # Model section
    'model': 'model',
    'num_stages': 'model',

    # Logging section
    'use_wandb': 'logging',
    'save_wandb_model': 'logging',
    'wandb_project': 'logging',
    'wandb_entity': 'logging',
    'wandb_run_name': 'logging',
    'wandb_id': 'logging',
    'val_interval': 'logging',
    'save_interval': 'logging',
    'val_stride_xy': 'logging',
    'val_stride_z': 'logging',
}


def parse_cli_overrides(unknown_args: List[str]) -> Dict[str, Any]:
    """
    Parse command line arguments in format:
    --fold 1 --data.batch_size 8 --method.base_lr 0.005 --use_wandb
    into nested dictionary. Supports both nested keys and legacy flat keys.
    """
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            key = arg[2:]
            if '=' in key:
                k, v = key.split('=', 1)
                val = _parse_cli_override_val(v)
                i += 1
            else:
                k = key
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                    val = _parse_cli_override_val(unknown_args[i + 1])
                    i += 2
                else:
                    # Boolean flag
                    val = True
                    i += 1

            if '.' in k:
                sub_keys = k.split('.')
            else:
                # Flat argument: check section mapping
                target_sec = SECTION_MAPPING.get(k)
                if target_sec is not None:
                    # Special aliasing: --model VNet -> model.name: VNet, --dataset bhsd -> data.name: bhsd
                    if k in ('model', 'exp', 'method', 'dataset'):
                        sub_keys = [target_sec, 'name']
                    else:
                        sub_keys = [target_sec, k]
                else:
                    sub_keys = [k]

            curr = overrides
            for sub_k in sub_keys[:-1]:
                if sub_k not in curr or not isinstance(curr[sub_k], dict):
                    curr[sub_k] = {}
                curr = curr[sub_k]
            curr[sub_keys[-1]] = val
        else:
            i += 1
    return overrides


def _resolve_config_path(category: str, identifier: str) -> Optional[str]:
    """
    Resolve config path for given category ('datasets', 'models', 'methods').
    Example: 'bhsd_multiclass' -> 'configs/datasets/bhsd_multiclass.yaml'
    """
    if not identifier:
        return None
    # If direct path exists
    if os.path.exists(identifier):
        return identifier

    configs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    for ext in ('.yaml', '.yml'):
        cand = os.path.join(configs_root, category, f"{identifier}{ext}")
        if os.path.exists(cand):
            return cand
    return None


def load_config(
    config_path: Optional[str] = None,
    dataset_config: Optional[str] = None,
    model_config: Optional[str] = None,
    method_config: Optional[str] = None,
    cli_args: Optional[List[str]] = None,
    default_base_path: Optional[str] = None
) -> Config:
    """
    Load YAML configuration file with multi-tier composition:
    Base Config -> Dataset Config -> Model Config -> Method Config -> Exp Config -> CLI Overrides.
    """
    merged_dict: Dict[str, Any] = {}

    configs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))

    # 1. Load base configuration
    if default_base_path is None:
        cand_base = os.path.join(configs_root, "base.yaml")
        if os.path.exists(cand_base):
            default_base_path = cand_base

    if default_base_path and os.path.exists(default_base_path):
        with open(default_base_path, 'r', encoding='utf-8') as f:
            base_dict = yaml.safe_load(f) or {}
            _deep_update(merged_dict, base_dict)

    # Peek at CLI args early to find requested dataset, model, or method overrides
    early_overrides = parse_cli_overrides(cli_args) if cli_args else {}

    # 2. Resolve and load Method config (sets algorithm defaults)
    req_method = (
        method_config or
        early_overrides.get('method', {}).get('name') or
        early_overrides.get('method')
    )
    if req_method:
        meth_path = _resolve_config_path('methods', str(req_method))
        if meth_path and os.path.exists(meth_path):
            with open(meth_path, 'r', encoding='utf-8') as f:
                _deep_update(merged_dict, yaml.safe_load(f) or {})

    # 3. Resolve and load Dataset config (sets dataset, classes, task, path)
    req_dataset = (
        dataset_config or
        early_overrides.get('data', {}).get('name') or
        early_overrides.get('dataset')
    )
    if req_dataset:
        d_path = _resolve_config_path('datasets', str(req_dataset))
        if d_path and os.path.exists(d_path):
            with open(d_path, 'r', encoding='utf-8') as f:
                _deep_update(merged_dict, yaml.safe_load(f) or {})

    # 4. Resolve and load Model config (explicit model takes precedence over method's default model)
    req_model = (
        model_config or
        early_overrides.get('model', {}).get('name') or
        early_overrides.get('model')
    )
    if req_model:
        m_path = _resolve_config_path('models', str(req_model))
        if m_path and os.path.exists(m_path):
            with open(m_path, 'r', encoding='utf-8') as f:
                _deep_update(merged_dict, yaml.safe_load(f) or {})

    # 5. Load specific experiment config if given
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            exp_dict = yaml.safe_load(f) or {}
            _deep_update(merged_dict, exp_dict)

    # 6. Apply final CLI overrides (highest precedence)
    if cli_args is not None:
        overrides = parse_cli_overrides(cli_args)
        _deep_update(merged_dict, overrides)

    cfg = Config(merged_dict)
    return cfg
