"""
Hierarchical YAML Configuration Manager with Dot-Access and Composable Overrides.
Enables clean, reproducible, and parameterized experiments without messy argparse boilerplate.
Supports composable 3-tier configurations: Datasets + Models + Methods.
"""

from typing import Any, Dict, List, Optional, Union
import os
import copy
import math
import yaml


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
    'intensity_units': 'data',
    'normalization': 'data',
    
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
    'ddp': 'experiment',

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
    'wandb_mode': 'logging',
    'wandb_required': 'logging',
    'artifact_interval': 'logging',
    'log_interval': 'logging',
    'val_interval': 'logging',
    'save_interval': 'logging',
    'val_stride_xy': 'logging',
    'val_stride_z': 'logging',
    'val_amp': 'logging',
    'val_gaussian': 'logging',
    'val_surface_metrics': 'logging',
    'val_tta': 'logging',
    'val_visualize': 'logging',
    'num_vis_samples': 'logging',
    'visual_every_n_validations': 'logging',
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


KNOWN_SECTION_KEYS = {
    'experiment': {
        'name', 'task', 'num_classes', 'fold', 'seed', 'deterministic', 'exp_dir',
        'resume', 'amp', 'phase', 'pretrain_checkpoint', 'debug', 'test_mode', 'ddp'
    },
    'data': {
        'name', 'root_path', 'patch_size', 'batch_size', 'labeled_bs', 'labelnum',
        'max_samples', 'intensity_units', 'normalization', 'num_classes', 'task',
        'class_names', 'num_workers', 'pin_memory', 'drop_last', 'train_split',
        'val_split', 'test_split'
    },
    'method': {
        'name', 'base_lr', 'momentum', 'weight_decay', 'lr_scheduler', 'poly_power',
        'max_iteration', 'pre_max_iteration', 'self_max_iteration', 'consistency',
        'consistency_rampup', 'consistency_rampdown_start', 'consistency_min_factor',
        'consistency_type', 'bg_consistency_weight', 'confidence_thresh',
        'perturb_student', 'student_noise_std', 'student_intensity_scale',
        'deep_supervision', 'lambda_', 'lambda_mim', 'mask_ratio', 'cps_weight',
        'ema_decay', 'ema_schedule', 'u_weight', 'temperature', 'threshold',
        'warmup_epochs', 'loss_name', 'aux_loss_weight', 'optimizer',
        'feedback_steps', 'feedback_scale', 'ds_weights'
    },
    'logging': {
        'use_wandb', 'save_wandb_model', 'wandb_project', 'wandb_entity', 'wandb_run_name',
        'wandb_id', 'wandb_mode', 'wandb_required', 'val_interval', 'val_stride_xy',
        'val_stride_z', 'log_interval', 'save_interval', 'artifact_interval',
        'val_amp', 'val_gaussian', 'val_surface_metrics', 'val_tta', 'val_visualize',
        'num_vis_samples', 'visual_every_n_validations', 'visualization'
    }
}


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _canonical_name(val: Optional[Any]) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return os.path.splitext(os.path.basename(s))[0].lower()


def validate_config(cfg_dict: Dict[str, Any]) -> None:
    """Validate structure, types, ranges, unknown keys, and cross-section invariants."""
    data = cfg_dict.get('data') or {}
    exp = cfg_dict.get('experiment') or {}
    method = cfg_dict.get('method') or {}

    # 0. Strict unknown key validation for known sections
    for sec, allowed in KNOWN_SECTION_KEYS.items():
        if isinstance(cfg_dict.get(sec), dict):
            unknown = set(cfg_dict[sec].keys()) - allowed
            if unknown:
                raise ValueError(f"Unknown parameter(s) in '{sec}': {sorted(list(unknown))}")

    # 1. Data invariants
    if 'batch_size' in data:
        bs = data['batch_size']
        if not _is_int(bs) or bs < 1:
            raise ValueError(f"data.batch_size must be positive integer, got {bs}")
    if 'labeled_bs' in data:
        lbs = data['labeled_bs']
        if not _is_int(lbs) or lbs < 1:
            raise ValueError(f"data.labeled_bs must be positive integer, got {lbs}")
        if 'batch_size' in data and lbs > data['batch_size']:
            raise ValueError(f"data.labeled_bs ({lbs}) cannot exceed data.batch_size ({data['batch_size']})")
    if 'labelnum' in data:
        ln = data['labelnum']
        if not _is_int(ln) or ln < 1:
            raise ValueError(f"data.labelnum must be positive integer, got {ln}")
    if 'max_samples' in data:
        ms = data['max_samples']
        if not _is_int(ms) or ms < 1:
            raise ValueError(f"data.max_samples must be positive integer, got {ms}")
    if 'patch_size' in data:
        ps = data['patch_size']
        if not (isinstance(ps, (list, tuple)) and len(ps) == 3 and all(_is_int(x) and x > 0 for x in ps)):
            raise ValueError(f"data.patch_size must be 3 positive integers (Z, Y, X), got {ps}")
    if 'num_classes' in data:
        nc = data['num_classes']
        if not _is_int(nc) or nc < 2:
            raise ValueError(f"data.num_classes must be integer >= 2, got {nc}")
        if 'class_names' in data and data['class_names'] is not None:
            cnames = data['class_names']
            if not isinstance(cnames, list) or len(cnames) != nc:
                raise ValueError(
                    f"len(data.class_names) ({len(cnames) if isinstance(cnames, list) else type(cnames)}) "
                    f"must match data.num_classes ({nc})"
                )

    # 2. Experiment invariants
    if 'task' in exp:
        task = exp['task']
        if task not in ('binary', 'multiclass'):
            raise ValueError(f"experiment.task must be 'binary' or 'multiclass', got '{task}'")
        if 'task' in data and data['task'] != task:
            raise ValueError(f"Mismatch between data.task ('{data['task']}') and experiment.task ('{task}')")
    if 'num_classes' in exp and 'num_classes' in data:
        if exp['num_classes'] != data['num_classes']:
            raise ValueError(f"Mismatch between data.num_classes ({data['num_classes']}) and experiment.num_classes ({exp['num_classes']})")
    if 'fold' in exp:
        f = exp['fold']
        if not _is_int(f) or f < 0:
            raise ValueError(f"experiment.fold must be non-negative integer, got {f}")
    if 'seed' in exp:
        s = exp['seed']
        if not _is_int(s) or s < 0:
            raise ValueError(f"experiment.seed must be non-negative integer, got {s}")
    if 'ddp' in exp:
        if not isinstance(exp['ddp'], bool):
            raise ValueError(f"experiment.ddp must be boolean, got {exp['ddp']}")

    # 3. Method invariants
    if 'base_lr' in method:
        lr = method['base_lr']
        if not (_is_finite_number(lr) and lr > 0):
            raise ValueError(f"method.base_lr must be positive finite float, got {lr}")


def load_config(
    config_path: Optional[str] = None,
    dataset_config: Optional[str] = None,
    model_config: Optional[str] = None,
    method_config: Optional[str] = None,
    cli_args: Optional[List[str]] = None,
    default_base_path: Optional[str] = None
) -> Config:
    """
    Load YAML configuration file with validated multi-tier composition:
    Base Config -> Dataset Config -> Method Config -> Isolated Model Defaults & Overrides -> Exp Config -> CLI Overrides.
    """
    merged_dict: Dict[str, Any] = {}
    configs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))

    # Parse CLI args early
    early_overrides = parse_cli_overrides(cli_args) if cli_args else {}

    # Inspect config_path if provided
    exp_dict: Dict[str, Any] = {}
    if config_path:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Experiment config not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            exp_dict = yaml.safe_load(f) or {}

    # Resolve task override early to select correct dataset variant if applicable
    task_override = (
        early_overrides.get('data', {}).get('task') or
        early_overrides.get('experiment', {}).get('task') or
        early_overrides.get('task')
    )
    if task_override and task_override not in ('binary', 'multiclass'):
        raise ValueError("Task must be binary or multiclass")

    # Detect method selectors
    cfg_method = exp_dict.get('method', {}).get('name') if isinstance(exp_dict.get('method'), dict) else exp_dict.get('method')
    cli_method = early_overrides.get('method', {}).get('name') or (early_overrides.get('method') if isinstance(early_overrides.get('method'), str) else None)
    req_method = method_config or cli_method or cfg_method

    if cfg_method and (method_config or cli_method):
        specified = method_config or cli_method
        if _canonical_name(specified) != _canonical_name(cfg_method):
            raise ValueError(
                f"Conflicting method specification: config specifies '{cfg_method}' but method requested '{specified}'"
            )

    # Detect model selectors
    cfg_model = exp_dict.get('model', {}).get('name') if isinstance(exp_dict.get('model'), dict) else exp_dict.get('model')
    cli_model = early_overrides.get('model', {}).get('name') or (early_overrides.get('model') if isinstance(early_overrides.get('model'), str) else None)

    if cfg_model and (model_config or cli_model):
        specified = model_config or cli_model
        if _canonical_name(specified) != _canonical_name(cfg_model):
            raise ValueError(
                f"Conflicting model specification: config specifies '{cfg_model}' but model requested '{specified}'"
            )

    # Detect dataset selectors
    cfg_dataset = None
    if isinstance(exp_dict.get('data'), dict):
        cfg_dataset = exp_dict['data'].get('name')
    if not cfg_dataset:
        cfg_dataset = exp_dict.get('dataset') or (exp_dict.get('data') if isinstance(exp_dict.get('data'), str) else None)

    cli_dataset = None
    if isinstance(early_overrides.get('data'), dict):
        cli_dataset = early_overrides['data'].get('name')
    if not cli_dataset:
        cli_dataset = early_overrides.get('dataset') or (early_overrides.get('data') if isinstance(early_overrides.get('data'), str) else None)

    def _normalize_dataset_name(ds_name: Optional[Any], default_task: Optional[str] = None) -> Optional[str]:
        if not ds_name:
            return None
        c_name = _canonical_name(ds_name)
        if c_name == 'bhsd':
            t = task_override or default_task or (exp_dict.get('data', {}).get('task') if isinstance(exp_dict.get('data'), dict) else None) or exp_dict.get('experiment', {}).get('task') or 'multiclass'
            return f"bhsd_{t}"
        return c_name

    norm_cfg_ds = _normalize_dataset_name(cfg_dataset)
    norm_cli_or_arg_ds = _normalize_dataset_name(dataset_config or cli_dataset)

    if norm_cfg_ds and norm_cli_or_arg_ds:
        if norm_cfg_ds != norm_cli_or_arg_ds:
            raise ValueError(
                f"Conflicting dataset specification: config specifies '{cfg_dataset}' but dataset requested '{dataset_config or cli_dataset}'"
            )

    raw_dataset = dataset_config or cli_dataset or cfg_dataset
    req_dataset = _normalize_dataset_name(raw_dataset) or 'bhsd_multiclass'

    # 1. Base configuration
    if default_base_path is None:
        cand_base = os.path.join(configs_root, "base.yaml")
        if os.path.exists(cand_base):
            default_base_path = cand_base

    if default_base_path and os.path.exists(default_base_path):
        with open(default_base_path, 'r', encoding='utf-8') as f:
            base_dict = yaml.safe_load(f) or {}
            _deep_update(merged_dict, base_dict)

    # 2. Resolve and load Dataset config
    if task_override:
        if 'multiclass' in str(req_dataset) and task_override == 'binary':
            cand = str(req_dataset).replace('multiclass', 'binary')
            if _resolve_config_path('datasets', cand):
                req_dataset = cand
        elif 'binary' in str(req_dataset) and task_override == 'multiclass':
            cand = str(req_dataset).replace('binary', 'multiclass')
            if _resolve_config_path('datasets', cand):
                req_dataset = cand
        elif _resolve_config_path('datasets', f"{req_dataset}_{task_override}"):
            req_dataset = f"{req_dataset}_{task_override}"

    if req_dataset:
        d_path = _resolve_config_path('datasets', str(req_dataset))
        if d_path:
            with open(d_path, 'r', encoding='utf-8') as f:
                d_dict = yaml.safe_load(f) or {}
                _deep_update(merged_dict, d_dict)
        elif not (isinstance(exp_dict.get('data'), dict) and 'num_classes' in exp_dict['data']):
            raise FileNotFoundError(f"Dataset config not found: {req_dataset}")

    # 3. Resolve and load Method config (sets algorithm defaults and parameters)
    meth_dict: Dict[str, Any] = {}
    if req_method:
        meth_path = _resolve_config_path('methods', _canonical_name(req_method))
        if meth_path is None:
            raise FileNotFoundError(f"Method config not found: {req_method}")
        with open(meth_path, 'r', encoding='utf-8') as f:
            meth_dict = yaml.safe_load(f) or {}
            # Deep update everything except model (which is handled in isolated model composition)
            meth_copy = copy.deepcopy(meth_dict)
            meth_copy.pop('model', None)
            _deep_update(merged_dict, meth_copy)

    # 4. Resolve Isolated Model Configuration
    # Model precedence: CLI override > Preset/exp_dict > model_config arg > method default > base default
    meth_model_name = _canonical_name(meth_dict.get('model', {}).get('name') if isinstance(meth_dict.get('model'), dict) else meth_dict.get('model'))
    target_model_name = _canonical_name(cli_model or cfg_model or model_config or meth_model_name or merged_dict.get('model', {}).get('name', 'vnet'))

    in_ch = 1
    if isinstance(merged_dict.get('model'), dict):
        in_ch = merged_dict['model'].get('in_channels', 1)
    isolated_model: Dict[str, Any] = {'name': target_model_name, 'in_channels': in_ch}

    # Load defaults from configs/models/{target_model_name}.yaml if available
    m_path = _resolve_config_path('models', target_model_name)
    if m_path:
        with open(m_path, 'r', encoding='utf-8') as f:
            m_yaml = yaml.safe_load(f) or {}
            if isinstance(m_yaml.get('model'), dict):
                _deep_update(isolated_model, m_yaml['model'])
            elif isinstance(m_yaml, dict):
                _deep_update(isolated_model, m_yaml)

    # If method provides model defaults, merge ONLY if its model name matches target_model_name
    if meth_dict and isinstance(meth_dict.get('model'), dict):
        m_meth_name = _canonical_name(meth_dict['model'].get('name'))
        if not m_meth_name or m_meth_name == target_model_name:
            m_copy = copy.deepcopy(meth_dict['model'])
            m_copy['name'] = target_model_name
            _deep_update(isolated_model, m_copy)

    # If preset (exp_dict) specifies model, merge
    if isinstance(exp_dict.get('model'), dict):
        e_meth_name = _canonical_name(exp_dict['model'].get('name'))
        if not e_meth_name or e_meth_name == target_model_name:
            e_copy = copy.deepcopy(exp_dict['model'])
            e_copy['name'] = target_model_name
            _deep_update(isolated_model, e_copy)

    # If CLI provides model overrides, merge
    if isinstance(early_overrides.get('model'), dict):
        c_copy = copy.deepcopy(early_overrides['model'])
        if 'name' in c_copy:
            c_copy['name'] = target_model_name
        _deep_update(isolated_model, c_copy)

    merged_dict['model'] = isolated_model

    # 5. Load Experiment config (preset or custom config_path)
    if exp_dict:
        exp_copy = copy.deepcopy(exp_dict)
        exp_copy.pop('model', None)
        _deep_update(merged_dict, exp_copy)

    # 6. Apply final CLI overrides (highest precedence)
    if cli_args is not None:
        overrides = parse_cli_overrides(cli_args)
        overrides_copy = copy.deepcopy(overrides)
        overrides_copy.pop('model', None)
        _deep_update(merged_dict, overrides_copy)

    # Task & classes synchronization
    if task_override:
        merged_dict.setdefault('data', {})['task'] = task_override
        merged_dict.setdefault('experiment', {})['task'] = task_override
        if task_override == 'binary':
            merged_dict['data'].setdefault('num_classes', 2)
            if merged_dict['data'].get('class_names') is None or len(merged_dict['data']['class_names']) != 2:
                merged_dict['data']['class_names'] = ['background', 'hemorrhage']

    # Track whether user/CLI explicitly specified experiment.num_classes or experiment.task
    user_exp_nc = early_overrides.get('experiment', {}).get('num_classes') or exp_dict.get('experiment', {}).get('num_classes')
    user_exp_task = (
        early_overrides.get('experiment', {}).get('task') or
        early_overrides.get('task') or
        exp_dict.get('experiment', {}).get('task')
    )

    # Synchronize dataset properties to experiment if not explicitly overridden by user
    if 'data' in merged_dict and 'num_classes' in merged_dict['data']:
        if user_exp_nc is None:
            merged_dict.setdefault('experiment', {})['num_classes'] = merged_dict['data']['num_classes']
    if 'data' in merged_dict and 'task' in merged_dict['data']:
        if user_exp_task is None:
            merged_dict.setdefault('experiment', {})['task'] = merged_dict['data']['task']

    # Schema & value range validation
    validate_config(merged_dict)

    cfg = Config(merged_dict)
    return cfg
