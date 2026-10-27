"""
Centralized Checkpoint Management for MedSSL-BHSD.
Provides standardized saving, loading, parameter stripping (module. prefix for DDP/DP),
and full state serialization across training, evaluation, and benchmarking.
"""

import os
import logging
import json
import math
import random
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def capture_rng_state() -> Dict[str, Any]:
    """Capture full RNG state across Python, NumPy, and PyTorch (CPU + CUDA) in weights_only=True compatible format."""
    np_s = np.random.get_state()
    return {
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'python_rng': random.getstate(),
        'numpy_rng_str': str(np_s[0]),
        'numpy_rng_keys': torch.from_numpy(np_s[1].copy()),
        'numpy_rng_pos': int(np_s[2]),
        'numpy_rng_has_gauss': int(np_s[3]),
        'numpy_rng_cached_gaussian': float(np_s[4]),
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state from dictionary captured by capture_rng_state."""
    if not isinstance(state, dict):
        return
    if 'torch_rng' in state and state['torch_rng'] is not None:
        torch_rng = state['torch_rng']
        if isinstance(torch_rng, torch.Tensor):
            torch_rng = torch_rng.cpu().to(torch.uint8)
        torch.set_rng_state(torch_rng)
    if 'cuda_rng' in state and state['cuda_rng'] and torch.cuda.is_available():
        try:
            cuda_rng = [t.cpu().to(torch.uint8) if isinstance(t, torch.Tensor) else t for t in state['cuda_rng']]
            if len(cuda_rng) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_rng)
            elif torch.cuda.current_device() < len(cuda_rng):
                torch.cuda.set_rng_state(cuda_rng[torch.cuda.current_device()])
        except Exception as e:
            logging.warning("[Checkpoint] Could not restore CUDA RNG state: %s", e)
    if 'python_rng' in state and state['python_rng'] is not None:
        random.setstate(state['python_rng'])
    if 'numpy_rng_str' in state and 'numpy_rng_keys' in state:
        keys_tensor = state['numpy_rng_keys']
        keys_arr = keys_tensor.cpu().numpy() if isinstance(keys_tensor, torch.Tensor) else np.array(keys_tensor, dtype=np.uint32)
        np_state = (
            str(state['numpy_rng_str']),
            keys_arr,
            int(state.get('numpy_rng_pos', 0)),
            int(state.get('numpy_rng_has_gauss', 0)),
            float(state.get('numpy_rng_cached_gaussian', 0.0))
        )
        np.random.set_state(np_state)


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip 'module.' prefix from state_dict keys created by DataParallel or DDP."""
    clean_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_dict[k[7:]] = v
        else:
            clean_dict[k] = v
    return clean_dict


def get_raw_model(model: nn.Module) -> nn.Module:
    """Unwrap model from DataParallel or DistributedDataParallel container."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def save_checkpoint(
    path: Union[str, os.PathLike],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    iter_num: int = 0,
    best_dice: float = 0.0,
    phase: str = 'train',
    extra_state: Optional[Dict[str, Any]] = None
) -> str:
    """Save model and training state to a checkpoint file.

    Args:
        path: Path to target .pth checkpoint file.
        model: Model instance (can be wrapped in DP or DDP).
        optimizer: Optimizer instance (optional).
        scaler: GradScaler for mixed precision (optional).
        iter_num: Current training iteration step.
        best_dice: Best validation metric achieved.
        phase: Training phase tag (e.g. 'train', 'self_train').
        extra_state: Optional dictionary containing additional metadata or configs.

    Returns:
        str: Absolute path of saved checkpoint.
    """
    raw_model = get_raw_model(model)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    state = {
        'net': raw_model.state_dict(),
        'opt': optimizer.state_dict() if optimizer is not None else None,
        'iter_num': iter_num,
        'best_dice': best_dice,
        'phase': phase,
        'checkpoint_kind': 'latest' if Path(path).name == 'checkpoint_latest.pth' else 'unverified',
        'rng_state': capture_rng_state(),
    }
    if scaler is not None and hasattr(scaler, 'state_dict'):
        state['scaler'] = scaler.state_dict()
    if extra_state:
        state.update(extra_state)

    temporary = str(path) + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)
    # Paths are relative to this phase directory. Publish only after checkpoint save.
    manifest_path = Path(path).parent / 'checkpoint_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema_version': 1, 'checkpoints': {}}
    manifest['checkpoints'][Path(path).name] = {
        'path': Path(path).name,
        'kind': state['checkpoint_kind'],
        'phase': phase,
        'iter_num': iter_num,
        'best_dice': best_dice if math.isfinite(best_dice) else None,
        'selection_metric': state.get('selection_metric'),
        'selected_iter': state.get('selected_iter'),
    }
    manifest_tmp = str(manifest_path) + '.tmp'
    with open(manifest_tmp, 'w') as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(manifest_tmp, manifest_path)
    return str(path)


def load_checkpoint(
    path: Union[str, os.PathLike],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = False,
    prefer_ema: bool = False,
    checkpoint_state: Optional[Dict[str, Any]] = None,
    restore_rng: bool = False
) -> Dict[str, Any]:
    """Load checkpoint weights into model and optionally restore optimizer/scaler/RNG state.

    Args:
        path: Path to .pth checkpoint file.
        model: Target model instance.
        optimizer: Optimizer instance to restore state into (optional).
        scaler: GradScaler instance to restore state into (optional).
        device: Device to map tensors onto (e.g. 'cuda:0', 'cpu').
        strict: Whether to strictly enforce state_dict key matching.
        prefer_ema: If True and 'ema_net' exists in checkpoint, loads 'ema_net' instead of 'net'.

    Returns:
        Dict[str, Any]: The loaded checkpoint dictionary containing metadata (iter_num, best_dice, etc.).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    map_location = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    state = checkpoint_state if checkpoint_state is not None else torch.load(str(path), map_location=map_location, weights_only=True)

    # Resolve state_dict for model weights
    net_dict = None
    if prefer_ema and 'ema_net' in state:
        net_dict = state['ema_net']
        logging.info("[Checkpoint] Loaded EMA teacher weights ('ema_net')")
    elif 'net' in state:
        net_dict = state['net']
    elif isinstance(state, dict):
        net_dict = state

    if net_dict is not None:
        clean_dict = strip_module_prefix(net_dict)
        raw_model = get_raw_model(model)
        missing, unexpected = raw_model.load_state_dict(clean_dict, strict=strict)
        if missing or unexpected:
            logging.debug(f"[Checkpoint] Model load note: missing={len(missing)}, unexpected={len(unexpected)}")

    # Requested training state must restore completely; partial resume is an error.
    if optimizer is not None:
        if state.get('opt') is None:
            raise ValueError('Checkpoint has no optimizer state; use weights-only transfer instead of resume')
        optimizer.load_state_dict(state['opt'])
    if scaler is not None and state.get('scaler') is not None:
        scaler.load_state_dict(state['scaler'])
    if restore_rng and 'rng_state' in state and state['rng_state'] is not None:
        restore_rng_state(state['rng_state'])
    return state


def validate_checkpoint_context(state, *, fold=None, task=None, num_classes=None, dataset_name=None):
    """Reject metadata conflicts that can invalidate cross-validation results."""
    config = state.get('config') or {}
    data = config.get('data') or {}
    experiment = config.get('experiment') or {}
    pairs = {
        'fold': (fold, experiment.get('fold')),
        'task': (task, data.get('task', experiment.get('task'))),
        'num_classes': (num_classes, data.get('num_classes', experiment.get('num_classes'))),
        'dataset': (dataset_name, data.get('name')),
    }
    for key, (expected, actual) in pairs.items():
        if expected is not None and actual is not None and expected != actual:
            raise ValueError(f'Checkpoint {key} mismatch: requested {expected}, saved {actual}')
