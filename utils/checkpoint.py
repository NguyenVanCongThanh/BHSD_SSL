"""
Centralized Checkpoint Management for MedSSL-BHSD.
Provides standardized saving, loading, parameter stripping (module. prefix for DDP/DP),
and full state serialization across training, evaluation, and benchmarking.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn


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
    }
    if scaler is not None and hasattr(scaler, 'state_dict'):
        state['scaler'] = scaler.state_dict()
    if extra_state:
        state.update(extra_state)

    torch.save(state, str(path))
    return str(path)


def load_checkpoint(
    path: Union[str, os.PathLike],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = False,
    prefer_ema: bool = False
) -> Dict[str, Any]:
    """Load checkpoint weights into model and optionally restore optimizer/scaler state.

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
    state = torch.load(str(path), map_location=map_location)

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

    # Restore optimizer
    if optimizer is not None and 'opt' in state and state['opt'] is not None:
        try:
            optimizer.load_state_dict(state['opt'])
        except Exception as e:
            logging.warning(f"[Checkpoint] Could not restore optimizer state: {e}")

    # Restore scaler
    if scaler is not None and 'scaler' in state and state['scaler'] is not None and hasattr(scaler, 'load_state_dict'):
        try:
            scaler.load_state_dict(state['scaler'])
        except Exception as e:
            logging.warning(f"[Checkpoint] Could not restore scaler state: {e}")

    return state
