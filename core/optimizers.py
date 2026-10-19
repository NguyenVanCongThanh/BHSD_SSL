"""
Modular Optimizer and Learning Rate Scheduler Factory for MedSSL-BHSD Framework.
Supports SGD, Adam, AdamW and schedulers (Polynomial decay, Cosine Annealing, Warmup).
"""

import math
import logging
from typing import Dict, Any, Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(model: nn.Module, cfg) -> optim.Optimizer:
    """
    Build optimizer from configuration.
    
    Supported types in cfg.method:
        - 'sgd': Standard SGD with momentum and weight decay
        - 'adam': Adam optimizer
        - 'adamw': Decoupled Weight Decay AdamW (recommended for Transformers like UNETR)
    """
    method_cfg = getattr(cfg, 'method', cfg)
    opt_type = str(getattr(method_cfg, 'optimizer', 'sgd')).lower()
    base_lr = float(getattr(method_cfg, 'base_lr', 0.01))
    weight_decay = float(getattr(method_cfg, 'weight_decay', 0.0001))
    momentum = float(getattr(method_cfg, 'momentum', 0.9))

    # Collect parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if opt_type == 'sgd':
        optimizer = optim.SGD(
            trainable_params,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=bool(getattr(method_cfg, 'nesterov', False))
        )
    elif opt_type == 'adam':
        betas = tuple(getattr(method_cfg, 'betas', (0.9, 0.999)))
        optimizer = optim.Adam(
            trainable_params,
            lr=base_lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=float(getattr(method_cfg, 'eps', 1e-8))
        )
    elif opt_type == 'adamw':
        betas = tuple(getattr(method_cfg, 'betas', (0.9, 0.999)))
        optimizer = optim.AdamW(
            trainable_params,
            lr=base_lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=float(getattr(method_cfg, 'eps', 1e-8))
        )
    else:
        logging.warning(f"[Optimizer] Unknown optimizer '{opt_type}', falling back to SGD.")
        optimizer = optim.SGD(trainable_params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer


def calculate_lr(
    iter_num: int,
    max_iterations: int,
    base_lr: float,
    scheduler_type: str = 'poly',
    poly_power: float = 0.9,
    min_lr: float = 1e-6,
    warmup_iters: int = 0
) -> float:
    """Calculate learning rate at current iteration with optional warmup."""
    max_iters = max(max_iterations, 1)

    # Linear Warmup phase
    if warmup_iters > 0 and iter_num < warmup_iters:
        alpha = float(iter_num) / float(max(warmup_iters, 1))
        return max(base_lr * alpha, min_lr)

    eff_iter = iter_num - warmup_iters
    eff_max = max(max_iters - warmup_iters, 1)

    sched = str(scheduler_type).lower()
    if sched == 'poly':
        decay = max(1.0 - (eff_iter / eff_max), 0.0) ** poly_power
        return max(base_lr * decay, min_lr)
    elif sched in ('cosine', 'cosine_annealing'):
        progress = min(max(eff_iter / eff_max, 0.0), 1.0)
        return max(base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)), min_lr)
    elif sched == 'constant':
        return base_lr
    else:
        # Default to poly
        decay = max(1.0 - (eff_iter / eff_max), 0.0) ** poly_power
        return max(base_lr * decay, min_lr)
