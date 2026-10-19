"""
Core package for MedSSL-BHSD framework.
Contains Registry system, YAML Configuration Parser with CLI overrides,
Distributed training utilities, and Optimizer/Scheduler factories.
"""

from core.registry import Registry, MODELS, METHODS, DATASETS, LOSSES
from core.config_parser import load_config, Config
from core.optimizers import build_optimizer, calculate_lr
from core.distributed import (
    is_dist_avail_and_initialized,
    get_world_size,
    get_rank,
    is_main_process,
    init_distributed_mode,
    cleanup_distributed,
    reduce_tensor
)

__all__ = [
    "Registry",
    "MODELS",
    "METHODS",
    "DATASETS",
    "LOSSES",
    "load_config",
    "Config",
    "build_optimizer",
    "calculate_lr",
    "is_dist_avail_and_initialized",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "init_distributed_mode",
    "cleanup_distributed",
    "reduce_tensor",
]
