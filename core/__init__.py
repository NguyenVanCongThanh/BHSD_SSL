"""
Core package for MedSSL-BHSD framework.
Contains Registry system and YAML Configuration Parser with CLI overrides.
"""

from core.registry import Registry, MODELS, METHODS, DATASETS, LOSSES
from core.config_parser import load_config, Config
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
    "is_dist_avail_and_initialized",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "init_distributed_mode",
    "cleanup_distributed",
    "reduce_tensor"
]
