"""
Core package for MedSSL-BHSD framework.
Contains Registry system and YAML Configuration Parser with CLI overrides.
"""

from core.registry import Registry, MODELS, METHODS, DATASETS, LOSSES
from core.config_parser import load_config, Config

__all__ = [
    "Registry",
    "MODELS",
    "METHODS",
    "DATASETS",
    "LOSSES",
    "load_config",
    "Config",
]
