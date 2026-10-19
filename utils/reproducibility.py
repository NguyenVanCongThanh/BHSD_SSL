"""
Reproducibility utilities for MedSSL-BHSD.
Provides deterministic random seed configuration across Python, NumPy, PyTorch, and CuDNN.
Supports DDP distributed rank-offset seeding to prevent duplicate data augmentations across GPUs.
"""

import os
import random
import logging
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 1337, deterministic: bool = True, rank: int = 0) -> int:
    """Set random seed across all libraries to ensure full reproducibility.

    Args:
        seed: Base random seed integer.
        deterministic: If True, configures CuDNN backends to be deterministic.
                       Note: True deterministic mode may slightly decrease training speed.
        rank: Distributed process rank (0 for single-GPU/main process).
              Offsetting seed by rank ensures each GPU worker gets unique data augmentations.

    Returns:
        int: The actual seed applied (seed + rank).
    """
    actual_seed = seed + rank

    random.seed(actual_seed)
    np.random.seed(actual_seed)
    os.environ['PYTHONHASHSEED'] = str(actual_seed)

    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual_seed)
        torch.cuda.manual_seed_all(actual_seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    logging.debug(f"[Reproducibility] Random seed set to {actual_seed} (deterministic={deterministic}, rank={rank})")
    return actual_seed
