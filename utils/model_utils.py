"""
Common Model and Tensor Output Utilities for MedSSL-BHSD.
Provides robust extraction of logits across single-head, multi-head, and composite networks.
"""

from typing import Any
import torch


def extract_logits(model_output: Any) -> torch.Tensor:
    """
    Safely extract primary segmentation logits from diverse model architectures:
    - Standard Net: Tensor of shape (B, C, D, H, W)
    - Multi-Head / PICK Composite: Tuple/List -> typically out[1] for main_outputs
    - Dictionary Output: dict with 'seg', 'logits', or 'out'
    - Deep Supervision: List/Tuple where element 0 or 1 contains primary logits
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, (tuple, list)):
        # PICK returns (encoder_features, segmentation, decoder_features).
        first = model_output[0] if model_output else None
        is_pick = isinstance(first, (list, tuple)) or (isinstance(first, torch.Tensor) and first.ndim == 2)
        if is_pick and len(model_output) >= 2 and isinstance(model_output[1], torch.Tensor):
            return model_output[1]
        # Deep supervision lists are ordered main/highest resolution first.
        if isinstance(first, torch.Tensor):
            return first
        raise ValueError(f'Unable to extract logits from sequence: {[type(x) for x in model_output]}')

    if isinstance(model_output, dict):
        for key in ('seg', 'logits', 'out', 'main'):
            if key in model_output and isinstance(model_output[key], torch.Tensor):
                return model_output[key]
        first_val = next(iter(model_output.values()), None)
        if isinstance(first_val, torch.Tensor):
            return first_val

    raise TypeError(f"Unsupported model output type for logit extraction: {type(model_output)}")
