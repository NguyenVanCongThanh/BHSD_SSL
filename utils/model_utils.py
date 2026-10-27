"""
Common Model and Tensor Output Utilities for MedSSL-BHSD.
Provides robust extraction of logits across single-head, multi-head, and composite networks.
"""

from typing import Any
import torch


NON_AUX_KEYS = {
    'features', 'reconstruction', 'raw', 'metadata', 'mask', 'latent', 'loss',
    'weight', 'weights', 'loss_dict', 'decoder_features', 'encoder_features', 'feedback'
}


def extract_logits(model_output: Any) -> torch.Tensor:
    """
    Safely extract primary segmentation logits from diverse model architectures:
    - Standard Net: Tensor of shape (B, C, D, H, W)
    - Multi-Head / PICK Composite: Tuple/List -> typically out[1] for main_outputs
    - LeFeD: Tuple/List -> out[0] for main_outputs
    - Dictionary Output: dict with 'seg', 'logits', 'out', 'main', 'custom_logits', etc.
    - Deep Supervision: List/Tuple where element 0 contains primary logits
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, (tuple, list)):
        if not model_output:
            raise ValueError("Sequence is empty")

        # LeFeD returns (out_seg1, out_seg2, feedback_masks, deep_out1)
        if (len(model_output) == 4 and isinstance(model_output[0], torch.Tensor)
                and isinstance(model_output[1], torch.Tensor)):
            return model_output[0]

        # PICK returns (encoder_features, segmentation, decoder_features).
        first = model_output[0]
        is_pick = isinstance(first, (list, tuple)) or (isinstance(first, torch.Tensor) and first.ndim == 2)
        if is_pick and len(model_output) >= 2 and isinstance(model_output[1], torch.Tensor):
            return model_output[1]

        # Deep supervision lists are ordered main/highest resolution first.
        if isinstance(first, torch.Tensor):
            return first
        raise ValueError(f'Unable to extract logits from sequence: {[type(x) for x in model_output]}')

    if isinstance(model_output, dict):
        if not model_output:
            raise ValueError("Empty dictionary received for logit extraction")
        for key in ('logits', 'seg', 'out', 'main', 'custom_logits', 'output'):
            if key in model_output and isinstance(model_output[key], torch.Tensor):
                return model_output[key]
        for k, v in model_output.items():
            if isinstance(v, torch.Tensor) and k.lower() not in NON_AUX_KEYS and k.lower() not in ('aux', 'aux_logits'):
                return v

    raise TypeError(f"Unsupported model output type for logit extraction: {type(model_output)}")
