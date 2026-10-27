"""
Deep Supervision Loss & Utilities for 3D Volumetric Segmentation.
Follows nnU-Net design principles (Isensee et al., 2021) with exponential decaying scale weights.
Provides unified extraction, interpolation, and loss calculation for both Supervised and SSL consistency branches.
"""

from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ds_weights(num_scales: int) -> List[float]:
    """
    Compute normalized exponential decay weights according to nnU-Net rule:
    w_i = 2^(-i) / sum(2^(-k))
    Example for 4 scales (scale 1.0, 0.5, 0.25, 0.125):
    [0.5333, 0.2667, 0.1333, 0.0667]
    """
    if num_scales <= 1:
        return [1.0]
    raw = [1.0 / (2 ** i) for i in range(num_scales)]
    total = sum(raw)
    return [w / total for w in raw]


def resolve_ds_weights(num_scales, weights=None):
    """Configured weights cover main plus every auxiliary head; never silently ignored."""
    if weights is None:
        return compute_ds_weights(num_scales)
    import math
    if len(weights) != num_scales or not all(math.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f'Deep supervision requires {num_scales} finite nonnegative weights (main + aux) with positive sum')
    return [float(w / sum(weights)) for w in weights]


NON_AUX_KEYS = {
    'features', 'reconstruction', 'raw', 'metadata', 'mask', 'latent', 'loss',
    'weight', 'weights', 'loss_dict', 'decoder_features', 'encoder_features', 'feedback', 'logits2'
}


def extract_multiscale_outputs(
    net_out: Union[torch.Tensor, Dict[str, Any], Tuple, List]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Unpack network predictions into (main_logits, aux_logits_list).
    Supports:
      - Pure Tensor: (logits, [])
      - Dict: {'seg'|'logits'|'out'|'main': main_logits, 'aux'|'aux_logits': [aux1, aux2, ...]}
      - Tuple/List:
          - If PICK format: (proj, main, feat) -> main is net_out[1]
          - If LeFeD format: (out1, out2, masks, deep_outs) -> main is out1, aux is deep_outs
          - Otherwise: net_out[0] is main, net_out[1:] are aux
    """
    if isinstance(net_out, torch.Tensor):
        return net_out, []

    if isinstance(net_out, dict):
        if not net_out:
            raise ValueError("Empty dictionary received for multiscale output extraction")

        # 1. Locate primary logits
        main = None
        main_key = None
        for k in ('logits', 'seg', 'out', 'main', 'custom_logits', 'output'):
            if k in net_out and isinstance(net_out[k], torch.Tensor):
                main = net_out[k]
                main_key = k
                break

        if main is None:
            # Fallback to first tensor whose key is not non-aux
            for k, v in net_out.items():
                if isinstance(v, torch.Tensor) and k.lower() not in NON_AUX_KEYS:
                    main = v
                    main_key = k
                    break
            if main is None:
                first_k = next(iter(net_out.keys()))
                main = net_out[first_k]
                main_key = first_k

        if not isinstance(main, torch.Tensor):
            raise TypeError(f"Primary output must be a torch.Tensor, got {type(main)}")

        # 2. Extract auxiliary logits
        aux_candidates = None
        for aux_key in ('aux', 'aux_logits', 'auxiliary'):
            if aux_key in net_out and aux_key != main_key:
                aux_candidates = net_out[aux_key]
                break

        aux: List[torch.Tensor] = []
        if aux_candidates is not None:
            if isinstance(aux_candidates, torch.Tensor):
                aux = [aux_candidates]
            elif isinstance(aux_candidates, (list, tuple)):
                aux = list(aux_candidates)
            else:
                raise TypeError(f"Auxiliary output must be a Tensor or list/tuple of Tensors, got {type(aux_candidates)}")
        else:
            # Fallback: strictly exclude known non-aux fields (features, reconstruction, etc.)
            for k, v in net_out.items():
                if k == main_key or k.lower() in NON_AUX_KEYS:
                    continue
                if isinstance(v, torch.Tensor):
                    aux.append(v)
                elif isinstance(v, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in v):
                    aux.extend(v)

        for i, a in enumerate(aux):
            if not isinstance(a, torch.Tensor):
                raise TypeError(f"Auxiliary output element at index {i} must be a torch.Tensor, got {type(a)}")

        return main, aux

    if isinstance(net_out, (tuple, list)):
        if len(net_out) == 0:
            raise ValueError("Empty sequence received for multiscale output extraction")

        # Check LeFeD: (out1, out2, feedback_masks, deep_out1)
        if (len(net_out) == 4 and isinstance(net_out[0], torch.Tensor) and isinstance(net_out[1], torch.Tensor)
                and isinstance(net_out[2], (list, tuple)) and isinstance(net_out[3], (list, tuple))):
            return net_out[0], list(net_out[3])

        # Check if PICK-style (proj, out_seg, feat)
        first = net_out[0]
        is_pick = isinstance(first, (list, tuple)) or (isinstance(first, torch.Tensor) and first.ndim == 2)
        if is_pick and len(net_out) >= 2 and isinstance(net_out[1], torch.Tensor) and net_out[1].ndim >= 4:
            return net_out[1], []

        main = net_out[0]
        if not isinstance(main, torch.Tensor):
            raise TypeError(f"Primary output in sequence must be a torch.Tensor, got {type(main)}")

        aux = list(net_out[1:])
        for i, a in enumerate(aux):
            if not isinstance(a, torch.Tensor):
                raise TypeError(f"Auxiliary output element at index {i} must be a torch.Tensor, got {type(a)}")
        return main, aux

    raise TypeError(f"Unsupported model output type for multiscale extraction: {type(net_out)}")


from core.registry import LOSSES


@LOSSES.register("deep_supervision")
class DeepSupervisionLoss(nn.Module):
    """
    Multi-scale supervised loss engine with trilinear interpolation and normalized weights.
    """
    def __init__(
        self,
        base_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        weights: Optional[List[float]] = None,
        upsample_logits: bool = True
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.weights = weights
        self.upsample_logits = upsample_logits

    def forward(
        self,
        main_logits: torch.Tensor,
        aux_logits_list: List[torch.Tensor],
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Compute weighted multi-scale supervised loss.
        Returns total_loss and per-scale individual loss values.
        """
        all_logits = [main_logits] + list(aux_logits_list)
        num_scales = len(all_logits)
        weights = resolve_ds_weights(num_scales, self.weights)

        total_loss = torch.tensor(0.0, device=main_logits.device, dtype=main_logits.dtype)
        scale_losses = []
        # Spatial dimensions (D, H, W) or (H, W)
        target_spatial = target.shape[2:] if target.dim() == main_logits.dim() else target.shape[1:]

        for idx, (logits, w) in enumerate(zip(all_logits, weights)):
            if w <= 0.0:
                continue

            if self.upsample_logits and logits.shape[2:] != target_spatial:
                logits_up = F.interpolate(logits, size=target_spatial, mode='trilinear', align_corners=False)
            else:
                logits_up = logits

            l_stage = self.base_criterion(logits_up, target)
            total_loss = total_loss + w * l_stage
            scale_losses.append(float(l_stage.item()))

        return total_loss, scale_losses
