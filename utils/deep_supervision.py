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


def extract_multiscale_outputs(
    net_out: Union[torch.Tensor, Dict[str, Any], Tuple, List]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Unpack network predictions into (main_logits, aux_logits_list).
    Supports:
      - Pure Tensor: (logits, [])
      - Dict: {'seg': main_logits, 'aux': [aux1, aux2, ...]}
      - Tuple/List:
          - If PICK format: (proj, main, feat) -> main is net_out[1]
          - Otherwise: net_out[0] is main, net_out[1:] are aux
    """
    if isinstance(net_out, dict):
        main = net_out.get('seg', net_out.get('out', None))
        if main is None:
            # Fallback to first value
            keys = list(net_out.keys())
            main = net_out[keys[0]]
            aux = [v for k, v in net_out.items() if k != keys[0] and isinstance(v, torch.Tensor)]
        else:
            aux = net_out.get('aux', [])
            if isinstance(aux, torch.Tensor):
                aux = [aux]
        return main, list(aux)

    if isinstance(net_out, (tuple, list)):
        # Check if PICK-style (proj, out_seg, feat)
        if len(net_out) >= 2 and isinstance(net_out[1], torch.Tensor) and net_out[1].ndim >= 4:
            # Check if this is PICK format where net_out[0] is 2D projection/feat
            if isinstance(net_out[0], torch.Tensor) and net_out[0].ndim == 2:
                return net_out[1], []
        # Multi-scale list format: [scale_0, scale_1, scale_2, ...]
        return net_out[0], list(net_out[1:])

    # Single tensor
    return net_out, []


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

        # Compute weights if not provided or length mismatch
        if self.weights is None or len(self.weights) != num_scales:
            weights = compute_ds_weights(num_scales)
        else:
            total_w = sum(self.weights)
            weights = [w / total_w for w in self.weights]

        total_loss = torch.tensor(0.0, device=main_logits.device, dtype=main_logits.dtype)
        scale_losses = []
        target_shape = target.shape[1:]  # (D, H, W)

        for idx, (logits, w) in enumerate(zip(all_logits, weights)):
            if w <= 0.0:
                continue

            if self.upsample_logits and logits.shape[2:] != target_shape:
                logits_up = F.interpolate(logits, size=target_shape, mode='trilinear', align_corners=False)
            else:
                logits_up = logits

            l_stage = self.base_criterion(logits_up, target)
            total_loss = total_loss + w * l_stage
            scale_losses.append(float(l_stage.item()))

        return total_loss, scale_losses
