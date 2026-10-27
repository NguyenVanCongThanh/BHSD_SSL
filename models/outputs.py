"""
Standard Model Output Schema and Adapters for MedSSL-BHSD.
Provides a unified representation of network predictions across single-head,
multi-head, deep supervision, and composite SSL architectures (PICK, LeFeD, CPS).
Guarantees compatibility with dict access, DataParallel gather, DDP autograd,
and all downstream loss functions.
"""

from typing import Optional, List, Dict, Any, Union
import torch


NON_AUX_KEYS = {
    'features', 'reconstruction', 'raw', 'metadata', 'mask', 'latent', 'loss',
    'weight', 'weights', 'loss_dict', 'decoder_features', 'encoder_features', 'feedback', 'logits2'
}


class ModelOutput(dict):
    """
    Standard container for model predictions, features, and auxiliary branches.
    Inherits from dict to ensure full compatibility with PyTorch DataParallel gather,
    torch serialization, and dict access. Provides attribute-style access for convenience.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def logits(self) -> Optional[torch.Tensor]:
        for k in ('logits', 'seg', 'out', 'main', 'custom_logits', 'output'):
            if k in self and isinstance(self[k], torch.Tensor):
                return self[k]
        return None

    @logits.setter
    def logits(self, val: torch.Tensor) -> None:
        self['logits'] = val

    @property
    def aux(self) -> List[torch.Tensor]:
        res = self.get('aux', self.get('aux_logits', []))
        return [res] if isinstance(res, torch.Tensor) else list(res)

    @aux.setter
    def aux(self, val: List[torch.Tensor]) -> None:
        self['aux'] = val

    @property
    def features(self) -> Any:
        return self.get('features')

    @features.setter
    def features(self, val: Any) -> None:
        self['features'] = val

    @property
    def reconstruction(self) -> Optional[torch.Tensor]:
        return self.get('reconstruction')

    @reconstruction.setter
    def reconstruction(self, val: Optional[torch.Tensor]) -> None:
        self['reconstruction'] = val

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ModelOutput' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def wrap_model_output(raw_out: Any) -> ModelOutput:
    """
    Adapter that converts any legacy network output (Tensor, Tuple, List, Dict)
    into a standardized ModelOutput without requiring changes to model code or state_dict keys.
    """
    if isinstance(raw_out, ModelOutput):
        return raw_out

    if isinstance(raw_out, torch.Tensor):
        return ModelOutput(logits=raw_out, aux=[])

    if isinstance(raw_out, dict):
        if not raw_out:
            raise ValueError("Empty dictionary received for wrap_model_output")
        main = None
        main_key = None
        for k in ('logits', 'seg', 'out', 'main', 'custom_logits', 'output'):
            if k in raw_out and isinstance(raw_out[k], torch.Tensor):
                main = raw_out[k]
                main_key = k
                break
        if main is None:
            # Search for any tensor not in NON_AUX_KEYS
            for k, v in raw_out.items():
                if isinstance(v, torch.Tensor) and k.lower() not in NON_AUX_KEYS and k.lower() not in ('aux', 'aux_logits'):
                    main = v
                    main_key = k
                    break
        if main is None:
            raise TypeError(f"Dictionary contains no primary tensor logits: {list(raw_out.keys())}")

        aux = raw_out.get('aux', raw_out.get('aux_logits', []))
        if isinstance(aux, torch.Tensor):
            aux = [aux]
        extra = {k: v for k, v in raw_out.items() if k not in (main_key, 'logits', 'aux', 'aux_logits')}
        return ModelOutput(logits=main, aux=list(aux), **extra)

    if isinstance(raw_out, (tuple, list)):
        if len(raw_out) == 0:
            raise ValueError("Empty sequence received for wrap_model_output")

        # Check LeFeD: (out1, out2, feedback_masks, deep_out1)
        if (len(raw_out) == 4 and isinstance(raw_out[0], torch.Tensor) and isinstance(raw_out[1], torch.Tensor)
                and isinstance(raw_out[2], (list, tuple)) and isinstance(raw_out[3], (list, tuple))):
            return ModelOutput(
                logits=raw_out[0],
                logits2=raw_out[1],
                feedback=raw_out[2],
                aux=list(raw_out[3])
            )

        first = raw_out[0]
        # Check PICK-style (proj, seg, feat_de)
        is_pick = isinstance(first, (list, tuple)) or (isinstance(first, torch.Tensor) and first.ndim == 2)
        if is_pick and len(raw_out) >= 2 and isinstance(raw_out[1], torch.Tensor) and raw_out[1].ndim >= 4:
            return ModelOutput(
                logits=raw_out[1],
                aux=[],
                features=first,
                decoder_features=raw_out[2] if len(raw_out) > 2 else None
            )

        # Deep supervision multi-scale list [scale_0, scale_1, ...]
        if isinstance(first, torch.Tensor):
            return ModelOutput(logits=first, aux=list(raw_out[1:]))

    raise TypeError(f"Unsupported model output type: {type(raw_out)}")


def extract_logits(model_output: Any) -> torch.Tensor:
    """
    Safely extract primary segmentation logits from diverse model architectures:
    - Standard Net: Tensor of shape (B, C, D, H, W)
    - Multi-Head / PICK Composite: Tuple/List -> typically out[1] for main_outputs
    - LeFeD: Tuple/List -> out[0] for main_outputs
    - Dictionary Output / ModelOutput: dict with 'seg', 'logits', 'out', 'main', 'custom_logits', etc.
    - Deep Supervision: List/Tuple where element 0 contains primary logits
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, ModelOutput):
        out = model_output.logits
        if out is not None:
            return out

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


