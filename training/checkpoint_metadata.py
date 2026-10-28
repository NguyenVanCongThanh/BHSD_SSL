"""
Pure data helper to build standardized checkpoint metadata payloads for MedSSL-BHSD.
Does not call W&B SDK, touch disk, or serialize models.
"""

from typing import Dict, Any, Optional
from evaluation.metrics import METRIC_POLICY


def build_checkpoint_metadata(
    *,
    config_dict: Optional[Dict[str, Any]],
    wandb_identity: Optional[Dict[str, Any]] = None,
    validation_options: Optional[Dict[str, Any]] = None,
    successful_updates: int = 0,
    validation_details: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct standardized extra_state dictionary for checkpoint persistence.
    Preserves existing key names, presence conditions, and merge order.
    Does not mutate input dictionaries.
    """
    merged: Dict[str, Any] = {'config': config_dict}
    if wandb_identity is not None:
        merged['wandb_run'] = wandb_identity
    merged['metric_policy'] = METRIC_POLICY
    merged['validation_inference'] = dict(validation_options or {})
    merged['successful_updates'] = int(successful_updates)
    if validation_details:
        merged['validation_metrics'] = dict(validation_details)
    if config_dict and config_dict.get('model_spec'):
        merged['model_spec'] = config_dict['model_spec']
    if extra_state:
        merged.update(extra_state)
    return merged

