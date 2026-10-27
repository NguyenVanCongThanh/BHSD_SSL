"""
Common Model and Tensor Output Utilities for MedSSL-BHSD.
Re-exports unified output extraction logic from models.outputs for backward compatibility.
"""
from models.outputs import NON_AUX_KEYS, extract_logits, ModelOutput, wrap_model_output

__all__ = ["NON_AUX_KEYS", "extract_logits", "ModelOutput", "wrap_model_output"]
