"""
Evaluation package for MedSSL-BHSD.
Contains unified 3D sliding-window inference, metrics computation, and 5-fold benchmark aggregation.
"""

from evaluation.metrics import (
    CLASS_NAMES_MULTICLASS,
    CLASS_NAMES_BINARY,
    cal_multiclass_dice,
    cal_binary_dice,
    calculate_case_metrics,
    format_dice_summary,
)
from evaluation.sliding_window import (
    sliding_window_inference_3d,
    evaluate_dataset_sliding_window,
)
from evaluation.complexity import (
    profile_model_complexity,
    format_complexity_summary,
)
from evaluation.native_space import (
    restore_volume_to_native_space,
    load_raw_nifti_ground_truth,
    restore_case_to_native,
)

__all__ = [
    "CLASS_NAMES_MULTICLASS",
    "CLASS_NAMES_BINARY",
    "cal_multiclass_dice",
    "cal_binary_dice",
    "calculate_case_metrics",
    "format_dice_summary",
    "sliding_window_inference_3d",
    "evaluate_dataset_sliding_window",
    "profile_model_complexity",
    "format_complexity_summary",
    "restore_volume_to_native_space",
    "load_raw_nifti_ground_truth",
    "restore_case_to_native",
]
