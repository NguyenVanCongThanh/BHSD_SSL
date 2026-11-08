"""
Shared Constants and Metadata for Datasets and Framework Defaults in MedSSL-BHSD.
Single Source of Truth for class names, labels, dataset specs, and baseline defaults
across training, augmentation, losses, validation, and tooling.
"""

from typing import Dict, List, Optional


# BHSD (Brain Hemorrhage Segmentation Dataset) - 6 classes
BHSD_MULTICLASS_NAMES: List[str] = [
    "Background",
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural",
]
BHSD_FOREGROUND_NAMES: List[str] = BHSD_MULTICLASS_NAMES[1:]
BHSD_CLASS_ABBREVS: List[str] = ["EDH", "IPH", "IVH", "SAH", "SDH"]

# BHSD Binary Task (Background vs Hemorrhage)
BHSD_BINARY_NAMES: List[str] = [
    "Background",
    "Hemorrhage",
]
BHSD_BINARY_FOREGROUND_NAMES: List[str] = BHSD_BINARY_NAMES[1:]

# ACDC (Automated Cardiac Diagnosis Challenge) - 4 classes (BG, RV, Myo, LV)
ACDC_CLASS_NAMES: List[str] = [
    "Background",
    "RightVentricle",
    "Myocardium",
    "LeftVentricle",
]

# Left Atrium (LA Heart) - Binary segmentation (BG vs LA)
LA_HEART_CLASS_NAMES: List[str] = [
    "Background",
    "LeftAtrium",
]

# Pancreas-CT - Binary segmentation (BG vs Pancreas)
PANCREAS_CLASS_NAMES: List[str] = [
    "Background",
    "Pancreas",
]

# Canonical dataset class registry keyed by (canonical_dataset_name -> task -> names)
DATASET_CLASS_NAMES: Dict[str, Dict[str, List[str]]] = {
    "bhsd": {
        "multiclass": BHSD_MULTICLASS_NAMES,
        "binary": BHSD_BINARY_NAMES,
    },
    "bhsd_multiclass": {"multiclass": BHSD_MULTICLASS_NAMES},
    "bhsd_binary": {"binary": BHSD_BINARY_NAMES},
    "acdc": {"multiclass": ACDC_CLASS_NAMES},
    "la": {"binary": LA_HEART_CLASS_NAMES},
    "pancreas": {"binary": PANCREAS_CLASS_NAMES},
}


def get_dataset_class_names(
    dataset_name: Optional[str] = "bhsd",
    task: Optional[str] = "multiclass",
    num_classes: Optional[int] = None,
) -> List[str]:
    """Resolve canonical class names (including Background at index 0) for a dataset/task."""
    ds_key = str(dataset_name or "bhsd").strip().lower()
    task_key = str(task or ("binary" if num_classes == 2 else "multiclass")).strip().lower()

    if ds_key in DATASET_CLASS_NAMES:
        variants = DATASET_CLASS_NAMES[ds_key]
        if task_key in variants:
            names = list(variants[task_key])
        else:
            names = list(next(iter(variants.values())))
        if num_classes is None or len(names) == num_classes:
            return names

    if num_classes == 2 or task_key == "binary":
        return list(BHSD_BINARY_NAMES)
    if num_classes is not None and num_classes > 0:
        if num_classes == len(BHSD_MULTICLASS_NAMES) and ds_key.startswith("bhsd"):
            return list(BHSD_MULTICLASS_NAMES)
        return ["Background"] + [f"Class_{idx}" for idx in range(1, num_classes)]
    return list(BHSD_MULTICLASS_NAMES)


def get_foreground_class_names(
    dataset_name: Optional[str] = "bhsd",
    task: Optional[str] = "multiclass",
    num_classes: Optional[int] = None,
) -> List[str]:
    """Return foreground class names (excluding Background index 0)."""
    return get_dataset_class_names(dataset_name, task=task, num_classes=num_classes)[1:]


# Framework paths, geometry, sliding-window, optimization, augmentation, and loss defaults
DEFAULT_DATA_ROOT = "~/notebook/preprocessed_bhsd_pick/"
DEFAULT_EXP_DIR = "~/experiments"
DEFAULT_PATCH_SIZE = (32, 160, 160)
DEFAULT_VAL_STRIDE_XY = 32
DEFAULT_VAL_STRIDE_Z = 8
DEFAULT_CLIP_NORM = 12.0

DEFAULT_FOREGROUND_OVERSAMPLE_PROB = 0.33
DEFAULT_CROP_JITTER_Z = 4
DEFAULT_CROP_JITTER_XY = 16
DEFAULT_AUGMENTATION_CONFIG: Dict[str, object] = {
    "scale_range": (0.85, 1.15),
    "scale_prob": 0.2,
    "gamma_range": (0.7, 1.5),
    "gamma_prob": 0.2,
    "blur_sigma_range": (0.5, 1.0),
    "blur_prob": 0.15,
}

DEFAULT_SUPERVISED_CE_WEIGHT = 0.5
DEFAULT_SUPERVISED_DICE_WEIGHT = 0.5
DEFAULT_CONSISTENCY_DICE_WEIGHT = 0.5
DEFAULT_CONSISTENCY_MSE_WEIGHT = 0.5
DEFAULT_FG_CONFIDENCE_MAX_THRESH = 0.4

