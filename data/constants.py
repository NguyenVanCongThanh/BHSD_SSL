"""
Shared Constants and Metadata for Datasets in MedSSL-BHSD.
Single Source of Truth for class names, labels, and dataset specs across training,
validation, and benchmark evaluation.
"""

# BHSD (Brain Hemorrhage Segmentation Dataset) - 6 classes
BHSD_MULTICLASS_NAMES = [
    "Background",
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural"
]

# BHSD Binary Task (Background vs Hemorrhage)
BHSD_BINARY_NAMES = [
    "Background",
    "Hemorrhage"
]

# ACDC (Automated Cardiac Diagnosis Challenge) - 4 classes (BG, RV, Myo, LV)
ACDC_CLASS_NAMES = [
    "Background",
    "RightVentricle",
    "Myocardium",
    "LeftVentricle"
]

# Left Atrium (LA Heart) - Binary segmentation (BG vs LA)
LA_HEART_CLASS_NAMES = [
    "Background",
    "LeftAtrium"
]

# Pancreas-CT - Binary segmentation (BG vs Pancreas)
PANCREAS_CLASS_NAMES = [
    "Background",
    "Pancreas"
]
