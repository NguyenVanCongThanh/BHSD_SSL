"""
Comprehensive Evaluation Metrics for BHSD (Brain Hemorrhage Segmentation Dataset).
Single Source of Truth for both Binary (Hemorrhage vs BG) and Multi-class (6 classes).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from data.constants import BHSD_MULTICLASS_NAMES as CLASS_NAMES_MULTICLASS, BHSD_BINARY_NAMES as CLASS_NAMES_BINARY


def cal_binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Dice score for binary segmentation.
    
    Args:
        pred: Binary prediction volume (0 or 1)
        gt: Ground truth binary volume (0 or 1)
    
    Returns:
        float: Dice coefficient in [0.0, 1.0]
    """
    p = (pred > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    intersection = np.sum(p * g)
    total = np.sum(p) + np.sum(g)
    if total == 0:
        return 1.0  # Agreement on true negative case
    return float((2.0 * intersection) / total)


def cal_multiclass_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int = 6
) -> List[float]:
    """
    Calculate per-class Dice scores for classes 1..num_classes-1 (foreground only).
    
    Args:
        pred: Predicted class map with integers in range [0, num_classes-1]
        gt: Ground truth class map with integers in range [0, num_classes-1]
        num_classes: Number of classes including background (default: 6 for BHSD)
        
    Returns:
        List of Dice scores for each foreground class.
    """
    class_dices = []
    for c in range(1, num_classes):
        p_c = (pred == c).astype(np.float32)
        g_c = (gt == c).astype(np.float32)
        intersection = np.sum(p_c * g_c)
        total = np.sum(p_c) + np.sum(g_c)
        if total == 0:
            dice = 1.0  # Agreement on absent lesion
        else:
            dice = (2.0 * intersection) / total
        class_dices.append(float(dice))
    return class_dices


def calculate_case_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    is_binary: bool = False,
    num_classes: int = 6,
    compute_surface: bool = False
) -> Dict[str, Union[float, List[float]]]:
    """
    Calculate comprehensive metrics for a single 3D case.
    
    Args:
        pred: Predicted volume
        gt: Ground truth volume
        is_binary: True if binary task, False if multi-class
        num_classes: Number of classes (2 for binary, 6 for multiclass)
        compute_surface: Whether to compute HD95 / ASD using medpy if available
        
    Returns:
        Dict containing mDice, per_class_dice, and optionally HD95/IoU.
    """
    if is_binary:
        p_bin = (pred > 0).astype(np.uint8)
        g_bin = (gt > 0).astype(np.uint8)
        dice = cal_binary_dice(p_bin, g_bin)
        
        result: Dict[str, Union[float, List[float]]] = {
            "mDice": dice,
            "per_class": [dice],
            "class_names": ["Hemorrhage"]
        }
        
        if compute_surface:
            try:
                from medpy import metric
                if np.sum(p_bin) > 0 and np.sum(g_bin) > 0:
                    result["hd95"] = float(metric.binary.hd95(p_bin, g_bin))
                    result["asd"] = float(metric.binary.asd(p_bin, g_bin))
                    result["iou"] = float(metric.binary.jc(p_bin, g_bin))
                else:
                    result["hd95"] = 0.0 if np.sum(p_bin) == np.sum(g_bin) else 100.0
                    result["asd"] = 0.0 if np.sum(p_bin) == np.sum(g_bin) else 100.0
                    result["iou"] = 1.0 if np.sum(p_bin) == np.sum(g_bin) else 0.0
            except Exception:
                pass
                
        return result
    else:
        per_class = cal_multiclass_dice(pred, gt, num_classes=num_classes)
        mdice = float(np.mean(per_class)) if len(per_class) > 0 else 0.0
        foreground_names = CLASS_NAMES_MULTICLASS[1:num_classes]
        
        return {
            "mDice": mdice,
            "per_class": per_class,
            "class_names": foreground_names
        }


def format_dice_summary(
    per_class: List[float],
    class_names: Optional[List[str]] = None
) -> str:
    """Format per-class dice scores into a human-readable string."""
    if class_names is None:
        if len(per_class) == 5:
            class_names = CLASS_NAMES_MULTICLASS[1:]
        elif len(per_class) == 1:
            class_names = ["Hemorrhage"]
        else:
            class_names = [f"Class_{i+1}" for i in range(len(per_class))]
            
    parts = [f"{name}: {score:.4f}" for name, score in zip(class_names, per_class)]
    return ", ".join(parts)
