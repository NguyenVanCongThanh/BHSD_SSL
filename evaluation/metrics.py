"""
Comprehensive Evaluation Metrics for BHSD (Brain Hemorrhage Segmentation Dataset).
Single Source of Truth for both Binary (Hemorrhage vs BG) and Multi-class (6 classes).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from data.constants import BHSD_MULTICLASS_NAMES as CLASS_NAMES_MULTICLASS, BHSD_BINARY_NAMES as CLASS_NAMES_BINARY


def _validate_class_maps(pred, gt, num_classes):
    if pred.shape != gt.shape or pred.size == 0:
        raise ValueError('Prediction and label must have equal nonempty shapes')
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError('Segmentation maps contain NaN/Inf')
    if not np.equal(pred, np.floor(pred)).all() or not np.equal(gt, np.floor(gt)).all():
        raise ValueError('Segmentation maps must contain integer class labels')
    if pred.min() < 0 or gt.min() < 0 or pred.max() >= num_classes or gt.max() >= num_classes:
        raise ValueError('Segmentation labels outside configured class range')


def cal_binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Dice score for binary segmentation.
    
    Args:
        pred: Binary prediction volume (0 or 1)
        gt: Ground truth binary volume (0 or 1)
    
    Returns:
        float: Dice coefficient in [0.0, 1.0]
    """
    if pred.shape != gt.shape:
        raise ValueError(f'Prediction/label shape mismatch: {pred.shape} vs {gt.shape}')
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
    _validate_class_maps(pred, gt, num_classes)
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
    compute_surface: bool = False,
    spacing=None
) -> Dict[str, Union[float, List[float]]]:
    """
    Calculate comprehensive metrics for a single 3D case.
    
    Args:
        pred: Predicted volume
        gt: Ground truth volume
        is_binary: True if binary task, False if multi-class
        num_classes: Number of classes (2 for binary, 6 for multiclass)
        compute_surface: Compute symmetric HD95/ASD in mm with required spacing
        
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
            if spacing is None:
                raise ValueError('Surface distances require physical spacing')
            result.update(surface_distances(p_bin.astype(bool), g_bin.astype(bool), spacing))
            intersection = int(np.count_nonzero(p_bin & g_bin))
            union = int(np.count_nonzero(p_bin | g_bin))
            result['iou'] = intersection / union if union else 1.

        return result
    else:
        per_class = cal_multiclass_dice(pred, gt, num_classes=num_classes)
        mdice = float(np.mean(per_class)) if len(per_class) > 0 else 0.0
        foreground_names = CLASS_NAMES_MULTICLASS[1:num_classes]
        
        result = {"mDice": mdice, "per_class": per_class, "class_names": foreground_names}
        if compute_surface:
            stats = case_segmentation_stats(pred, gt, num_classes, spacing, True)
            result['per_class_surface'] = [{key: row[key] for key in ('hd95_mm', 'asd_mm', 'surface_status')}
                                           for row in stats['classes']]
        return result


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


METRIC_POLICY = {
    'selection_metric': 'foreground_case_mean_dice',
    'empty_prediction_and_target_dice': 1.0,
    'positive_dice': 'mean over cases with ground-truth foreground for each class',
    'undefined': None,
}


def case_segmentation_stats(pred, gt, num_classes=6, spacing=None, compute_surface=False):
    """Native-grid counts; physical volumes exist only with valid spacing."""
    _validate_class_maps(pred, gt, num_classes)
    voxel_ml = None
    if spacing is not None:
        values = np.asarray(spacing, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Spacing must contain positive finite Z,Y,X millimetres')
        voxel_ml = float(np.prod(values) / 1000.)
    if compute_surface and voxel_ml is None:
        raise ValueError('Surface distances require physical voxel spacing')
    classes = []
    for cls in range(1, num_classes):
        p, g = pred == cls, gt == cls
        tp = int(np.count_nonzero(p & g))
        fp, fn = int(np.count_nonzero(p & ~g)), int(np.count_nonzero(~p & g))
        surface = surface_distances(p, g, spacing) if compute_surface else {}
        classes.append({**surface, 'tp': tp, 'fp': fp, 'fn': fn,
                        'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                        'fp_ml': fp * voxel_ml if voxel_ml is not None else None})
    return {'classes': classes}


def aggregate_segmentation_stats(records):
    """Preserve legacy Dice while exposing positive cases and negative-case FP."""
    if not records:
        raise ValueError('Cannot aggregate an empty validation set')
    result = {'num_cases': len(records), 'policy': METRIC_POLICY, 'per_class': []}
    for index in range(len(records[0]['classes'])):
        rows = [record['classes'][index] for record in records]
        positives = [row for row in rows if row['tp'] + row['fn'] > 0]
        negatives = [row for row in rows if row['tp'] + row['fn'] == 0]
        tp, fp, fn = (sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn'))
        physical = [row['fp_ml'] for row in negatives]
        surface_rows = [r for r in rows if r.get('hd95_mm') is not None]
        result['per_class'].append({
            'surface_valid_pairs': len(surface_rows),
            'hd95_mm_mean': float(np.mean([r['hd95_mm'] for r in surface_rows])) if surface_rows else None,
            'asd_mm_mean': float(np.mean([r['asd_mm'] for r in surface_rows])) if surface_rows else None,
            'class_id': index + 1, 'dice_case_mean': float(np.mean([r['dice'] for r in rows])),
            'dice_positive_mean': float(np.mean([r['dice'] for r in positives])) if positives else None,
            'positive_cases': len(positives), 'negative_cases': len(negatives),
            'negative_fp_case_rate': sum(r['fp'] > 0 for r in negatives) / len(negatives) if negatives else None,
            'negative_fp_ml_mean': float(np.mean(physical)) if physical and all(v is not None for v in physical) else None,
            'negative_fp_voxels_mean': float(np.mean([r['fp'] for r in negatives])) if negatives else None,
            'precision_micro': tp / (tp + fp) if tp + fp else None,
            'recall_micro': tp / (tp + fn) if tp + fn else None,
            'tp_voxels': tp, 'fp_voxels': fp, 'fn_voxels': fn,
        })
    return result


def surface_distances(pred, gt, spacing):
    """Symmetric surface distances in mm; no fabricated finite penalty for empty masks."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    if not pred.any() or not gt.any():
        return {'hd95_mm': None, 'asd_mm': None,
                'surface_status': 'both_empty' if not pred.any() and not gt.any() else 'one_empty'}
    p_boundary = pred & ~binary_erosion(pred)
    g_boundary = gt & ~binary_erosion(gt)
    p_to_g = distance_transform_edt(~g_boundary, sampling=spacing)[p_boundary]
    g_to_p = distance_transform_edt(~p_boundary, sampling=spacing)[g_boundary]
    distances = np.concatenate([p_to_g, g_to_p])
    return {'hd95_mm': float(np.percentile(distances, 95)),
            'asd_mm': float(np.mean(distances)), 'surface_status': 'valid'}
