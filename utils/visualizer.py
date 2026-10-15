"""
Visualizer Module for Medical 3D Segmentation (BHSD).
Provides utilities to extract informative 2D slices from 3D volumes
and construct interactive W&B Segmentation Masks and Comparison Panels.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def extract_best_slice(
    image_3d: np.ndarray,
    mask_3d: np.ndarray,
    pred_3d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Select the most clinically informative axial slice (Z plane) containing
    the maximum amount of lesion/pathology (foreground pixels).
    Falls back to middle slice if volume has no positive annotations.

    Args:
        image_3d: 3D CT/MRI array of shape (Z, Y, X).
        mask_3d: Ground truth discrete mask of shape (Z, Y, X).
        pred_3d: Predicted discrete mask of shape (Z, Y, X).

    Returns:
        slice_img: 2D normalized image of shape (Y, X) in range [0, 1].
        slice_mask: 2D ground truth mask of shape (Y, X).
        slice_pred: 2D predicted mask of shape (Y, X).
        best_z: Selected Z index.
    """
    z_len = mask_3d.shape[0]

    # Count foreground voxels per slice
    gt_fg_counts = (mask_3d > 0).sum(axis=(1, 2))
    pred_fg_counts = (pred_3d > 0).sum(axis=(1, 2))
    total_fg_counts = gt_fg_counts + pred_fg_counts

    if total_fg_counts.max() > 0:
        best_z = int(np.argmax(total_fg_counts))
    else:
        best_z = z_len // 2

    # Extract slices
    slice_img = image_3d[best_z].astype(np.float32)
    slice_mask = mask_3d[best_z].astype(np.uint8)
    slice_pred = pred_3d[best_z].astype(np.uint8)

    # Normalize image to [0, 1] for crisp display
    min_val, max_val = slice_img.min(), slice_img.max()
    if max_val - min_val > 1e-6:
        slice_img = (slice_img - min_val) / (max_val - min_val)
    else:
        slice_img = np.zeros_like(slice_img)

    return slice_img, slice_mask, slice_pred, best_z


def create_wandb_segmentation_samples(
    visual_records: List[Dict[str, Any]],
    class_labels: Dict[int, str]
) -> List[Any]:
    """
    Construct a list of wandb.Image objects with interactive segmentation mask overlays.

    Args:
        visual_records: List of dicts with keys:
            - 'case': str case identifier
            - 'image': np.ndarray 3D (Z, Y, X)
            - 'label': np.ndarray 3D (Z, Y, X)
            - 'pred': np.ndarray 3D (Z, Y, X)
            - Optional 'dice': float or list of floats
        class_labels: Mapping from class integer (0, 1, ...) to class name string.

    Returns:
        List of wandb.Image objects with masks for W&B Dashboard.
    """
    try:
        import wandb
    except ImportError:
        return []

    wandb_images = []
    for record in visual_records:
        case_id = record.get('case', 'sample')
        image_3d = record['image']
        mask_3d = record['label']
        pred_3d = record['pred']

        slice_img, slice_gt, slice_pr, best_z = extract_best_slice(image_3d, mask_3d, pred_3d)

        caption_parts = [f"Case: {case_id}", f"Slice Z: {best_z}"]
        if 'dice' in record:
            d = record['dice']
            dice_str = f"Dice: {d:.4f}" if isinstance(d, (int, float)) else f"mDice: {np.mean(d):.4f}"
            caption_parts.append(dice_str)
        caption = " | ".join(caption_parts)

        # Build wandb.Image with dual masks (ground_truth vs predictions)
        wandb_img = wandb.Image(
            slice_img,
            caption=caption,
            masks={
                "predictions": {
                    "mask_data": slice_pr,
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": slice_gt,
                    "class_labels": class_labels
                }
            }
        )
        wandb_images.append(wandb_img)

    return wandb_images
