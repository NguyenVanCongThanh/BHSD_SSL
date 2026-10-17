"""
Visualizer Module for Medical 3D Segmentation (BHSD).
Provides utilities to extract informative 2D slices from 3D volumes
and construct interactive W&B Segmentation Masks and Comparison Panels.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F


def standardize_2d_slice(
    arr: np.ndarray,
    target_shape: Tuple[int, int] = (512, 512),
    is_mask: bool = False
) -> np.ndarray:
    """
    Ensure every 2D slice has identical spatial resolution (512, 512)
    to completely eliminate W&B UI 'Images sizes do not match' warnings
    and ensure smooth step-by-step slider transitions.
    """
    if arr.shape == target_shape:
        return arr

    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    mode = 'nearest' if is_mask else 'bilinear'
    align_corners = False if mode == 'bilinear' else None
    t_resized = F.interpolate(t, size=target_shape, mode=mode, align_corners=align_corners)
    out = t_resized.squeeze(0).squeeze(0).numpy()
    return out.astype(np.uint8) if is_mask else out.astype(np.float32)


def extract_best_slice(
    image_3d: np.ndarray,
    mask_3d: np.ndarray,
    pred_3d: np.ndarray,
    target_shape: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Select the most clinically informative axial slice (Z plane) containing
    the maximum amount of lesion/pathology (foreground pixels).
    
    Crucial fix for W&B:
    1. Prioritizes Ground Truth lesion center so the slice index Z remains
       100% CONSTANT across all validation iterations (200, 400, 600... 12000),
       allowing doctors/researchers to watch the exact same lesion slice improve.
    2. Enforces consistent (512, 512) resolution across all cases so W&B
       can build a uniform media gallery without display corruption.
    """
    z_len = mask_3d.shape[0]

    # Count foreground voxels per slice
    gt_fg_counts = (mask_3d > 0).sum(axis=(1, 2))
    pred_fg_counts = (pred_3d > 0).sum(axis=(1, 2))

    # Fix: ground truth is deterministic and static across all training steps!
    if gt_fg_counts.max() > 0:
        best_z = int(np.argmax(gt_fg_counts))
    elif pred_fg_counts.max() > 0:
        best_z = int(np.argmax(pred_fg_counts))
    else:
        best_z = z_len // 2

    # Extract slices at selected Z
    slice_img = image_3d[best_z].astype(np.float32)
    slice_mask = mask_3d[best_z].astype(np.uint8)
    slice_pred = pred_3d[best_z].astype(np.uint8)

    # Normalize image to [0, 1] for crisp display
    min_val, max_val = slice_img.min(), slice_img.max()
    if max_val - min_val > 1e-6:
        slice_img = (slice_img - min_val) / (max_val - min_val)
    else:
        slice_img = np.zeros_like(slice_img)

    # Standardize dimensions to target_shape (512, 512)
    slice_img = standardize_2d_slice(slice_img, target_shape=target_shape, is_mask=False)
    slice_mask = standardize_2d_slice(slice_mask, target_shape=target_shape, is_mask=True)
    slice_pred = standardize_2d_slice(slice_pred, target_shape=target_shape, is_mask=True)

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
