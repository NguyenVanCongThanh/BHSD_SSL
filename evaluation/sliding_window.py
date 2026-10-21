"""
Unified 3D Sliding-Window Inference Engine for Volumetric Medical Segmentation.
Optimized for BHSD (Brain Hemorrhage Segmentation Dataset) with AMP FP16 and low VRAM footprint.
Single Source of Truth across training, validation, and benchmarking.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from evaluation.metrics import (
    CLASS_NAMES_MULTICLASS,
    CLASS_NAMES_BINARY,
    cal_multiclass_dice,
    cal_binary_dice,
    format_dice_summary,
)
from methods.ssl_utils import extract_logits


def _forward_model(model: nn.Module, patch_tensor: torch.Tensor) -> torch.Tensor:
    """Safely extract primary segmentation logits from diverse model output signatures."""
    return extract_logits(model(patch_tensor))


def sliding_window_inference_3d(
    model: nn.Module,
    image: np.ndarray,
    num_classes: int,
    patch_size: Tuple[int, int, int] = (32, 160, 160),
    stride_z: int = 8,
    stride_xy: int = 32,
    use_amp: bool = True,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Run 3D sliding-window inference on a single 3D volume.
    
    Args:
        model: PyTorch model in eval mode.
        image: 3D numpy array of shape (Z, Y, X).
        num_classes: Number of segmentation classes (2 for binary, 6 for multiclass).
        patch_size: Patch dimensions (pz, py, px).
        stride_z: Stride along slice (Z) dimension.
        stride_xy: Stride along axial (Y, X) dimensions.
        use_amp: If True, uses torch.cuda.amp.autocast() for FP16 speedup.
        device: Torch device (defaults to cuda if available).
        
    Returns:
        np.ndarray: Predicted discrete segmentation map of shape (Z, Y, X) with class integers.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    z, y, x = image.shape
    pz, py, px = patch_size

    # 1. Symmetric padding if volume is smaller than patch size
    pad_z = max(pz - z, 0)
    pad_y = max(py - y, 0)
    pad_x = max(px - x, 0)
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        image = np.pad(
            image,
            [(0, pad_z), (0, pad_y), (0, pad_x)],
            mode='constant',
            constant_values=0
        )

    w_z, w_y, w_x = image.shape

    # 2. Compute step counts
    sz = max(int(np.ceil((w_z - pz) / stride_z)) + 1, 1)
    sy = max(int(np.ceil((w_y - py) / stride_xy)) + 1, 1)
    sx = max(int(np.ceil((w_x - px) / stride_xy)) + 1, 1)

    score_map = np.zeros((num_classes, w_z, w_y, w_x), dtype=np.float32)
    cnt = np.zeros((w_z, w_y, w_x), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for iz in range(sz):
            zs = min(iz * stride_z, w_z - pz)
            for iy in range(sy):
                ys = min(iy * stride_xy, w_y - py)
                for ix in range(sx):
                    xs = min(ix * stride_xy, w_x - px)

                    patch = image[zs:zs + pz, ys:ys + py, xs:xs + px]
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

                    if use_amp and device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            logits = _forward_model(model, patch_tensor)
                    else:
                        logits = _forward_model(model, patch_tensor)

                    prob = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    score_map[:, zs:zs + pz, ys:ys + py, xs:xs + px] += prob
                    cnt[zs:zs + pz, ys:ys + py, xs:xs + px] += 1.0

    score_map = score_map / np.expand_dims(cnt, axis=0)
    pred = np.argmax(score_map, axis=0)[:z, :y, :x]
    return pred


def evaluate_dataset_sliding_window(
    model: nn.Module,
    val_dataset,
    num_classes: int = 6,
    is_binary: bool = False,
    patch_size: Tuple[int, int, int] = (32, 160, 160),
    stride_z: int = 8,
    stride_xy: int = 32,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
    log_progress: bool = True,
    return_visual_samples: int = 0,
    rank: Optional[int] = None,
    world_size: Optional[int] = None
) -> Union[Tuple[float, List[float]], Tuple[float, List[float], List[Dict[str, Any]]]]:
    """
    Run 3D sliding window inference across a full validation dataset.
    Supports Distributed Multi-GPU validation: partitions cases across ranks and gathers results seamlessly.
    
    Args:
        model: Trained PyTorch model.
        val_dataset: BHSDDataset or similar iterable dataset.
        num_classes: Number of classes (2 for binary, 6 for multiclass).
        is_binary: True if binary task, False if multi-class.
        patch_size: 3D patch dimensions.
        stride_z: Z stride.
        stride_xy: XY stride.
        use_amp: Whether to use FP16 AMP.
        device: Target torch device.
        log_progress: Whether to print progress periodically.
        return_visual_samples: Number of visual sample records to return (0 = disable).
        rank: Distributed process rank (auto-detected if None).
        world_size: Total distributed processes (auto-detected if None).
        
    Returns:
        If return_visual_samples == 0:
            Tuple of (mean_dice, per_class_dices).
        If return_visual_samples > 0:
            Tuple of (mean_dice, per_class_dices, visual_records).
    """
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        if rank is None:
            rank = dist.get_rank()
        if world_size is None:
            world_size = dist.get_world_size()
    else:
        rank = 0 if rank is None else rank
        world_size = 1 if world_size is None else world_size

    model.eval()
    local_case_records: List[Tuple[int, List[float]]] = []
    visual_records: List[Dict[str, Any]] = []
    total_cases = len(val_dataset)
    t0 = time.time()

    # Partition validation cases evenly among all ranks
    if world_size > 1:
        my_indices = [idx for idx in range(total_cases) if idx % world_size == rank]
    else:
        my_indices = list(range(total_cases))

    num_my_cases = len(my_indices)

    if log_progress:
        dist_info = f" | Multi-GPU: Rank {rank}/{world_size} evaluating {num_my_cases}/{total_cases} cases" if world_size > 1 else ""
        logging.info(
            f"  [Validation] Evaluating {total_cases} cases "
            f"(patch={patch_size}, stride_xy={stride_xy}, stride_z={stride_z}, amp={use_amp}){dist_info}..."
        )

    for step_i, case_idx in enumerate(my_indices):
        sample_item = val_dataset[case_idx]
        sample = sample_item[0] if isinstance(sample_item, tuple) else sample_item
        image, label = sample['image'], sample['label']
        case_name = sample.get('case', f"case_{case_idx:03d}")

        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.squeeze().cpu().numpy()

        pred = sliding_window_inference_3d(
            model=model,
            image=image,
            num_classes=num_classes,
            patch_size=patch_size,
            stride_z=stride_z,
            stride_xy=stride_xy,
            use_amp=use_amp,
            device=device
        )

        if is_binary:
            dice = cal_binary_dice(pred, label)
            local_case_records.append((case_idx, [dice]))
            case_dice_metric = dice
        else:
            case_dices = cal_multiclass_dice(pred, label, num_classes=num_classes)
            local_case_records.append((case_idx, case_dices))
            case_dice_metric = case_dices

        # Collect sample for visualization (rank 0 only)
        if rank == 0 and return_visual_samples > 0 and len(visual_records) < return_visual_samples:
            visual_records.append({
                'case': str(case_name),
                'image': image,
                'label': label,
                'pred': pred,
                'dice': case_dice_metric
            })

        if log_progress and ((step_i + 1) % 5 == 0 or (step_i + 1) == num_my_cases):
            elapsed = time.time() - t0
            pct = ((step_i + 1) / max(num_my_cases, 1)) * 100
            prefix = f"[Rank {rank}] " if world_size > 1 else ""
            msg = f"  {prefix}[Validation Progress] Done {step_i + 1}/{num_my_cases} cases ({pct:.0f}%) | {elapsed:.1f}s"
            logging.info(msg)
            print(msg, flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Synchronize and gather per-case Dice metrics across all DDP ranks
    if is_distributed and world_size > 1:
        gathered_lists: List[List[Tuple[int, List[float]]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_lists, local_case_records)
        all_flattened: List[Tuple[int, List[float]]] = []
        for r_list in gathered_lists:
            if r_list is not None:
                all_flattened.extend(r_list)
        # Sort by original case index to maintain deterministic ordering
        all_flattened.sort(key=lambda x: x[0])
        all_case_dices = [dice for _, dice in all_flattened]
    else:
        all_case_dices = [dice for _, dice in local_case_records]

    empty_classes = [0.0] * (1 if is_binary else num_classes - 1)
    if len(all_case_dices) == 0:
        if return_visual_samples > 0:
            return 0.0, empty_classes, []
        return 0.0, empty_classes

    mean_per_class = np.mean(np.array(all_case_dices), axis=0).tolist()
    mean_dice = float(np.mean(mean_per_class))

    if return_visual_samples > 0:
        return mean_dice, mean_per_class, visual_records

    return mean_dice, mean_per_class
