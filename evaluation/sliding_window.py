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

from functools import lru_cache
from scipy.ndimage import gaussian_filter

from evaluation.metrics import (
    CLASS_NAMES_MULTICLASS,
    CLASS_NAMES_BINARY,
    cal_multiclass_dice,
    cal_binary_dice,
    format_dice_summary,
    case_segmentation_stats, aggregate_segmentation_stats,
)
from utils.model_utils import extract_logits


@lru_cache(maxsize=4)
def compute_gaussian_weight_map(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 1.0 / 8,
    value_scaling_factor: float = 1.0
) -> np.ndarray:
    """
    Compute a 3D Gaussian importance weighting map following nnU-Net design.
    The center of the patch has maximum weight (1.0), smoothly decaying towards borders.
    This suppresses stitching artifacts and edge distortion during sliding-window inference.
    Cached via @lru_cache for zero overhead across volumes.
    """
    tmp = np.zeros(patch_size, dtype=np.float32)
    center = tuple(s // 2 for s in patch_size)
    tmp[center] = 1.0
    sigmas = [s * sigma_scale for s in patch_size]
    gaussian = gaussian_filter(tmp, sigmas, mode='constant', cval=0)
    gaussian = (gaussian / np.max(gaussian)) * value_scaling_factor
    # Set minimum threshold to avoid zero division / complete zero weight at extreme edges
    min_val = np.min(gaussian[gaussian > 0]) if np.any(gaussian > 0) else 1e-4
    gaussian[gaussian == 0] = min_val
    return gaussian.astype(np.float32)


def _forward_model(model: nn.Module, patch_tensor: torch.Tensor) -> torch.Tensor:
    """Safely extract primary segmentation logits from diverse model output signatures."""
    return extract_logits(model(patch_tensor))


def _forward_with_tta(
    model: nn.Module,
    patch_tensor: torch.Tensor,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
    flip_axes: Tuple[Tuple[int, ...], ...] = ((2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4))
) -> torch.Tensor:
    """
    Test-Time Augmentation (TTA) using 3D spatial flip combinations (following nnUNet design).
    Averages predicted probabilities over the original patch and mirrored variations.
    patch_tensor shape: (1, 1, Z, Y, X) -> spatial axes in 5D tensor are 2 (Z), 3 (Y), 4 (X).
    """
    def _predict(inp: torch.Tensor) -> torch.Tensor:
        if use_amp and device is not None and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = _forward_model(model, inp)
        else:
            logits = _forward_model(model, inp)
        return F.softmax(logits.float(), dim=1)

    # 1. Base prediction on original orientation
    prob = _predict(patch_tensor)

    # 2. Mirror predictions across spatial axes combinations
    num_variants = 1 + len(flip_axes)
    for axes in flip_axes:
        flipped_input = torch.flip(patch_tensor, dims=axes)
        flipped_prob = _predict(flipped_input)
        unflipped_prob = torch.flip(flipped_prob, dims=axes)
        prob += unflipped_prob

    return prob / float(num_variants)


def sliding_window_inference_3d(
    model: nn.Module,
    image: np.ndarray,
    num_classes: int,
    patch_size: Tuple[int, int, int] = (32, 160, 160),
    stride_z: int = 8,
    stride_xy: int = 32,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
    use_gaussian: bool = True,
    use_tta: bool = False
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
        use_gaussian: If True, weights overlapping patches with 3D Gaussian importance mask (nnUNet standard).
        use_tta: If True, applies 3D Test-Time Augmentation (flips across X, Y, Z axes).
        
    Returns:
        np.ndarray: Predicted discrete segmentation map of shape (Z, Y, X) with class integers.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
        raise ValueError('Expected a nonempty 3D image')
    if len(patch_size) != 3 or any(size <= 0 for size in patch_size):
        raise ValueError('Patch dimensions must be positive')
    if not 0 < stride_z <= patch_size[0] or not 0 < stride_xy <= min(patch_size[1:]):
        raise ValueError('Strides must be positive and no larger than patch dimensions to cover every voxel')
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

    # Precompute Gaussian importance weight map
    gaussian = compute_gaussian_weight_map((pz, py, px)) if use_gaussian else None

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

                    if use_tta:
                        prob_tensor = _forward_with_tta(model, patch_tensor, use_amp=use_amp, device=device)
                        prob = prob_tensor.squeeze(0).cpu().numpy()
                    else:
                        if use_amp and device.type == 'cuda':
                            with torch.amp.autocast('cuda'):
                                logits = _forward_model(model, patch_tensor)
                        else:
                            logits = _forward_model(model, patch_tensor)
                        prob = F.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()

                    if prob.shape != (num_classes, pz, py, px) or not np.isfinite(prob).all():
                        raise ValueError(f'Invalid model probabilities: shape={prob.shape}')
                    if gaussian is not None:
                        score_map[:, zs:zs + pz, ys:ys + py, xs:xs + px] += prob * gaussian
                        cnt[zs:zs + pz, ys:ys + py, xs:xs + px] += gaussian
                    else:
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
    world_size: Optional[int] = None,
    use_gaussian: bool = True,
    use_tta: bool = False,
    details_sink=None,
    visual_case_ids=None,
    visual_settings=None,
    compute_surface=False,
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
        use_gaussian: Whether to apply 3D Gaussian importance weighting mask to overlapping patches.
        use_tta: Whether to enable 3D Test-Time Augmentation (TTA).
        
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
    if total_cases == 0:
        raise ValueError('Validation dataset is empty')
    if world_size > 1 and not is_distributed:
        raise ValueError('Partitioned validation requires an initialized process group')
    if is_distributed:
        if rank != dist.get_rank() or world_size != dist.get_world_size():
            raise ValueError('Validation rank/world_size disagree with process group')
        # Training may use per-rank BatchNorm buffers. Evaluate one coherent model.
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    local_stats = []
    requested_visuals = return_visual_samples
    if is_distributed:
        count = [requested_visuals if rank == 0 else None]
        dist.broadcast_object_list(count, src=0)
        requested_visuals = count[0]
    if requested_visuals < 0:
        raise ValueError('num_vis_samples must be nonnegative')
    if visual_case_ids and len(set(visual_case_ids)) != len(visual_case_ids):
        raise ValueError('Visualization case IDs must be unique')
    names = list(getattr(val_dataset, 'image_list', []))
    if not names:
        names = [str(i) for i in range(total_cases)]
    if visual_case_ids:
        missing = set(visual_case_ids) - set(names)
        if missing:
            raise ValueError(f'Visualization case IDs absent from validation split: {sorted(missing)}')
        chosen = list(visual_case_ids)[:requested_visuals]
    else:
        chosen = sorted(names)[:requested_visuals]
    selected = set(chosen)
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
            f"(patch={patch_size}, stride_xy={stride_xy}, stride_z={stride_z}, amp={use_amp}, gaussian={use_gaussian}, tta={use_tta}){dist_info}..."
        )

    for step_i, case_idx in enumerate(my_indices):
        sample_item = val_dataset[case_idx]
        sample = sample_item[0] if isinstance(sample_item, tuple) else sample_item
        image, label = sample['image'], sample['label']
        case_name = sample.get('case', names[case_idx])

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
            device=device,
            use_gaussian=use_gaussian,
            use_tta=use_tta
        )

        stats = case_segmentation_stats(pred, label, num_classes,
            spacing=sample.get('metadata', {}).get('spacing'), compute_surface=compute_surface)
        local_stats.append((case_idx, stats))
        case_dices = [row['dice'] for row in stats['classes']]
        local_case_records.append((case_idx, case_dices))
        case_dice_metric = case_dices[0] if is_binary else case_dices

        # Every owner rank renders only bounded 2D records; never gather 3D volumes.
        if names[case_idx] in selected:
            from utils.visualizer import prepare_visual_record
            visual_records.append(prepare_visual_record({
                'case': str(case_name), 'image': image, 'label': label, 'pred': pred,
                'dice': case_dice_metric, 'metadata': sample.get('metadata', {}),
                'num_classes': num_classes,
            }, visual_settings))

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
        if [idx for idx, _ in all_flattened] != list(range(total_cases)):
            raise RuntimeError('Distributed validation did not cover each case exactly once')
        all_case_dices = [dice for _, dice in all_flattened]
    else:
        all_case_dices = [dice for _, dice in local_case_records]

    if is_distributed and world_size > 1:
        gathered_stats = [None] * world_size
        dist.all_gather_object(gathered_stats, local_stats)
        ordered_stats = [record for group in gathered_stats if group is not None for record in group]
        ordered_stats.sort(key=lambda item: item[0])
        stats = [record for _, record in ordered_stats]
        if requested_visuals:
            gathered_visuals = [None] * world_size if rank == 0 else None
            dist.gather_object(visual_records, gathered_visuals, dst=0)
            visual_records = [r for group in gathered_visuals for r in group] if rank == 0 else []
    else:
        stats = [record for _, record in local_stats]
    if rank == 0:
        order = {name: i for i, name in enumerate(chosen)}
        visual_records.sort(key=lambda r: order.get(r['case'], len(order)))
    if details_sink is not None:
        details_sink.update(aggregate_segmentation_stats(stats))
        details_sink['inference'] = {'patch_size': list(patch_size), 'stride_xy': stride_xy,
            'stride_z': stride_z, 'amp': use_amp, 'gaussian': use_gaussian, 'tta': use_tta, 'surface_metrics': compute_surface}
        details_sink['visual_cohort'] = chosen

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
