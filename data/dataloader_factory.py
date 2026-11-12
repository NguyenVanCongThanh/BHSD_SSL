"""
Data Factory for 3D Semi-Supervised Medical Image Segmentation (BHSD).
Constructs 3D augmentation pipelines, datasets, split partition indices,
and supervised / semi-supervised data loaders with single-GPU and DDP samplers.
"""

from dataclasses import dataclass
from typing import Tuple, Sequence, Optional, List, Dict, Any
import logging
import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, Sampler
from torchvision import transforms

from core.config_parser import Config
from core.batching import BatchLayout
from core.registry import DATASETS
from data.constants import (
    BHSD_FOREGROUND_NAMES,
    DEFAULT_AUGMENTATION_CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_FOREGROUND_OVERSAMPLE_PROB,
)
from data.samplers import TwoStreamBatchSampler, DistributedTwoStreamBatchSampler
from data.transforms3d import (
    RandomRotFlip,
    RandomAxialRotate3D,
    RandomCrop,
    ToTensor,
    RandomScaling3D,
    GaussianBlur3D,
    GaussianNoise,
    GammaTransform,
    MultiplicativeBrightness3D,
    ContrastTransform3D,
    SimulateLowResolution3D,
)

DEFAULT_STRATIFIED_FG_CLASSES = len(BHSD_FOREGROUND_NAMES)
DEFAULT_STRATIFIED_STRONG_VOXEL_THRESH = 1000
STRATIFIED_MIN_CASE_WEIGHT = 1000.0
STRATIFIED_COVERAGE_WEIGHT = 100.0
STRATIFIED_VOXEL_STD_PENALTY = 0.5


def _resolve_data_root(cfg: Config) -> str:
    """Resolve dataset root directory from cfg.data.root_path, MEDSSL_DATA_ROOT, or default relative dir."""
    data_cfg = getattr(cfg, 'data', None)
    root_path = getattr(data_cfg, 'root_path', None) if data_cfg is not None else None
    if not root_path:
        root_path = os.environ.get('MEDSSL_DATA_ROOT') or DEFAULT_DATA_ROOT
    return os.path.expanduser(os.path.expandvars(str(root_path)))


def _extract_aug_config(cfg: Config) -> Dict[str, Any]:
    """Read optional data augmentation overrides from cfg.data with canonical defaults."""
    data_cfg = getattr(cfg, 'data', None)
    if data_cfg is None:
        return dict(DEFAULT_AUGMENTATION_CONFIG)
    cfg_map = {
        'scale_range': getattr(data_cfg, 'aug_scale_range', None),
        'scale_prob': getattr(data_cfg, 'aug_scale_prob', None),
        'gamma_range': getattr(data_cfg, 'aug_gamma_range', None),
        'gamma_prob': getattr(data_cfg, 'aug_gamma_prob', None),
        'blur_sigma_range': getattr(data_cfg, 'aug_blur_sigma_range', None),
        'blur_prob': getattr(data_cfg, 'aug_blur_prob', None),
        'rotation_prob': getattr(data_cfg, 'aug_rotation_prob', None),
        'noise_prob': getattr(data_cfg, 'aug_noise_prob', None),
        'brightness_prob': getattr(data_cfg, 'aug_brightness_prob', None),
        'contrast_prob': getattr(data_cfg, 'aug_contrast_prob', None),
        'lowres_prob': getattr(data_cfg, 'aug_lowres_prob', None),
        'mirror_z': getattr(data_cfg, 'aug_mirror_z', None),
    }
    merged = dict(DEFAULT_AUGMENTATION_CONFIG)
    for k, v in cfg_map.items():
        if v is not None:
            if k == 'mirror_z':
                merged[k] = bool(v)
            elif k.endswith('_range'):
                merged[k] = tuple(v)
            else:
                merged[k] = float(v)
    return merged


@dataclass
class DataLoaderBundle:
    """Named bundle returned by build_bhsd_dataloaders."""
    db_train: Dataset
    db_val: Dataset
    trainloader: DataLoader
    train_sampler: Optional[Sampler] = None
    batch_sampler: Optional[Sampler] = None


def build_bhsd_transforms(
    patch_size: Tuple[int, int, int],
    extended_aug: bool = True,
    foreground_prob: Optional[float] = None,
    aug_config: Optional[Dict[str, Any]] = None,
    aug_preset: str = "legacy",
) -> transforms.Compose:
    """Build 3D data augmentation pipeline for training ('legacy' or 'nnunet')."""
    preset = str(aug_preset or "legacy").strip().lower()
    if preset not in ("legacy", "nnunet"):
        raise ValueError(f"Unsupported aug_preset {aug_preset!r}; expected 'legacy' or 'nnunet'")
    resolved_aug = dict(DEFAULT_AUGMENTATION_CONFIG)
    if aug_config:
        resolved_aug.update(aug_config)
    fg_prob = DEFAULT_FOREGROUND_OVERSAMPLE_PROB if foreground_prob is None else float(foreground_prob)

    if preset == "nnunet":
        mirror_z = bool(resolved_aug.get('mirror_z', True))
        aug_list = [RandomRotFlip(mirror_z=mirror_z)]
        if extended_aug:
            deg = float(resolved_aug.get('rotation_deg', 180.0))
            aug_list.extend([
                RandomAxialRotate3D(
                    deg_range=(-deg, deg),
                    prob=float(resolved_aug['rotation_prob']),
                    pad_to_min=True,
                ),
                RandomScaling3D(
                    scale_range=tuple(resolved_aug['scale_range']),
                    prob=float(resolved_aug['scale_prob']),
                    pad_to_min=True,
                ),
            ])
        aug_list.append(RandomCrop(patch_size, foreground_prob=fg_prob, pad_to_min=True))
        if extended_aug:
            aug_list.extend([
                GaussianNoise(
                    std_range=tuple(resolved_aug['noise_std_range']),
                    prob=float(resolved_aug['noise_prob']),
                ),
                GaussianBlur3D(
                    sigma_range=tuple(resolved_aug['blur_sigma_range']),
                    prob=float(resolved_aug['blur_prob']),
                ),
                MultiplicativeBrightness3D(
                    multiplier_range=tuple(resolved_aug['brightness_range']),
                    prob=float(resolved_aug['brightness_prob']),
                ),
                ContrastTransform3D(
                    contrast_range=tuple(resolved_aug['contrast_range']),
                    prob=float(resolved_aug['contrast_prob']),
                ),
                SimulateLowResolution3D(
                    zoom_range=tuple(resolved_aug['lowres_scale_range']),
                    prob=float(resolved_aug['lowres_prob']),
                ),
                GammaTransform(
                    gamma_range=tuple(resolved_aug['gamma_range']),
                    prob=float(resolved_aug['invert_gamma_prob']),
                    retain_stats=True,
                    invert_image=True,
                ),
                GammaTransform(
                    gamma_range=tuple(resolved_aug['gamma_range']),
                    prob=float(resolved_aug['gamma_prob']),
                    retain_stats=True,
                    invert_image=False,
                ),
            ])
        aug_list.append(ToTensor())
        return transforms.Compose(aug_list)

    aug_list = [
        RandomRotFlip(),
    ]
    if extended_aug:
        aug_list.extend([
            RandomScaling3D(
                scale_range=tuple(resolved_aug['scale_range']),
                prob=float(resolved_aug['scale_prob']),
            ),
            GammaTransform(
                gamma_range=tuple(resolved_aug['gamma_range']),
                prob=float(resolved_aug['gamma_prob']),
            ),
            GaussianBlur3D(
                sigma_range=tuple(resolved_aug['blur_sigma_range']),
                prob=float(resolved_aug['blur_prob']),
            ),
        ])
    aug_list.extend([
        RandomCrop(patch_size, foreground_prob=fg_prob),
        ToTensor()
    ])
    return transforms.Compose(aug_list)


def select_stratified_labeled_indices(
    db_train: Dataset,
    labelnum: int,
    max_samples: int,
    num_fg_classes: int = DEFAULT_STRATIFIED_FG_CLASSES,
    strong_voxel_thresh: int = DEFAULT_STRATIFIED_STRONG_VOXEL_THRESH,
) -> List[int]:
    """
    Deterministically select `labelnum` cases out of `0..max_samples-1` using
    multi-label greedy stratification so that all foreground hemorrhage classes (1..5)
    are represented evenly in both case count and voxel volume.

    Important: Always scans raw HDF5 labels on Classes 1..5 regardless of whether
    the training task is 'multiclass' or 'binary', ensuring identical 5% labeled
    cohorts across both tasks for fair benchmarking.
    """
    if labelnum >= max_samples:
        return list(range(max_samples))

    base_dir = getattr(db_train, 'base_dir', '')
    image_list = getattr(db_train, 'image_list', [])
    pool_size = min(max_samples, len(image_list))
    if pool_size <= 0 or not base_dir:
        return list(range(labelnum))

    vox_mat = np.zeros((pool_size, num_fg_classes), dtype=np.float64)
    valid_scan = False
    for idx in range(pool_size):
        case_name = image_list[idx]
        h5_path = os.path.join(base_dir, 'data', f'{case_name}.h5')
        if not os.path.isfile(h5_path):
            continue
        with h5py.File(h5_path, 'r') as h5f:
            if 'label' not in h5f:
                continue
            lbl = h5f['label'][:]
            vals, counts = np.unique(lbl, return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                if 1 <= int(v) <= num_fg_classes:
                    vox_mat[idx, int(v) - 1] = float(c)
                    valid_scan = True

    if not valid_scan:
        logging.warning('[StratifiedSplit] No multi-class foreground annotations found; falling back to first %d indices.', labelnum)
        return list(range(labelnum))

    strong_mat = (vox_mat >= strong_voxel_thresh).astype(np.int64)
    target_cases_per_class = max(1, labelnum // 2)

    selected: List[int] = []
    remaining = set(range(pool_size))

    for _ in range(labelnum):
        cur_strong = strong_mat[selected].sum(axis=0) if selected else np.zeros(num_fg_classes, dtype=np.int64)
        cur_vox = vox_mat[selected].sum(axis=0) if selected else np.zeros(num_fg_classes, dtype=np.float64)

        best_idx = None
        best_score = -1e18
        for idx in sorted(remaining):
            new_strong = cur_strong + strong_mat[idx]
            new_vox = cur_vox + vox_mat[idx]
            min_cases = float(new_strong.min())
            sum_covered = float(np.minimum(new_strong, target_cases_per_class).sum())
            log_vox = np.log1p(new_vox)
            vox_balance = float(log_vox.sum() - STRATIFIED_VOXEL_STD_PENALTY * np.std(log_vox))
            score = (
                min_cases * STRATIFIED_MIN_CASE_WEIGHT
                + sum_covered * STRATIFIED_COVERAGE_WEIGHT
                + vox_balance
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(int(best_idx))
        remaining.remove(int(best_idx))

    selected_sorted = sorted(selected)
    sel_has = (vox_mat[selected_sorted] > 0).sum(axis=0)
    sel_vox = vox_mat[selected_sorted].sum(axis=0).astype(np.int64)
    summary_parts = [f"C{c+1}:{int(sel_has[c])}/{labelnum}({int(sel_vox[c])}v)" for c in range(num_fg_classes)]
    logging.info(
        "[StratifiedSplit] Selected %d/%d labeled indices: %s | Class coverage: %s",
        labelnum, pool_size, selected_sorted, ", ".join(summary_parts)
    )
    return selected_sorted


def resolve_labeled_unlabeled_indices(
    db_train: Dataset,
    cfg: Config,
    labelnum: int,
    max_samples: int,
) -> Tuple[List[int], List[int]]:
    """
    Resolve labeled and unlabeled index partitions within `[0, max_samples)`.
    Supports:
    - `labeled_sampling='first'` (default): `[0..labelnum-1]` vs `[labelnum..max_samples-1]`
    - `labeled_sampling='stratified'`: Multi-label class-balanced selection of `labelnum` cases
    - `labeled_indices=[...]`: Explicit user-specified list of indices
    """
    explicit_indices = getattr(cfg.data, 'labeled_indices', None)
    sampling_mode = str(getattr(cfg.data, 'labeled_sampling', 'first') or 'first').lower()

    if explicit_indices is not None and len(explicit_indices) > 0:
        labeled_idxs = sorted(int(i) for i in explicit_indices)
        if any(i < 0 or i >= max_samples for i in labeled_idxs):
            raise ValueError(
                f"data.labeled_indices contains index outside [0, max_samples={max_samples}): {labeled_idxs}"
            )
    elif sampling_mode == 'stratified':
        strong_thresh_cfg = getattr(cfg.data, 'strong_voxel_thresh', None)
        strong_thresh = (
            DEFAULT_STRATIFIED_STRONG_VOXEL_THRESH
            if strong_thresh_cfg is None
            else int(strong_thresh_cfg)
        )
        labeled_idxs = select_stratified_labeled_indices(
            db_train=db_train,
            labelnum=labelnum,
            max_samples=max_samples,
            strong_voxel_thresh=strong_thresh,
        )
    else:
        labeled_idxs = list(range(labelnum))

    labeled_set = set(labeled_idxs)
    unlabeled_idxs = [i for i in range(max_samples) if i not in labeled_set]
    return labeled_idxs, unlabeled_idxs


def build_bhsd_dataloaders(
    *,
    cfg: Config,
    batch_layout: BatchLayout,
    dataset_name: str,
    fold: int,
    is_ddp: bool,
    rank: int,
    world_size: int,
    patch_size: Tuple[int, int, int],
    num_classes: int,
    is_binary: bool,
    class_names: Sequence[str],
) -> DataLoaderBundle:
    """
    Construct training and validation datasets, verify split isolation,
    assign labeled/unlabeled indices, and build DataLoaders with appropriate samplers.
    """
    root_path = _resolve_data_root(cfg)
    use_extended_da = bool(getattr(cfg.data, 'extended_aug', True))
    fg_prob_cfg = getattr(cfg.data, 'foreground_prob', None)
    aug_preset = str(getattr(cfg.data, 'aug_preset', 'legacy') or 'legacy').lower()
    train_transforms = build_bhsd_transforms(
        patch_size=patch_size,
        extended_aug=use_extended_da,
        foreground_prob=fg_prob_cfg,
        aug_config=_extract_aug_config(cfg),
        aug_preset=aug_preset,
    )

    db_train = DATASETS.build(
        dataset_name,
        base_dir=root_path,
        split='train',
        binary=is_binary,
        patch_size=patch_size,
        fold=fold,
        transform=train_transforms,
        num_classes=num_classes,
        class_names=class_names,
        intensity_units=getattr(cfg.data, 'intensity_units', None),
        normalization=getattr(cfg.data, 'normalization', None)
    )

    db_val = DATASETS.build(
        dataset_name,
        base_dir=root_path,
        split='val',
        binary=is_binary,
        patch_size=patch_size,
        fold=fold,
        num_classes=num_classes,
        class_names=class_names,
        intensity_units=getattr(cfg.data, 'intensity_units', None),
        normalization=getattr(cfg.data, 'normalization', None)
    )

    train_cases = set(getattr(db_train, 'image_list', []))
    val_cases = set(getattr(db_val, 'image_list', []))
    overlap = train_cases & val_cases
    if overlap:
        raise ValueError(f'Train/validation split overlap: {sorted(overlap)[:5]}')

    batch_size = batch_layout.local_batch_size
    labelnum = min(int(getattr(cfg.data, 'labelnum', 15)), len(db_train))
    max_samples_cfg = int(getattr(cfg.data, 'max_samples', -1))
    max_samples = len(db_train) if max_samples_cfg <= 0 else min(max_samples_cfg, len(db_train))

    if labelnum > max_samples:
        raise ValueError('labelnum cannot exceed max_samples')
    labeled_idxs, unlabeled_idxs = resolve_labeled_unlabeled_indices(
        db_train=db_train,
        cfg=cfg,
        labelnum=labelnum,
        max_samples=max_samples,
    )
    labelnum = len(labeled_idxs)
    # Hide unlabeled annotations before any label-dependent augmentation.
    db_train.labeled_indices = set(labeled_idxs)
    db_train.unlabeled_indices = set(unlabeled_idxs)
    db_train.max_samples = max_samples

    method_name = str(getattr(cfg.method, 'name', 'pick')).lower()
    if method_name == 'supervised':
        if labelnum < batch_layout.global_batch_size:
            raise ValueError('Labeled pool must contain at least one global supervised batch.')
        labeled_dataset = Subset(db_train, labeled_idxs)
        train_sampler = None
        if is_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                labeled_dataset, num_replicas=world_size, rank=rank,
                shuffle=True, drop_last=True,
            )
        num_workers = int(getattr(cfg.data, 'num_workers', 2 if is_ddp else 4))
        pin_memory = bool(getattr(cfg.data, 'pin_memory', is_ddp))
        drop_last = bool(getattr(cfg.data, 'drop_last', True))
        trainloader = DataLoader(
            labeled_dataset, batch_size=batch_size, sampler=train_sampler,
            shuffle=train_sampler is None, drop_last=drop_last,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        return DataLoaderBundle(
            db_train=db_train,
            db_val=db_val,
            trainloader=trainloader,
            train_sampler=train_sampler,
            batch_sampler=None
        )

    if labelnum < batch_layout.global_labeled_bs or len(unlabeled_idxs) < batch_layout.global_unlabeled_bs:
        raise ValueError('Dataset must contain at least one global batch of both labeled and unlabeled cases.')

    unlabeled_bs = batch_layout.local_unlabeled_bs
    if is_ddp:
        batch_sampler = DistributedTwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs,
            batch_size=batch_size, secondary_batch_size=unlabeled_bs,
            num_replicas=world_size, rank=rank,
            seed=int(getattr(cfg.experiment, 'seed', 1337)),
        )
    else:
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, batch_size, unlabeled_bs,
        )

    num_workers = int(getattr(cfg.data, 'num_workers', 2 if is_ddp else 4))
    pin_mem = getattr(cfg.data, 'pin_memory', None)
    pin_memory = (is_ddp if pin_mem is None else bool(pin_mem))
    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return DataLoaderBundle(
        db_train=db_train,
        db_val=db_val,
        trainloader=trainloader,
        train_sampler=None,
        batch_sampler=batch_sampler
    )

