"""
Data Factory for 3D Semi-Supervised Medical Image Segmentation (BHSD).
Constructs 3D augmentation pipelines, datasets, split partition indices,
and supervised / semi-supervised data loaders with single-GPU and DDP samplers.
"""

from dataclasses import dataclass
from typing import Tuple, Sequence, Optional
import torch
from torch.utils.data import DataLoader, Subset, Dataset, Sampler
from torchvision import transforms

from core.config_parser import Config
from core.batching import BatchLayout
from core.registry import DATASETS
from data.samplers import TwoStreamBatchSampler, DistributedTwoStreamBatchSampler
from data.transforms3d import (
    RandomRotFlip,
    RandomCrop,
    ToTensor,
    RandomScaling3D,
    GaussianBlur3D,
    GammaTransform,
)


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
    extended_aug: bool = True
) -> transforms.Compose:
    """Build 3D data augmentation pipeline for training."""
    aug_list = [
        RandomRotFlip(),
    ]
    if extended_aug:
        aug_list.extend([
            RandomScaling3D(scale_range=(0.85, 1.15), prob=0.2),
            GammaTransform(gamma_range=(0.7, 1.5), prob=0.2),
            GaussianBlur3D(sigma_range=(0.5, 1.0), prob=0.15),
        ])
    aug_list.extend([
        RandomCrop(patch_size),
        ToTensor()
    ])
    return transforms.Compose(aug_list)


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
    root_path = getattr(cfg.data, 'root_path', '/home/u001015/notebook/preprocessed_bhsd_pick/')
    use_extended_da = bool(getattr(cfg.data, 'extended_aug', True))
    train_transforms = build_bhsd_transforms(patch_size=patch_size, extended_aug=use_extended_da)

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
    labeled_idxs = list(range(labelnum))
    # Hide unlabeled annotations before any label-dependent augmentation.
    db_train.labeled_indices = set(labeled_idxs)
    unlabeled_idxs = list(range(labelnum, max_samples))
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
        trainloader = DataLoader(
            labeled_dataset, batch_size=batch_size, sampler=train_sampler,
            shuffle=train_sampler is None, drop_last=True,
            num_workers=2 if is_ddp else 4, pin_memory=is_ddp,
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
        trainloader = DataLoader(
            db_train, batch_sampler=batch_sampler,
            num_workers=2, pin_memory=True,
        )
    else:
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, batch_size, unlabeled_bs,
        )
        trainloader = DataLoader(
            db_train, batch_sampler=batch_sampler,
            num_workers=4, pin_memory=False,
        )

    return DataLoaderBundle(
        db_train=db_train,
        db_val=db_val,
        trainloader=trainloader,
        train_sampler=None,
        batch_sampler=batch_sampler
    )

