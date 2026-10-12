"""
Unified BHSD (Brain Hemorrhage Segmentation Dataset) PyTorch Dataset.
Supports both Binary (Hemorrhage vs Background) and Multi-class (6 classes: 0..5) tasks
on the same underlying HDF5 dataset.
Integrated with 5-fold cross-validation and MIM token masking.
"""

from typing import Callable, Optional, Tuple, Dict, Any, Union
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.masking import MaskGeneratorBHSD


class BHSDDataset(Dataset):
    """
    BHSD Dataset loader for semi-supervised volumetric CT segmentation.
    
    Args:
        base_dir: Root directory containing 'data/' (.h5 files) and 'splits/'.
        split: 'train', 'val', or 'test'.
        num: Optional upper limit on training cases.
        transform: Callable augmentation transforms pipeline.
        binary: If True, merges all lesion classes 1..5 into label 1 (binary task).
                If False, retains multi-class labels 0..5 (Background + 5 hemorrhage types).
        patch_size: 3D patch dimensions (Z, Y, X). Default (32, 160, 160).
        fold: Fold index (0..4) for 5-fold cross-validation.
    """
    def __init__(
        self,
        base_dir: str,
        split: str = 'train',
        num: Optional[int] = None,
        transform: Optional[Callable] = None,
        binary: bool = False,
        patch_size: Tuple[int, int, int] = (32, 160, 160),
        fold: Optional[int] = None
    ):
        self.base_dir = os.path.abspath(base_dir)
        self.split = split
        self.transform = transform
        self.binary = binary
        self.patch_size = patch_size
        self.fold = fold
        self.mask_generator = MaskGeneratorBHSD(input_size=patch_size)

        list_file = None
        if fold is not None:
            fold_file = os.path.join(self.base_dir, "splits", f"fold{fold}_{split}.txt")
            if os.path.exists(fold_file):
                list_file = fold_file

        if list_file is None:
            list_file = os.path.join(self.base_dir, f"{split}.txt")
            if not os.path.exists(list_file):
                alt_map = {'train': 'train.list', 'val': 'val.list', 'test': 'test.list'}
                list_file = os.path.join(self.base_dir, alt_map.get(split, f"{split}.txt"))

        if not os.path.exists(list_file):
            raise FileNotFoundError(
                f"Split list file not found in {self.base_dir}! (looked for fold {fold} {split})"
            )

        with open(list_file, 'r') as f:
            self.image_list = [line.strip() for line in f if line.strip()]

        if num is not None and split == 'train':
            self.image_list = self.image_list[:num]

        task_type = "Binary (Hemorrhage vs Background)" if self.binary else "Multi-class (6 classes: 0..5)"
        fold_str = f"Fold {fold}" if fold is not None else "Default Split"
        print(f"[BHSDDataset] {fold_str} | Split '{split}' | Task: {task_type} | Samples: {len(self.image_list)}")

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], np.ndarray]:
        case_name = self.image_list[idx]
        h5_path = os.path.join(self.base_dir, "data", f"{case_name}.h5")

        with h5py.File(h5_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]

        if self.binary:
            label = (label > 0).astype(np.uint8)
        else:
            label = label.astype(np.uint8)

        sample = {
            'image': image,
            'label': label,
            'case': case_name
        }

        if self.transform is not None:
            sample = self.transform(sample)

        mim_mask = self.mask_generator()
        return sample, mim_mask
