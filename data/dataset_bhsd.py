"""
BHSD (Brain Hemorrhage Segmentation Dataset) PyTorch Dataset.
Supports both Multi-class (6 classes: BG, EDH, IPH, IVH, SAH, SDH) and Binary (BG vs Hemorrhage) tasks
on 3D volumetric CT scans stored in HDF5 (.h5) format.
Integrated with 5-fold cross-validation and optional MIM masking.
"""

from typing import Callable, Optional, Tuple, Dict, Any, Union, List
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from core.registry import DATASETS
from data.masking import MaskGeneratorBHSD

BHSD_MULTICLASS_NAMES = [
    "Background",
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural"
]

BHSD_BINARY_NAMES = [
    "Background",
    "Hemorrhage"
]


@DATASETS.register("bhsd")
@DATASETS.register("bhsd_dataset")
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
        class_names: Optional custom class names list.
        return_mim_mask: If True, returns (sample, mim_mask) tuple for algorithms requiring
                         pre-computed MIM masks (e.g. legacy PICK). Defaults to False.
    """
    def __init__(
        self,
        base_dir: str,
        split: str = 'train',
        num: Optional[int] = None,
        transform: Optional[Callable] = None,
        binary: bool = False,
        patch_size: Tuple[int, int, int] = (32, 160, 160),
        fold: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        return_mim_mask: bool = False,
        num_classes: Optional[int] = None,
        **kwargs
    ):
        super(BHSDDataset, self).__init__()
        self.base_dir = os.path.abspath(base_dir) if base_dir else ""
        self.split = str(split).lower()
        self.fold = fold
        self.transform = transform
        self.patch_size = tuple(patch_size) if patch_size else (32, 160, 160)
        self.binary = bool(binary)
        self.return_mim_mask = return_mim_mask
        self._mask_generator = None

        if self.binary:
            self.num_classes = 2
            self.class_names = class_names or BHSD_BINARY_NAMES
        else:
            self.num_classes = num_classes if num_classes is not None else 6
            self.class_names = class_names or BHSD_MULTICLASS_NAMES

        # Load cases for this split
        self.image_list = self._load_split_cases()

        if num is not None and num > 0 and self.split == 'train':
            self.image_list = self.image_list[:num]

        task_type = "Binary (Hemorrhage vs Background)" if self.binary else f"Multi-class ({self.num_classes} classes)"
        fold_str = f"Fold {self.fold}" if self.fold is not None else "Default Split"
        print(f"[BHSDDataset] {fold_str} | Split '{self.split}' | Task: {task_type} | Samples: {len(self.image_list)}")

    def _load_split_cases(self) -> List[str]:
        list_file = None
        if self.fold is not None:
            fold_file = os.path.join(self.base_dir, "splits", f"fold{self.fold}_{self.split}.txt")
            if os.path.exists(fold_file):
                list_file = fold_file

        if list_file is None:
            list_file = os.path.join(self.base_dir, f"{self.split}.txt")
            if not os.path.exists(list_file):
                alt_map = {'train': 'train.list', 'val': 'val.list', 'test': 'test.list'}
                list_file = os.path.join(self.base_dir, alt_map.get(self.split, f"{self.split}.txt"))

        if not os.path.exists(list_file):
            raise FileNotFoundError(
                f"Split list file not found in {self.base_dir}! (looked for fold {self.fold} {self.split})"
            )

        with open(list_file, 'r', encoding='utf-8') as f:
            cases = [line.strip() for line in f if line.strip()]
        return cases

    def get_mask_generator(self) -> MaskGeneratorBHSD:
        if self._mask_generator is None:
            self._mask_generator = MaskGeneratorBHSD(input_size=self.patch_size)
        return self._mask_generator

    def __len__(self) -> int:
        return len(self.image_list)

    def get_case_name(self, idx: int) -> str:
        return self.image_list[idx]

    def __getitem__(self, idx: int) -> Union[Dict[str, Any], Tuple[Dict[str, Any], np.ndarray]]:
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

        if self.return_mim_mask:
            mim_mask = self.get_mask_generator()()
            return sample, mim_mask

        return sample

    def __repr__(self) -> str:
        fold_str = f"Fold {self.fold}" if self.fold is not None else "Default Split"
        task_str = "Binary" if self.binary else f"Multiclass ({self.num_classes} classes)"
        return (
            f"<BHSDDataset | {fold_str} | Split: '{self.split}' | "
            f"Task: {task_str} | Samples: {len(self.image_list)}>"
        )
