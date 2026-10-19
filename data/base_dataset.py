"""
Abstract Base Medical Dataset for 2D/3D Semi-Supervised Segmentation.
Standardizes dataset contracts across different medical imaging datasets (BHSD, ACDC, LA, etc.).
"""

import abc
import os
import logging
from typing import Callable, Optional, Tuple, Dict, Any, List, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseMedicalDataset(Dataset, abc.ABC):
    """
    Abstract Base Class for Volumetric / 2D Medical Segmentation Datasets.
    
    Guarantees consistent data interface for all SSL training and evaluation algorithms:
    - Every __getitem__ returns a Dict containing at least:
        {
            'image': torch.Tensor or np.ndarray (C, D, H, W) or (D, H, W),
            'label': torch.Tensor or np.ndarray (D, H, W) integer mask,
            'case': str case identifier
        }
    """
    def __init__(
        self,
        base_dir: str,
        split: str = 'train',
        fold: Optional[int] = None,
        transform: Optional[Callable] = None,
        patch_size: Tuple[int, ...] = (32, 160, 160),
        binary: bool = False,
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        **kwargs
    ):
        super(BaseMedicalDataset, self).__init__()
        self.base_dir = os.path.abspath(base_dir) if base_dir else ""
        self.split = str(split).lower()
        self.fold = fold
        self.transform = transform
        self.patch_size = tuple(patch_size) if patch_size else (32, 160, 160)
        self.binary = bool(binary)
        self.max_samples = max_samples
        self.kwargs = kwargs

        # Resolve number of classes and names
        if self.binary:
            self.num_classes = 2
            self.class_names = class_names or ["Background", "Lesion"]
        else:
            self.num_classes = num_classes if num_classes is not None else 2
            self.class_names = class_names or [f"Class_{i}" for i in range(self.num_classes)]

        # Load cases for this split
        self.image_list = self._load_split_cases()

        if self.max_samples is not None and self.max_samples > 0 and self.split == 'train':
            self.image_list = self.image_list[:self.max_samples]

    @abc.abstractmethod
    def _load_split_cases(self) -> List[str]:
        """Load list of case identifiers for the requested split and fold."""
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Any]]:
        """Retrieve and process a sample by index."""
        pass

    def __len__(self) -> int:
        return len(self.image_list)

    def get_case_name(self, idx: int) -> str:
        return self.image_list[idx]

    def __repr__(self) -> str:
        fold_str = f"Fold {self.fold}" if self.fold is not None else "Default Split"
        task_str = "Binary" if self.binary else f"Multiclass ({self.num_classes} classes)"
        return (
            f"<{self.__class__.__name__} | {fold_str} | Split: '{self.split}' | "
            f"Task: {task_str} | Samples: {len(self.image_list)}>"
        )
