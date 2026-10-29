"""
BHSD (Brain Hemorrhage Segmentation Dataset) PyTorch Dataset.
Supports both Multi-class (6 classes: BG, EDH, IPH, IVH, SAH, SDH) and Binary (BG vs Hemorrhage) tasks
on 3D volumetric CT scans stored in HDF5 (.h5) format.
Integrated with 5-fold cross-validation and optional MIM masking.
"""

from typing import Callable, Optional, Tuple, Dict, Any, Union, List
import os
import ast
import hashlib
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from core.registry import DATASETS
from data.masking import MaskGeneratorBHSD
from data.constants import BHSD_MULTICLASS_NAMES, BHSD_BINARY_NAMES


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
        self.intensity_units = kwargs.get('intensity_units')
        self.normalization = kwargs.get('normalization')
        self._mask_generator = None
        self.labeled_indices = None
        self.unlabeled_indices = None
        self.max_samples = kwargs.get('max_samples')
        self._fallback_depth = 0

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
        if self.split not in ('train', 'val', 'test'):
            raise ValueError(f'Unsupported split: {self.split}')
        if self.fold is not None:
            if not 0 <= int(self.fold) < 5:
                raise ValueError('fold must be in 0..4')
            list_file = os.path.join(self.base_dir, 'splits', f'fold{self.fold}_{self.split}.txt')
        else:
            list_file = os.path.join(self.base_dir, f'{self.split}.txt')
            if not os.path.exists(list_file):
                list_file = os.path.join(self.base_dir, f'{self.split}.list')

        if not os.path.exists(list_file):
            raise FileNotFoundError(
                f"Split list file not found in {self.base_dir}! (looked for fold {self.fold} {self.split})"
            )

        with open(list_file, 'rb') as stream:
            self.split_fingerprint = hashlib.sha256(stream.read()).hexdigest()
        with open(list_file, 'r', encoding='utf-8') as f:
            cases = [line.strip() for line in f if line.strip()]
        if not cases or len(set(cases)) != len(cases):
            raise ValueError(f'Empty split or duplicate case IDs: {list_file}')
        if self.fold is not None and self.split in ('train', 'val'):
            counterpart = 'val' if self.split == 'train' else 'train'
            other = os.path.join(self.base_dir, 'splits', f'fold{self.fold}_{counterpart}.txt')
            if os.path.isfile(other):
                with open(other, encoding='utf-8') as stream:
                    overlap = set(cases) & {line.strip() for line in stream if line.strip()}
                if overlap:
                    raise ValueError(f'Train/validation overlap in fold {self.fold}: {sorted(overlap)[:5]}')
        return cases

    def get_mask_generator(self) -> MaskGeneratorBHSD:
        if self._mask_generator is None:
            self._mask_generator = MaskGeneratorBHSD(input_size=self.patch_size)
        return self._mask_generator

    def __len__(self) -> int:
        return len(self.image_list)

    def get_case_name(self, idx: int) -> str:
        return self.image_list[idx]

    def _get_partition_safe_index(self, idx: int) -> int:
        """Find an alternative sample index strictly within the same labeled or unlabeled partition."""
        if self.labeled_indices is not None:
            if idx in self.labeled_indices:
                pool = sorted(self.labeled_indices)
                name = "labeled"
            else:
                if self.unlabeled_indices is not None:
                    pool = sorted(self.unlabeled_indices)
                else:
                    max_idx = self.max_samples if (self.max_samples is not None and 0 < self.max_samples <= len(self.image_list)) else len(self.image_list)
                    pool = [i for i in range(max_idx) if i not in self.labeled_indices]
                name = "unlabeled"
            if len(pool) <= 1:
                raise ValueError(f"No alternative {name} sample available for corrupt index {idx}")
            current_pos = pool.index(idx) if idx in pool else 0
            next_pos = (current_pos + 1) % len(pool)
            return pool[next_pos]
        else:
            max_idx = self.max_samples if (self.max_samples is not None and 0 < self.max_samples <= len(self.image_list)) else len(self.image_list)
            if max_idx <= 1:
                raise ValueError(f"No alternative sample available for corrupt index {idx}")
            return (idx + 1) % max_idx

    def __getitem__(self, idx: int) -> Union[Dict[str, Any], Tuple[Dict[str, Any], np.ndarray]]:
        case_name = self.image_list[idx]
        h5_path = os.path.join(self.base_dir, "data", f"{case_name}.h5")

        with h5py.File(h5_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]
            metadata = {}
            if self.split != 'train':
                for source, key in [
                    ('target_spacing', 'spacing'),
                    ('orig_spacing', 'orig_spacing'),
                    ('orig_shape', 'orig_shape'),
                    ('bbox_used', 'bbox_used'),
                    ('cropped_shape', 'cropped_shape'),
                    ('final_shape', 'final_shape'),
                    ('norm_params', 'norm_params'),
                    ('clip_range', 'clip_range')
                ]:
                    value = h5f.attrs.get(source)
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    if isinstance(value, str):
                        try:
                            value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            if key in ('spacing', 'norm_params', 'clip_range'):
                                raise ValueError(f'{case_name}: invalid metadata {source}')
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    if value is not None:
                        metadata[key] = value
                metadata.update(intensity_units=self.intensity_units, normalization=self.normalization)

        if (image.ndim != 3 or image.shape != label.shape or any(dim == 0 for dim in image.shape)
                or not np.isfinite(image).all() or not np.isfinite(label).all()):
            if self.split == 'train':
                if self._fallback_depth >= 5:
                    raise ValueError(f'{case_name}: exceeded fallback depth limit for corrupt volumes')
                safe_idx = self._get_partition_safe_index(idx)
                partition_str = 'labeled' if (self.labeled_indices is not None and idx in self.labeled_indices) else 'unlabeled'
                import logging
                logging.warning(
                    'Corrupt/empty volume %s at index %d (%s); falling back to index %d in the same partition',
                    case_name, idx, partition_str, safe_idx
                )
                try:
                    self._fallback_depth += 1
                    return self.__getitem__(safe_idx)
                finally:
                    self._fallback_depth -= 1
            raise ValueError(f'{case_name}: expected matching finite 3D image and label with non-zero dimensions')
        if label.size > 0:
            if not np.all(np.equal(np.mod(label, 1), 0)):
                raise ValueError(f'{case_name}: non-integer label values detected')
            if label.min() < 0:
                raise ValueError(f'{case_name}: negative labels not allowed')
            if not self.binary and label.max() >= self.num_classes:
                raise ValueError(f'{case_name}: labels outside configured class range')
            if self.binary and label.max() > 5:
                raise ValueError(f'{case_name}: labels outside valid source class range')

        if self.labeled_indices is not None and idx not in self.labeled_indices:
            label = np.zeros_like(label)
        if self.binary:
            label = (label > 0).astype(np.uint8)
        else:
            label = label.astype(np.uint8)

        sample = {
            'image': image,
            'label': label,
            'case': case_name
        }

        if self.split != 'train':
            sample['metadata'] = metadata

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
