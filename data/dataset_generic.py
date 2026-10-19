"""
Generic Volumetric HDF5 Medical Dataset Loader.
Allows loading any volumetric medical segmentation dataset structured as HDF5 (.h5) files
with train/val/test split lists. Compatible with datasets like ACDC, LA Heart, Pancreas, BraTS.
"""

from typing import Callable, Optional, Tuple, Dict, Any, Union, List
import os
import h5py
import numpy as np

from core.registry import DATASETS
from data.base_dataset import BaseMedicalDataset


@DATASETS.register("generic_h5")
@DATASETS.register("acdc")
@DATASETS.register("la_heart")
@DATASETS.register("pancreas")
class GenericH5Dataset(BaseMedicalDataset):
    """
    Generic Volumetric HDF5 Medical Dataset.
    Expects directory structure:
        <base_dir>/
            data/           (contains <case_name>.h5 with 'image' and 'label')
            splits/         (optional: fold<fold>_train.txt, fold<fold>_val.txt)
            train.txt / val.txt / test.txt
    """
    def __init__(
        self,
        base_dir: str,
        split: str = 'train',
        num: Optional[int] = None,
        transform: Optional[Callable] = None,
        binary: bool = False,
        patch_size: Tuple[int, ...] = (32, 160, 160),
        fold: Optional[int] = None,
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        image_key: str = 'image',
        label_key: str = 'label',
        **kwargs
    ):
        self.image_key = image_key
        self.label_key = label_key
        super(GenericH5Dataset, self).__init__(
            base_dir=base_dir,
            split=split,
            fold=fold,
            transform=transform,
            patch_size=patch_size,
            binary=binary,
            num_classes=num_classes,
            class_names=class_names,
            max_samples=num,
            **kwargs
        )

    def _load_split_cases(self) -> List[str]:
        if not os.path.exists(self.base_dir):
            return []

        list_file = None
        if self.fold is not None:
            fold_file = os.path.join(self.base_dir, "splits", f"fold{self.fold}_{self.split}.txt")
            if os.path.exists(fold_file):
                list_file = fold_file

        if list_file is None:
            for cand in [f"{self.split}.txt", f"{self.split}.list", f"{self.split}_slices.list"]:
                cand_path = os.path.join(self.base_dir, cand)
                if os.path.exists(cand_path):
                    list_file = cand_path
                    break

        if list_file and os.path.exists(list_file):
            with open(list_file, 'r', encoding='utf-8') as f:
                cases = [line.strip() for line in f if line.strip()]
            return cases

        # Fallback: scan data directory for .h5 files
        data_dir = os.path.join(self.base_dir, "data")
        if os.path.exists(data_dir):
            cases = [
                os.path.splitext(f)[0]
                for f in sorted(os.listdir(data_dir))
                if f.endswith('.h5')
            ]
            return cases

        return []

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_name = self.image_list[idx]
        h5_path = os.path.join(self.base_dir, "data", f"{case_name}.h5")
        if not os.path.exists(h5_path):
            h5_path = os.path.join(self.base_dir, f"{case_name}.h5")

        with h5py.File(h5_path, 'r') as h5f:
            image = h5f[self.image_key][:]
            label = h5f[self.label_key][:]

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

        return sample
