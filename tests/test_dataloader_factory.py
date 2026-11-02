"""Unit tests for data/dataloader_factory.py."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset

from core.config_parser import Config
from core.batching import resolve_batch_layout
from data.dataloader_factory import (
    build_bhsd_dataloaders,
    build_bhsd_transforms,
    DataLoaderBundle,
    select_stratified_labeled_indices,
    resolve_labeled_unlabeled_indices,
)


class TestDataLoaderFactory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_build_bhsd_transforms(self):
        t1 = build_bhsd_transforms((32, 64, 64), extended_aug=True)
        self.assertEqual([type(t).__name__ for t in t1.transforms], [
            'RandomRotFlip', 'RandomScaling3D', 'GammaTransform',
            'GaussianBlur3D', 'RandomCrop', 'ToTensor'])
        scale, gamma, blur, crop = t1.transforms[1:5]
        self.assertEqual((scale.scale_range, scale.prob), ((0.85, 1.15), 0.2))
        self.assertEqual((gamma.gamma_range, gamma.prob), ((0.7, 1.5), 0.2))
        self.assertEqual((blur.sigma_range, blur.prob), ((0.5, 1.0), 0.15))
        self.assertEqual(crop.output_size, (32, 64, 64))

        t2 = build_bhsd_transforms((32, 64, 64), extended_aug=False)
        self.assertEqual([type(t).__name__ for t in t2.transforms],
                         ['RandomRotFlip', 'RandomCrop', 'ToTensor'])

    def test_build_bhsd_dataloaders_supervised(self):
        cfg = Config({
            'data': {'batch_size': 4, 'labeled_bs': 4, 'labelnum': 4, 'extended_aug': False, 'root_path': str(self.root)},
            'method': {'name': 'supervised'},
            'experiment': {'seed': 42}
        })
        layout = resolve_batch_layout(4, 4, 1, 'supervised')

        mock_train = TensorDataset(torch.arange(10))
        mock_train.image_list = ['case1', 'case2']
        mock_val = TensorDataset(torch.arange(5))
        mock_val.image_list = ['case3', 'case4']

        def fake_build(name, **kwargs):
            return mock_train if kwargs.get('split') == 'train' else mock_val

        with patch('data.dataloader_factory.DATASETS.build', side_effect=fake_build):
            bundle = build_bhsd_dataloaders(
                cfg=cfg,
                batch_layout=layout,
                dataset_name='bhsd',
                fold=0,
                is_ddp=False,
                rank=0,
                world_size=1,
                patch_size=(32, 64, 64),
                num_classes=2,
                is_binary=True,
                class_names=['BG', 'Hem']
            )

        self.assertIsInstance(bundle, DataLoaderBundle)
        self.assertIsInstance(bundle.trainloader.dataset, torch.utils.data.Subset)
        self.assertEqual(len(bundle.trainloader.dataset), 4)
        self.assertIsNone(bundle.batch_sampler)

    def test_build_bhsd_dataloaders_ssl(self):
        cfg = Config({
            'data': {'batch_size': 4, 'labeled_bs': 2, 'labelnum': 4, 'extended_aug': False, 'root_path': str(self.root)},
            'method': {'name': 'mean_teacher'},
            'experiment': {'seed': 42}
        })
        layout = resolve_batch_layout(4, 2, 1, 'mean_teacher')

        mock_train = TensorDataset(torch.arange(10))
        mock_train.image_list = ['case1', 'case2']
        mock_val = TensorDataset(torch.arange(5))
        mock_val.image_list = ['case3', 'case4']

        def fake_build(name, **kwargs):
            return mock_train if kwargs.get('split') == 'train' else mock_val

        with patch('data.dataloader_factory.DATASETS.build', side_effect=fake_build):
            bundle = build_bhsd_dataloaders(
                cfg=cfg,
                batch_layout=layout,
                dataset_name='bhsd',
                fold=0,
                is_ddp=False,
                rank=0,
                world_size=1,
                patch_size=(32, 64, 64),
                num_classes=2,
                is_binary=True,
                class_names=['BG', 'Hem']
            )

        self.assertIsInstance(bundle, DataLoaderBundle)
        self.assertIsNotNone(bundle.batch_sampler)
        batch = next(iter(bundle.trainloader))
        self.assertEqual(len(batch[0]), 4)

    def test_stratified_and_explicit_labeled_indices(self):
        data_dir = self.root / 'data'
        data_dir.mkdir(exist_ok=True)
        # Create 10 synthetic .h5 cases where rare classes are at later indices
        case_classes = [
            [2],       # idx 0: only C2
            [2],       # idx 1: only C2
            [2],       # idx 2: only C2
            [1, 5],    # idx 3: C1, C5
            [3, 4],    # idx 4: C3, C4
            [1, 2, 3], # idx 5: C1, C2, C3
            [4, 5],    # idx 6: C4, C5
            [2],       # idx 7: only C2
            [0],       # idx 8: unlabeled pool (outside max_samples=8)
            [0],       # idx 9: unlabeled pool (outside max_samples=8)
        ]
        image_list = []
        for idx, cls_list in enumerate(case_classes):
            cname = f'case_{idx:02d}'
            image_list.append(cname)
            lbl = np.zeros((10, 20, 20), dtype=np.uint8)
            for c_idx, c in enumerate(cls_list):
                if c > 0:
                    lbl[c_idx * 2:(c_idx + 1) * 2, :, :] = c  # 2*20*20 = 800 -> let's make >=1000 voxels
                    lbl[c_idx * 2:(c_idx + 1) * 2 + 1, :, :] = c  # 3*20*20 = 1200 voxels
            with h5py.File(data_dir / f'{cname}.h5', 'w') as hf:
                hf['image'] = np.zeros((10, 20, 20), dtype=np.float32)
                hf['label'] = lbl

        mock_train = TensorDataset(torch.arange(10))
        mock_train.base_dir = str(self.root)
        mock_train.image_list = image_list

        # 1. Stratified selection of 4 cases out of max_samples=8 must cover all 5 foreground classes
        cfg_strat = Config({
            'data': {
                'labelnum': 4,
                'max_samples': 8,
                'labeled_sampling': 'stratified',
            }
        })
        lab_idxs, unlab_idxs = resolve_labeled_unlabeled_indices(mock_train, cfg_strat, labelnum=4, max_samples=8)
        self.assertEqual(len(lab_idxs), 4)
        self.assertEqual(len(unlab_idxs), 4)
        self.assertEqual(set(lab_idxs) | set(unlab_idxs), set(range(8)))
        self.assertEqual(set(lab_idxs) & set(unlab_idxs), set())
        # Verify all 5 classes (1..5) are present in the 4 stratified cases
        covered = set()
        for i in lab_idxs:
            covered.update(case_classes[i])
        self.assertTrue({1, 2, 3, 4, 5}.issubset(covered))

        # 2. Default 'first' mode preserves [0..3] and [4..7]
        cfg_first = Config({
            'data': {
                'labelnum': 4,
                'max_samples': 8,
            }
        })
        lab_first, unlab_first = resolve_labeled_unlabeled_indices(mock_train, cfg_first, labelnum=4, max_samples=8)
        self.assertEqual(lab_first, [0, 1, 2, 3])
        self.assertEqual(unlab_first, [4, 5, 6, 7])


