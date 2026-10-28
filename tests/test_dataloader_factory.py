"""Unit tests for data/dataloader_factory.py."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from core.config_parser import Config
from core.batching import resolve_batch_layout
from data.dataloader_factory import build_bhsd_dataloaders, build_bhsd_transforms, DataLoaderBundle


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

