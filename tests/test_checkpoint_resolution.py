"""Checkpoint selection, provenance, and strict evaluation regression tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from core.registry import MODELS
from utils.checkpoint import save_checkpoint
from utils.checkpoint_resolver import resolve_checkpoint
from utils.evaluation_checkpoint import build_evaluation_model
from evaluation.benchmark import benchmark_5folds


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, relative, **kwargs):
        path = self.root / relative
        save_checkpoint(path, torch.nn.Linear(2, 2), **kwargs)
        return path

    def test_best_and_legacy(self):
        self.save('fold_0/checkpoint_latest.pth')
        legacy = self.save('fold_0/VNet_best_model.pth')
        self.assertEqual(resolve_checkpoint(self.root, 0, 'vnet'), (str(legacy), 'best'))
        canonical = self.save('fold_0/best_model.pth')
        self.assertEqual(resolve_checkpoint(self.root, 0, 'vnet')[0], str(canonical))

    def test_phase_before_score(self):
        self.save('fold_0/pre_train/best_model.pth', best_dice=.99)
        selected = self.save('fold_0/self_train/best_model.pth', best_dice=.2)
        self.assertEqual(resolve_checkpoint(self.root, 0, include_pretrain=True)[0], str(selected))

    def test_latest_requires_opt_in(self):
        latest = self.save('fold_0/self_train/checkpoint_latest.pth')
        self.save('fold_0/best_model.pth')
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(self.root, 0)
        self.assertEqual(resolve_checkpoint(self.root, 0, allow_latest=True), (str(latest), 'latest'))
        self.assertEqual(resolve_checkpoint(self.root, 0, prefer='latest')[0], str(latest))

    def test_missing_and_pretrain(self):
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(self.root, 0)
        path = self.save('fold_0/pre_train/best_model.pth')
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(self.root, 0)
        self.assertEqual(resolve_checkpoint(self.root, 0, include_pretrain=True)[0], str(path))

    def test_manifest_provenance(self):
        path = self.save('fold_0/best_model.pth', iter_num=7, best_dice=.8,
                         extra_state={'checkpoint_kind': 'best', 'selected_iter': 7, 'selection_metric': 'val_mDice'})
        self.save('fold_0/checkpoint_latest.pth', iter_num=10, best_dice=.8)
        entries = json.loads((path.parent / 'checkpoint_manifest.json').read_text())['checkpoints']
        self.assertEqual(entries['best_model.pth']['kind'], 'best')
        self.assertEqual(entries['checkpoint_latest.pth']['kind'], 'latest')
        self.assertEqual(entries['best_model.pth']['selected_iter'], 7)

    def test_strict_load_and_legacy_requires_model(self):
        path = self.save('fold_0/best_model.pth', extra_state={'checkpoint_kind': 'best'})
        builder = lambda *a, **kw: torch.nn.Linear(2, 2)
        with self.assertRaises(ValueError):
            build_evaluation_model(path, builder)
        build_evaluation_model(path, builder, 'test')
        with self.assertRaises(RuntimeError):
            build_evaluation_model(path, lambda *a, **kw: torch.nn.Sequential(torch.nn.Linear(2, 2)), 'test')

    def test_config_architecture_and_ema(self):
        class Tiny(torch.nn.Module):
            def __init__(self, in_channels=1, num_classes=2, base_filters=2):
                super().__init__()
                self.layer = torch.nn.Linear(base_filters, num_classes)
        with patch.dict(MODELS._module_dict, {'audit_tiny': Tiny}):
            path = self.root / 'best_model.pth'
            source = Tiny(base_filters=3)
            ema = {k: torch.ones_like(v) for k, v in source.state_dict().items()}
            save_checkpoint(path, source, extra_state={'config': {'model': {'name': 'audit_tiny', 'base_filters': 3}}, 'ema_net': ema})
            model, _, name = build_evaluation_model(path, None, num_classes=2)
            self.assertEqual(name, 'audit_tiny')
            self.assertEqual(model.layer.in_features, 3)
            self.assertTrue(torch.equal(model.layer.weight, ema['layer.weight']))
            with self.assertRaises(ValueError):
                build_evaluation_model(path, None, 'vnet', 2)

    def test_class_config_mismatch(self):
        path = self.save('fold_0/best_model.pth', extra_state={
            'config': {'data': {'num_classes': 2}, 'model': {'name': 'unet3d'}}})
        with self.assertRaisesRegex(ValueError, 'checkpoint config has 2'):
            build_evaluation_model(path, None, num_classes=6)

    def test_benchmark_preflight_and_partial_summary(self):
        self.save('fold_0/best_model.pth', extra_state={'checkpoint_kind': 'best'})
        builder = lambda *a, **kw: torch.nn.Linear(2, 2)
        with self.assertRaises(FileNotFoundError):
            benchmark_5folds(builder, str(self.root), '', model_name='test')
        class Dataset:
            def __init__(self, **kwargs): pass
            def __len__(self): return 1
            def __getitem__(self, idx): return {'image': None, 'label': None}
        with patch('evaluation.benchmark.sliding_window_inference_3d', return_value=None), patch('evaluation.benchmark.cal_binary_dice', return_value=.5):
            summary = benchmark_5folds(builder, str(self.root), '', task='binary', model_name='test', dataset_cls=Dataset, on_missing='skip')
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['evaluated_folds'], [0])
        self.assertEqual(len(summary['skipped_folds']), 4)
        self.assertEqual(summary['fold_mean_dice'], .5)
        self.assertEqual(summary['fold_results']['fold_0']['checkpoint_kind'], 'best')
        self.assertTrue((self.root / 'benchmark_summary.json').exists())


if __name__ == '__main__':
    unittest.main()
