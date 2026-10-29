"""
Unit tests for Epoch Boundary Checkpoints and Resume Policy.
Verifies specifications from docs/CHECKPOINT_RESUME_EPOCH_AUDIT_2026_10_29.md.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from methods.base_method import BaseMethod
from core.config_parser import Config
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.checkpoint_resolver import resolve_checkpoint


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class DummyTrainer(BaseMethod):
    def __init__(self, model, cfg, trainloader):
        self.model = model
        self.cfg = cfg
        self.device = torch.device('cpu')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.trainloader = trainloader
        self.writer = None
        self.wandb_manager = SimpleNamespace(
            enabled=False,
            save_artifacts=False,
            restore_artifact=MagicMock(),
            log_artifact=MagicMock(),
            log_metrics=MagicMock(),
            update_summary=MagicMock(),
            finish=MagicMock(),
            identity=lambda: None
        )
        self.is_ddp = False
        self.rank = 0
        self.world_size = 1
        self.fold = 0
        self.task = 'binary'
        self.num_classes = 2
        self.dataset_name = 'dummy'
        self.exp_name = 'dummy_exp'
        self.use_amp = False
        self.class_names = ['Background', 'Lesion']
        self.snapshot_path = str(cfg.experiment.exp_dir)

    def train_step(self, sampled_batch, iter_num, mim_mask=None):
        if isinstance(sampled_batch, dict):
            x = sampled_batch['image']
        elif isinstance(sampled_batch, (tuple, list)):
            x = sampled_batch[0]
        else:
            x = sampled_batch
        loss = (self.model(x) ** 2).mean()
        return loss, {'loss': float(loss.item())}

    def validate(self, return_visuals=False):
        return 0.75, [0.75]

    def validation_options(self):
        return {}


class TestEpochBoundaryCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snap_dir = Path(self.tmp.name)

        # 4 samples with batch_size=1 -> epoch_len = 4
        dataset = TensorDataset(torch.randn(4, 4), torch.zeros(4, dtype=torch.long))
        self.loader = DataLoader(dataset, batch_size=1, shuffle=False)

    def _build_cfg(self, resume=False, resume_policy='latest', max_iter=10):
        cfg_dict = {
            'experiment': {
                'name': 'dummy_exp',
                'task': 'binary',
                'num_classes': 2,
                'fold': 0,
                'seed': 1337,
                'deterministic': True,
                'exp_dir': str(self.snap_dir),
                'resume': resume,
                'resume_policy': resume_policy,
                'phase': 'train',
                'ddp': False
            },
            'data': {
                'name': 'dummy',
                'patch_size': [32, 32, 32],
                'batch_size': 1,
                'labeled_bs': 1,
            },
            'model': {
                'name': 'vnet',
                'in_channels': 1,
            },
            'method': {
                'name': 'mean_teacher',
                'max_iteration': max_iter,
                'base_lr': 0.01,
            },
            'logging': {
                'val_interval': 100,  # no val during 10 iters
                'save_interval': 2,    # latest saved every 2 iters
                'use_wandb': False,
                'save_wandb_model': False,
            }
        }
        return Config(cfg_dict)

    def test_epoch_boundary_checkpoint_saved_only_at_batch_end(self):
        """Epoch boundary checkpoint is saved only when batch_idx == epoch_len - 1."""
        cfg = self._build_cfg(max_iter=10)  # 10 iters with epoch_len=4 -> 2 full epochs (iter 4, 8) + 2 extra iters
        model = SimpleModel()
        trainer = DummyTrainer(model, cfg, self.loader)

        iter_num, best_dice = trainer.run_training_loop(
            phase='train',
            max_iterations=10,
            val_interval=100,
            save_interval=2,
            snapshot_dir=str(self.snap_dir)
        )

        self.assertEqual(iter_num, 10)
        latest_file = self.snap_dir / "checkpoint_latest.pth"
        epoch_file = self.snap_dir / "checkpoint_epoch_latest.pth"
        manifest_file = self.snap_dir / "checkpoint_manifest.json"

        self.assertTrue(latest_file.exists())
        self.assertTrue(epoch_file.exists())
        self.assertTrue(manifest_file.exists())

        # Check latest checkpoint state
        state_latest = torch.load(latest_file, weights_only=True)
        self.assertEqual(state_latest['iter_num'], 10)

        # Check epoch boundary checkpoint state: should be at iter 8 (completed epoch 2)
        state_epoch = torch.load(epoch_file, weights_only=True)
        self.assertEqual(state_epoch['iter_num'], 8)
        self.assertEqual(state_epoch['checkpoint_kind'], 'epoch_latest')
        self.assertEqual(state_epoch['completed_epochs'], 2)
        self.assertTrue(state_epoch['at_epoch_boundary'])

        # Check manifest
        manifest = json.loads(manifest_file.read_text())
        ckpts = manifest['checkpoints']
        self.assertIn('checkpoint_latest.pth', ckpts)
        self.assertIn('checkpoint_epoch_latest.pth', ckpts)
        self.assertEqual(ckpts['checkpoint_epoch_latest.pth']['kind'], 'epoch_latest')
        self.assertEqual(ckpts['checkpoint_epoch_latest.pth']['completed_epochs'], 2)
        self.assertTrue(ckpts['checkpoint_epoch_latest.pth']['at_epoch_boundary'])

    def test_resume_policy_epoch_boundary_restores_from_epoch_checkpoint(self):
        """When resume_policy='epoch_boundary', training restores from checkpoint_epoch_latest.pth."""
        # 1. Run first session to iter 10
        cfg1 = self._build_cfg(max_iter=10)
        trainer1 = DummyTrainer(SimpleModel(), cfg1, self.loader)
        trainer1.run_training_loop(snapshot_dir=str(self.snap_dir))

        # 2. Resume session with max_iter=12 and resume_policy='epoch_boundary'
        cfg2 = self._build_cfg(resume=True, resume_policy='epoch_boundary', max_iter=12)
        model2 = SimpleModel()
        trainer2 = DummyTrainer(model2, cfg2, self.loader)

        iter_num, _ = trainer2.run_training_loop(snapshot_dir=str(self.snap_dir))

        # Started from iter 8 (end of epoch 2), ran to 12 (4 iters = 1 full epoch)
        self.assertEqual(iter_num, 12)
        state_epoch2 = torch.load(self.snap_dir / "checkpoint_epoch_latest.pth", weights_only=True)
        self.assertEqual(state_epoch2['iter_num'], 12)
        self.assertEqual(state_epoch2['completed_epochs'], 3)

    def test_resume_policy_latest_restores_from_latest_checkpoint(self):
        """When resume_policy='latest', training restores from checkpoint_latest.pth."""
        # 1. Run first session to iter 10
        cfg1 = self._build_cfg(max_iter=10)
        trainer1 = DummyTrainer(SimpleModel(), cfg1, self.loader)
        trainer1.run_training_loop(snapshot_dir=str(self.snap_dir))

        # 2. Resume session with max_iter=12 and resume_policy='latest'
        cfg2 = self._build_cfg(resume=True, resume_policy='latest', max_iter=12)
        model2 = SimpleModel()
        trainer2 = DummyTrainer(model2, cfg2, self.loader)

        iter_num, _ = trainer2.run_training_loop(snapshot_dir=str(self.snap_dir))

        # Started from iter 10, ran to 12
        self.assertEqual(iter_num, 12)

    def test_resume_policy_epoch_boundary_fails_cleanly_if_missing(self):
        """If checkpoint_epoch_latest.pth is missing, resume_policy='epoch_boundary' raises FileNotFoundError."""
        # Create only checkpoint_latest.pth
        save_checkpoint(self.snap_dir / "checkpoint_latest.pth", SimpleModel(), iter_num=5)

        cfg = self._build_cfg(resume=True, resume_policy='epoch_boundary', max_iter=10)
        trainer = DummyTrainer(SimpleModel(), cfg, self.loader)
        with self.assertRaises(FileNotFoundError):
            trainer.run_training_loop(snapshot_dir=str(self.snap_dir))

    def test_fresh_run_guard_detects_epoch_checkpoint(self):
        """Fresh run guard rejects starting fresh in a directory with checkpoint_epoch_latest.pth."""
        save_checkpoint(self.snap_dir / "checkpoint_epoch_latest.pth", SimpleModel(), iter_num=4)

        cfg = self._build_cfg(resume=False, max_iter=10)
        trainer = DummyTrainer(SimpleModel(), cfg, self.loader)
        with self.assertRaises(FileExistsError):
            trainer.run_training_loop(snapshot_dir=str(self.snap_dir))

    def test_canonical_best_model_only_by_default(self):
        """By default, legacy VNet_best_model.pth is NOT created when saving best model."""
        cfg = self._build_cfg(max_iter=2)
        cfg.logging.val_interval = 2  # trigger validation at iter 2
        trainer = DummyTrainer(SimpleModel(), cfg, self.loader)
        trainer.run_training_loop(snapshot_dir=str(self.snap_dir))

        best_canonical = self.snap_dir / "best_model.pth"
        best_legacy = self.snap_dir / "VNet_best_model.pth"

        self.assertTrue(best_canonical.exists())
        self.assertFalse(best_legacy.exists())

    def test_checkpoint_resolver_prefers_epoch_latest(self):
        """resolve_checkpoint correctly finds checkpoint_epoch_latest.pth when prefer='epoch_latest'."""
        fold_dir = self.snap_dir / "fold_0"
        save_checkpoint(fold_dir / "checkpoint_latest.pth", SimpleModel(), iter_num=10)
        epoch_pth = save_checkpoint(fold_dir / "checkpoint_epoch_latest.pth", SimpleModel(), iter_num=8)

        path, kind = resolve_checkpoint(str(self.snap_dir), 0, prefer='epoch_latest')
        self.assertEqual(path, str(epoch_pth))
        self.assertEqual(kind, 'epoch_latest')


if __name__ == '__main__':
    unittest.main()
