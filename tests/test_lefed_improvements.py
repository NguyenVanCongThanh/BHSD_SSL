"""Regression tests for LeFeD P0/P1/P2 architectural, loss, optimization, and I/O fixes."""

from datetime import timedelta
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from core.config_parser import Config, load_config
from core.model_builder import build_model_from_spec, model_spec_from_config
from methods.base_method import BaseMethod
from methods.lefed import LeFeDMethod
from models.backbones.lefed_net import LeFeD_Net
from models.backbones.vnet import VNet
from tests.test_epoch_boundary_checkpoint import DummyTrainer, SimpleModel


def _ddp_accumulate_worker(rank: int, root: str):
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo',
        init_method='file://' + str(Path(root) / 'rendezvous'),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        cfg = Config({
            'data': {'batch_size': 4, 'labeled_bs': 2, 'task': 'binary', 'num_classes': 2},
            'model': {},
            'method': {
                'name': 'lefed',
                'max_iteration': 2,
                'base_lr': 0.01,
                'feedback_steps': 2,
                'feedback_update_mode': 'accumulate',
                'consistency': 0.5,
                'consistency_type': 'hybrid',
                'ds_downsample_target': True,
                'ds_zero_lowest_scale': True,
                'batch_dice': True,
            },
            'experiment': {'exp_dir': root, 'name': 'lefed_ddp_accum', 'amp': False},
            'logging': {},
        })

        def loaders(trainer):
            trainer.trainloader = [0]

        def logging_setup(trainer):
            trainer.writer = None
            trainer.wandb_manager = SimpleNamespace(
                enabled=False,
                save_artifacts=False,
                log_metrics=lambda *a, **k: None,
                update_summary=lambda *a, **k: None,
            )

        torch.manual_seed(1337)
        net = LeFeD_Net(
            in_channels=1,
            num_classes=2,
            n_filters=4,
            normalization='instancenorm',
            has_dropout=False,
            anisotropic_kernels=True,
            activation='leaky_relu',
            skip_mode='concat',
        )
        with patch.object(BaseMethod, 'setup_dataloaders', loaders), \
             patch.object(BaseMethod, 'setup_logging', logging_setup), \
             patch('torch.cuda.is_available', return_value=False):
            trainer = LeFeDMethod(net, cfg)

        torch.manual_seed(100 + rank)
        batch = {
            'image': torch.randn(2, 1, 8, 32, 32),
            'label': torch.randint(0, 2, (2, 8, 32, 32)),
        }
        loss, logs = trainer.optimize_batch(batch, 0)
        assert torch.isfinite(loss), 'Accumulated DDP loss must be finite'
        assert trainer.successful_updates == 1, 'Expected exactly 1 update per iter in accumulate mode'
        assert logs['total'] > 0

        for param in trainer.model.parameters():
            expected = param.detach().clone()
            dist.broadcast(expected, src=0)
            assert torch.equal(param, expected), 'DDP parameters diverged after accumulated feedback update'
    finally:
        dist.destroy_process_group()


class LeFeDImprovementsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _build_trainer(self, model, method_overrides=None):
        method_cfg = {
            'name': 'lefed',
            'max_iteration': 10,
            'base_lr': 0.01,
            'feedback_steps': 2,
            'feedback_update_mode': 'accumulate',
            'consistency': 0.5,
            'consistency_rampup': 0.0,
            'consistency_type': 'hybrid',
            'consistency_dice_weight': 0.8,
            'consistency_mse_weight': 0.2,
            'ds_weights': [0.5, 0.25, 0.125, 0.0625, 0.03125],
            'ds_loss_weight': 0.5,
            'ds_downsample_target': True,
            'ds_zero_lowest_scale': True,
        }
        if method_overrides:
            method_cfg.update(method_overrides)

        cfg = Config({
            'data': {'batch_size': 4, 'labeled_bs': 2, 'task': 'binary', 'num_classes': 2},
            'model': {},
            'method': method_cfg,
            'experiment': {'exp_dir': str(self.root), 'name': 'lefed_test', 'amp': False},
            'logging': {},
        })

        def loaders(trainer):
            trainer.trainloader = [0]

        def logging_setup(trainer):
            trainer.writer = None
            trainer.wandb_manager = SimpleNamespace(
                enabled=False,
                save_artifacts=False,
                update_summary=lambda *a, **k: None,
                log_metrics=lambda *a, **k: None,
                finish=lambda: None,
            )

        with patch.object(BaseMethod, 'setup_dataloaders', loaders), \
             patch.object(BaseMethod, 'setup_logging', logging_setup), \
             patch('torch.cuda.is_available', return_value=False):
            return LeFeDMethod(model, cfg)

    def test_lefed_anisotropic_shapes_match_vnet5_and_sideconv_order_finest_to_coarsest(self):
        x = torch.randn(2, 1, 8, 32, 32)
        vnet5 = VNet(
            in_channels=1, num_classes=2, n_filters=4, num_stages=5,
            normalization='instancenorm', has_dropout=False, has_residual=False,
            anisotropic_kernels=True, activation='leaky_relu', skip_mode='concat',
        )
        lefed = LeFeD_Net(
            in_channels=1, num_classes=2, n_filters=4,
            normalization='instancenorm', has_dropout=False, has_residual=False,
            anisotropic_kernels=True, activation='leaky_relu', skip_mode='concat',
            eval_ensemble=True,
        )

        # Encoder shapes must match VNet5 stage-by-stage (preserving Z resolution)
        with torch.no_grad():
            vnet_feats = vnet5.encoder(x)
            lefed_feats = lefed.encoder(x)
        self.assertEqual([tuple(t.shape) for t in lefed_feats], [tuple(t.shape) for t in vnet_feats])

        # Training forward: deep_outs must be ordered finest -> coarsest
        lefed.train()
        out1, out2, masks, deep_outs = lefed(x, None)
        self.assertEqual(tuple(out1.shape), (2, 2, 8, 32, 32))
        self.assertEqual(tuple(out2.shape), (2, 2, 8, 32, 32))
        expected_ds_shapes = [
            (2, 2, 8, 32, 32),   # out1 (finest)
            (2, 2, 8, 16, 16),   # out2
            (2, 2, 8, 8, 8),     # out3
            (2, 2, 4, 4, 4),     # out4
            (2, 2, 2, 2, 2),     # out5 (coarsest)
        ]
        self.assertEqual([tuple(d.shape) for d in deep_outs], expected_ds_shapes)

        # Feedback pass en = 1e-3 * (stage_feat2 - stage_feat1) must add cleanly to encoder stages
        en = [1e-3 * (m2.detach() - m1.detach()) for m2, m1 in zip(masks[0], masks[1])]
        out1_fb, out2_fb, _, _ = lefed(x, en)
        self.assertEqual(tuple(out1_fb.shape), (2, 2, 8, 32, 32))
        self.assertEqual(tuple(out2_fb.shape), (2, 2, 8, 32, 32))

        # Eval ensemble mode returns tensor of averaged decoder logits
        lefed.eval()
        with torch.no_grad():
            ens_out = lefed(x)
            feats = lefed.encoder(x, None)
            o1, s1 = lefed.decoder1(feats)
            o2, _ = lefed.decoder2(feats, s1)
        self.assertTrue(torch.allclose(ens_out, 0.5 * (o1 + o2)))

    def test_lefed_method_foreground_dice_ds_weights_and_accumulate_mode(self):
        net = LeFeD_Net(
            in_channels=1, num_classes=2, n_filters=4,
            normalization='instancenorm', has_dropout=False,
            anisotropic_kernels=True, activation='leaky_relu', skip_mode='concat',
        )
        trainer = self._build_trainer(net)

        # 1. Foreground-only Dice even in binary task
        self.assertFalse(trainer.dice_loss.include_background)

        # 2. DS weights: finest gets largest, coarsest zeroed, total equals ds_loss_weight (0.5)
        w = trainer._resolve_ds_stage_weights(5)
        self.assertEqual(len(w), 5)
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[2])
        self.assertGreater(w[2], w[3])
        self.assertEqual(w[4], 0.0)
        self.assertAlmostEqual(sum(w), 0.5, places=6)

        # 3. Accumulate mode: 2 feedback forward passes, exactly 1 optimizer step
        batch = {
            'image': torch.randn(4, 1, 8, 32, 32),
            'label': torch.randint(0, 2, (4, 8, 32, 32)),
        }
        with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step_mock:
            loss, logs = trainer.optimize_batch(batch, 0)
            self.assertEqual(step_mock.call_count, 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(trainer.successful_updates, 1)
        self.assertGreater(logs['supervised'], 0.0)
        self.assertGreater(logs['deep_supervision'], 0.0)

    def test_lefed_experiment_yaml_validates_and_builds_model(self):
        cfg = load_config(config_path='configs/experiments/bhsd_lefed_vnet5_binary_fold0.yaml')
        spec = model_spec_from_config(cfg.to_dict(), num_classes=2)
        self.assertEqual(spec['name'], 'lefed_net')
        self.assertTrue(spec['kwargs']['anisotropic_kernels'])
        self.assertEqual(spec['kwargs']['skip_mode'], 'concat')
        self.assertEqual(spec['kwargs']['activation'], 'leaky_relu')
        self.assertTrue(spec['kwargs']['eval_ensemble'])
        model = build_model_from_spec(spec, seed=1337)
        self.assertIsInstance(model, LeFeD_Net)

    def test_lefed_ddp_accumulate_mode_keeps_ranks_synchronized(self):
        mp.spawn(_ddp_accumulate_worker, args=(str(self.root),), nprocs=2, join=True)

    def test_epoch_boundary_checkpoint_throttled_for_tiny_epochs_under_latest_policy(self):
        # Simulate low-label SSL where epoch_len = 2 and save_interval = 6 over 10 iters
        dataset = TensorDataset(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        cfg = Config({
            'experiment': {
                'name': 'tiny_epoch_exp',
                'task': 'binary',
                'num_classes': 2,
                'fold': 0,
                'seed': 1337,
                'deterministic': True,
                'exp_dir': str(self.root),
                'resume': False,
                'resume_policy': 'latest',
                'phase': 'train',
                'ddp': False,
            },
            'data': {'name': 'dummy', 'patch_size': [32, 32, 32], 'batch_size': 1, 'labeled_bs': 1},
            'model': {'name': 'vnet', 'in_channels': 1},
            'method': {'name': 'mean_teacher', 'max_iteration': 10, 'base_lr': 0.01},
            'logging': {'val_interval': 100, 'save_interval': 6, 'use_wandb': False, 'save_wandb_model': False},
        })
        trainer = DummyTrainer(SimpleModel(), cfg, loader)
        epoch_save_iters = []
        orig_save = trainer.save_checkpoint

        def spy_save(*args, **kwargs):
            path = kwargs.get('path') or (args[0] if args else '')
            if str(path).endswith('checkpoint_epoch_latest.pth'):
                epoch_save_iters.append(kwargs.get('iter_num', args[1] if len(args) > 1 else None))
            return orig_save(*args, **kwargs)

        with patch.object(trainer, 'save_checkpoint', side_effect=spy_save):
            trainer.run_training_loop(max_iterations=10, val_interval=100, save_interval=6, snapshot_dir=str(self.root))

        # Should save only once at iter 6 (not at 2, 4, 6, 8, 10)
        self.assertEqual(epoch_save_iters, [6])


if __name__ == '__main__':
    unittest.main()
