"""Contract tests for BaseMethod lifecycle, hook dispatch, and exception handling."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn

from core.config_parser import Config
from methods.base_method import BaseMethod


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, 1)

    def forward(self, x):
        return self.conv(x)


class MinimalTrainer(BaseMethod):
    def train_step(self, sampled_batch, iter_num, mim_mask=None):
        return torch.tensor(0.5, requires_grad=True), {"total": 0.5}


class TestBaseMethodContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = Config({
            'data': {'batch_size': 2, 'labeled_bs': 1, 'task': 'binary', 'num_classes': 2},
            'model': {},
            'method': {'name': 'supervised', 'max_iteration': 10},
            'experiment': {'exp_dir': str(self.root), 'name': 'contract_test', 'amp': False, 'resume': False},
            'logging': {'val_interval': 5, 'save_interval': 5}
        })

    def _build_trainer(self, cls=MinimalTrainer, model=None):
        def loaders(trainer):
            trainer.trainloader = [{'image': torch.randn(2, 1, 4, 8, 8), 'label': torch.zeros(2, 1, 4, 8, 8, dtype=torch.long)}]
            trainer.train_sampler = None
            trainer.batch_sampler = None
        def logging_setup(trainer):
            trainer.writer = MagicMock()
            trainer.wandb_manager = SimpleNamespace(
                enabled=False,
                save_artifacts=False,
                update_summary=MagicMock(),
                log_metrics=MagicMock(),
                finish=MagicMock(),
                identity=lambda: {'run_id': 'contract_run'}
            )
            trainer.use_wandb = False
        with patch.object(BaseMethod, 'setup_dataloaders', loaders), \
             patch.object(BaseMethod, 'setup_logging', logging_setup), \
             patch('torch.cuda.is_available', return_value=False):
            return cls(model or DummyModel(), self.cfg)

    def test_train_exception_handling_closes_writer_and_finishes_wandb(self):
        """Test Step 0: Ensure train() catches fatal errors, closes writer, finishes W&B with exit_code=1, and re-raises."""
        trainer = self._build_trainer()
        trainer.run_training_loop = MagicMock(side_effect=RuntimeError("Simulated GPU failure"))

        with self.assertRaises(RuntimeError) as ctx:
            trainer.train()

        self.assertIn("Simulated GPU failure", str(ctx.exception))
        self.assertTrue(trainer.writer.close.called)
        trainer.wandb_manager.finish.assert_called_once_with(exit_code=1)

    def test_existing_checkpoint_rejected_before_logging(self):
        folder = self.root / 'contract_test/fold_0'
        folder.mkdir(parents=True)
        (folder / 'checkpoint_latest.pth').touch()
        identity_path = folder / 'wandb_run.json'
        identity_path.write_text('{"id": "original-run"}')
        with patch.object(BaseMethod, 'setup_logging') as setup_logging:
            with self.assertRaises(FileExistsError):
                MinimalTrainer(DummyModel(), self.cfg)
        setup_logging.assert_not_called()
        self.assertEqual(identity_path.read_text(), '{"id": "original-run"}')

    def test_validation_and_checkpoint_dispatch_via_override(self):
        """Verify that subclass overrides of validate() and save_checkpoint() are respected by training loop."""
        class OverridingTrainer(MinimalTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.custom_validate_called = False
                self.custom_save_called = False

            def validate(self, return_visuals=False):
                self.custom_validate_called = True
                return 0.85, [0.85]

            def save_checkpoint(self, *args, **kwargs):
                self.custom_save_called = True
                return super().save_checkpoint(*args, **kwargs)

        trainer = self._build_trainer(cls=OverridingTrainer)
        best = trainer._handle_periodic_validation(
            snap_dir=str(self.root),
            phase='train',
            iter_num=5,
            global_step=5,
            best_dice=float('-inf'),
            best_model_name='best_model.pth',
            legacy_best_model_name=None,
            art_prefix='test',
            latest_pth=str(self.root / 'latest.pth'),
            should_vis=False
        )
        self.assertTrue(trainer.custom_validate_called, "Subclass validate() override must be dispatched")
        self.assertTrue(trainer.custom_save_called, "Subclass save_checkpoint() override must be dispatched")
        self.assertEqual(best, 0.85)

    def test_validation_state_is_published_before_checkpoint_hooks(self):
        for initial in ('missing', None, {'previous': 'details'}):
            with self.subTest(initial=initial):
                trainer = self._build_trainer()
                trainer.last_validation_dice = -1.0
                if initial != 'missing':
                    trainer.validation_details = initial
                expected = {'iteration': 5, 'global_step': 105, 'phase': 'train'}
                if isinstance(initial, dict):
                    expected.update(initial)
                original_save = trainer.save_checkpoint

                def save(**kwargs):
                    self.assertEqual(trainer.last_validation_dice, 0.8)
                    self.assertEqual(trainer.validation_details, expected)
                    return original_save(**kwargs)

                with patch.object(trainer, 'validate', return_value=(0.8, [0.8])), \
                     patch.object(trainer, 'save_checkpoint', side_effect=save) as save_hook:
                    trainer._handle_periodic_validation(
                        str(self.root), 'train', 5, 105, float('-inf'),
                        'best_model.pth', None, 'contract',
                        str(self.root / 'checkpoint_latest.pth'), False)
                self.assertEqual(save_hook.call_count, 2)
                for name in ('best_model.pth', 'checkpoint_latest.pth'):
                    checkpoint = torch.load(self.root / name, weights_only=True, map_location='cpu')
                    self.assertEqual(checkpoint['validation_metrics'], expected)

    def test_pick_phases_keep_session_open_and_offset_self_training(self):
        from methods.pick import PICKMethod
        self.cfg.data.batch_size = 4
        self.cfg.data.labeled_bs = 2
        self.cfg.method.name = 'pick'
        self.cfg.method.pre_max_iteration = 2
        self.cfg.method.self_max_iteration = 3
        trainer = self._build_trainer(cls=PICKMethod)
        trainer.cfg.experiment.resume = True
        calls = []

        def run_phase(**kwargs):
            trainer.writer.close.assert_not_called()
            trainer.wandb_manager.finish.assert_not_called()
            calls.append(kwargs)
            return kwargs['max_iterations'], 0.8

        with patch.object(trainer, 'run_training_loop', side_effect=run_phase), \
             patch.object(trainer, 'load_pretrain_weights_for_selftrain') as transfer:
            trainer.train()
        self.assertEqual([call['phase'] for call in calls], ['pre_train', 'self_train'])
        self.assertEqual([call['max_iterations'] for call in calls], [2, 3])
        self.assertTrue(all(not call['close_wandb_on_finish'] for call in calls))
        self.assertEqual(calls[1]['start_step_offset'], 2)
        self.assertTrue(calls[1]['allow_new_phase'])
        self.assertEqual(calls[0]['snapshot_dir'], trainer.pre_snapshot_path)
        self.assertEqual(calls[1]['snapshot_dir'], trainer.self_snapshot_path)
        transfer.assert_called_once()
        trainer.writer.close.assert_called_once()
        trainer.wandb_manager.finish.assert_called_once_with(exit_code=0)
