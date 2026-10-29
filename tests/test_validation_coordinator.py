"""Unit tests for training/validation_coordinator.py."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from training.validation_coordinator import (
    ValidationCoordinator,
    ValidationRequest,
    ValidationRuntime,
    ValidationCallbacks,
)


class TestValidationCoordinator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _build_request_and_runtime(self, best_dice=float('-inf'), is_ddp=False, rank=0, legacy_best_model_name=None):
        req = ValidationRequest(
            snap_dir=str(self.root),
            phase='train',
            iter_num=500,
            global_step=500,
            best_dice=best_dice,
            best_model_name='best_model.pth',
            legacy_best_model_name=legacy_best_model_name,
            art_prefix='test_exp',
            latest_pth=str(self.root / 'checkpoint_latest.pth'),
            should_vis=False,
        )
        rt = ValidationRuntime(
            device=torch.device('cpu'),
            is_ddp=is_ddp,
            rank=rank,
            world_size=1,
            class_names=['Background', 'Hem'],
        )
        return req, rt

    def test_validation_successful_and_updates_best(self):
        req, rt = self._build_request_and_runtime(best_dice=0.5)
        details = {'macro_positive_dice': 0.8}
        saved_checkpoints = []

        callbacks = ValidationCallbacks(
            validate=MagicMock(return_value=(0.85, np.array([0.85]))),
            save_checkpoint=lambda **kw: saved_checkpoints.append(kw),
            cloud_latest_due=lambda it: True,
            build_visual_payload=lambda r: {},
            get_validation_details=lambda: details,
            resume_training_mode=MagicMock(),
            publish_validation_state=MagicMock(),
        )
        wandb_mgr = SimpleNamespace(
            enabled=True,
            update_summary=MagicMock(),
            log_metrics=MagicMock(),
            build_evaluation_table=MagicMock(return_value=None),
        )
        writer = MagicMock()

        coordinator = ValidationCoordinator()
        outcome = coordinator.handle(
            request=req,
            runtime=rt,
            callbacks=callbacks,
            writer=writer,
            wandb_manager=wandb_mgr,
        )

        self.assertEqual(outcome.best_dice, 0.85)
        self.assertEqual(outcome.last_validation_dice, 0.85)
        self.assertEqual(details['iteration'], 500)
        self.assertEqual(details['phase'], 'train')
        self.assertTrue(callbacks.resume_training_mode.called)
        wandb_mgr.log_metrics.assert_called_once()

        # Ensure only canonical best and latest checkpoints were saved by default
        paths = [kw['path'] for kw in saved_checkpoints]
        self.assertEqual(paths, [str(self.root / name) for name in
                                 ('best_model.pth', 'checkpoint_latest.pth')])

    def test_validation_saves_legacy_when_explicitly_configured(self):
        req, rt = self._build_request_and_runtime(best_dice=0.5, legacy_best_model_name='VNet_best_model.pth')
        saved_checkpoints = []
        callbacks = ValidationCallbacks(
            validate=MagicMock(return_value=(0.85, np.array([0.85]))),
            save_checkpoint=lambda **kw: saved_checkpoints.append(kw),
            cloud_latest_due=lambda it: False,
            build_visual_payload=lambda r: {},
            get_validation_details=lambda: {},
            resume_training_mode=MagicMock(),
            publish_validation_state=MagicMock(),
        )
        wandb_mgr = SimpleNamespace(
            enabled=False,
            update_summary=MagicMock(),
            log_metrics=MagicMock(),
            build_evaluation_table=MagicMock(return_value=None),
        )
        coordinator = ValidationCoordinator()
        coordinator.handle(
            request=req,
            runtime=rt,
            callbacks=callbacks,
            writer=MagicMock(),
            wandb_manager=wandb_mgr,
        )
        paths = [kw['path'] for kw in saved_checkpoints]
        self.assertEqual(paths, [str(self.root / name) for name in
                                 ('best_model.pth', 'VNet_best_model.pth', 'checkpoint_latest.pth')])

    def test_validation_dice_zero_is_valid_best_if_initial_is_neg_inf(self):
        req, rt = self._build_request_and_runtime(best_dice=float('-inf'))
        saved_checkpoints = []
        callbacks = ValidationCallbacks(
            validate=MagicMock(return_value=(0.0, np.array([0.0]))),
            save_checkpoint=lambda **kw: saved_checkpoints.append(kw),
            cloud_latest_due=lambda it: False,
            build_visual_payload=lambda r: {},
            get_validation_details=lambda: {},
            resume_training_mode=MagicMock(),
            publish_validation_state=MagicMock(),
        )
        wandb_mgr = SimpleNamespace(
            enabled=False,
            update_summary=MagicMock(),
            log_metrics=MagicMock(),
            build_evaluation_table=MagicMock(return_value=None),
        )
        coordinator = ValidationCoordinator()
        outcome = coordinator.handle(
            request=req,
            runtime=rt,
            callbacks=callbacks,
            writer=None,
            wandb_manager=wandb_mgr,
        )
        self.assertEqual(outcome.best_dice, 0.0)
        paths = [kw['path'] for kw in saved_checkpoints]
        self.assertIn(str(self.root / 'best_model.pth'), paths)

    def test_validation_nonfinite_raises_floating_point_error(self):
        req, rt = self._build_request_and_runtime(best_dice=0.5)
        callbacks = ValidationCallbacks(
            validate=MagicMock(return_value=(float('nan'), np.array([0.5]))),
            save_checkpoint=MagicMock(),
            cloud_latest_due=lambda it: False,
            build_visual_payload=lambda r: {},
            get_validation_details=lambda: {},
            resume_training_mode=MagicMock(),
            publish_validation_state=MagicMock(),
        )
        coordinator = ValidationCoordinator()
        with self.assertRaises(FloatingPointError):
            coordinator.handle(
                request=req,
                runtime=rt,
                callbacks=callbacks,
                writer=None,
                wandb_manager=SimpleNamespace(enabled=False),
            )

    def test_tie_logs_metrics_visuals_and_table_once_without_replacing_best(self):
        req, rt = self._build_request_and_runtime(best_dice=0.85)
        req.should_vis = True
        req.global_step = 1500
        details = {'per_class': [{'class_id': 1, 'dice_case_mean': 0.85}]}
        table = object()
        visuals = [{'case': 'fixed-case'}]
        callbacks = ValidationCallbacks(
            validate=MagicMock(return_value=(0.85, [0.85], visuals)),
            save_checkpoint=MagicMock(), cloud_latest_due=lambda it: False,
            build_visual_payload=MagicMock(return_value={'val/predictions': 'image'}),
            get_validation_details=lambda: details,
            resume_training_mode=MagicMock(), publish_validation_state=MagicMock(),
        )
        manager = SimpleNamespace(enabled=True, update_summary=MagicMock(),
                                  log_metrics=MagicMock(),
                                  build_evaluation_table=MagicMock(return_value=table))
        outcome = ValidationCoordinator().handle(request=req, runtime=rt, callbacks=callbacks,
                                               writer=None, wandb_manager=manager)
        self.assertEqual(outcome.best_dice, 0.85)
        callbacks.validate.assert_called_once_with(return_visuals=True)
        callbacks.save_checkpoint.assert_called_once()
        self.assertEqual(callbacks.save_checkpoint.call_args.kwargs['path'], req.latest_pth)
        manager.update_summary.assert_not_called()
        manager.log_metrics.assert_called_once()
        payload = manager.log_metrics.call_args.args[0]
        self.assertEqual(manager.log_metrics.call_args.kwargs['step'], 1500)
        self.assertEqual(payload['val/mDice'], 0.85)
        self.assertEqual(payload['val/predictions'], 'image')
        self.assertIs(payload['val/evaluation_table'], table)
        callbacks.publish_validation_state.assert_called_once_with(0.85, details)
