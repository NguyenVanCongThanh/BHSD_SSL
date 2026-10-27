"""Regression tests for training ownership, leakage, and numerical correctness."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import h5py
import numpy as np
import torch
from core.config_parser import Config, load_config
from core.model_builder import model_spec_from_config, build_model_from_spec
from data.dataset_bhsd import BHSDDataset
from losses.compound import DiceCELoss
from losses.dice import MaskDiceLoss
from losses.deep_supervision import extract_multiscale_outputs
from methods.base_method import BaseMethod
from methods.cps import CPSMethod
from methods.lefed import LeFeDMethod
from methods.mean_teacher import MeanTeacherMethod
from methods.pick import PICKMethod
from methods.bcp import BCPMethod
from methods.ssl_utils import compute_masked_ce_loss
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.model_utils import extract_logits
from evaluation.sliding_window import sliding_window_inference_3d


class TinyTrainer(BaseMethod):
    def train_step(self, batch, iteration, mim_mask=None):
        loss = self.model(batch['image']).square().mean()
        return loss, {'total': float(loss.detach())}


class FeedbackNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, 1)

    def forward(self, image, feedback=None):
        out = self.conv(image)
        if feedback:
            out = out + feedback[0]
        return out, out * .9, [[out * .9], [out]], [out]


class LogicAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.batch = {'image': torch.randn(4, 1, 2, 4, 4),
                      'label': torch.randint(0, 2, (4, 2, 4, 4))}

    def trainer(self, cls=TinyTrainer, model=None):
        cfg = Config({'data': {'batch_size': 4, 'labeled_bs': 2, 'task': 'binary', 'num_classes': 2},
                      'model': {}, 'method': {'name': 'mean_teacher', 'max_iteration': 2,
                      'base_lr': .1, 'feedback_steps': 3, 'perturb_student': False},
                      'experiment': {'exp_dir': str(self.root), 'name': cls.__name__, 'amp': False},
                      'logging': {}})
        def loaders(trainer): trainer.trainloader = [self.batch]
        def logging(trainer):
            trainer.writer = None
            trainer.wandb_manager = SimpleNamespace(enabled=False, save_artifacts=False,
                update_summary=lambda *a, **k: None, log_metrics=lambda *a, **k: None,
                finish=lambda: None)
        with patch.object(BaseMethod, 'setup_dataloaders', loaders), \
             patch.object(BaseMethod, 'setup_logging', logging), \
             patch('torch.cuda.is_available', return_value=False):
            return cls(model if model is not None else torch.nn.Conv3d(1, 2, 1), cfg)

    def test_dice_respects_configured_class_weights(self):
        logits = torch.tensor([[[[[8.]]], [[[-8.]]]]])
        target = torch.zeros(1, 1, 1, 1, dtype=torch.long)
        background_only = MaskDiceLoss(2, class_weights=[1., 0.], include_background=True)
        foreground_only = MaskDiceLoss(2, class_weights=[0., 1.], include_background=True)
        self.assertLess(background_only(logits, target), foreground_only(logits, target))

    def test_bcp_zero_pseudo_weight_ignores_unlabeled_voxels(self):
        trainer = self.trainer(BCPMethod)
        trainer.u_weight = 0.
        altered = {key: value.clone() for key, value in self.batch.items()}
        altered['image'][2:] += 100
        np.random.seed(10)
        first, _ = trainer.train_step(self.batch, 0)
        np.random.seed(10)
        second, _ = trainer.train_step(altered, 0)
        self.assertTrue(torch.allclose(first, second))
        trainer.u_weight = 1.
        np.random.seed(10)
        third, _ = trainer.train_step(altered, 0)
        self.assertFalse(torch.allclose(first, third))

    def test_fp16_dice_reduction_stays_finite(self):
        logits = torch.zeros(1, 2, 32, 64, 64, dtype=torch.float16)
        target = torch.ones(1, 32, 64, 64, dtype=torch.long)
        loss = MaskDiceLoss(2)(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dtype, torch.float32)

    def test_seed_applies_before_model_initialization(self):
        spec = {'name': 'vnet', 'kwargs': {'in_channels': 1, 'num_classes': 2, 'n_filters': 2}}
        first = build_model_from_spec(spec, seed=42)
        second = build_model_from_spec(spec, seed=42)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_resume_missing_checkpoint_fails(self):
        trainer = self.trainer()
        trainer.cfg.experiment.resume = True
        with self.assertRaisesRegex(FileNotFoundError, 'Cannot resume'):
            trainer.run_training_loop(max_iterations=1)

    def test_zero_validation_score_is_valid_best(self):
        trainer = self.trainer()
        with patch.object(trainer, 'validate', return_value=(0., [0.])):
            trainer.run_training_loop(max_iterations=1)
        self.assertTrue((Path(trainer.snapshot_path) / 'best_model.pth').exists())

    def test_cloud_artifacts_are_fold_scoped(self):
        from unittest.mock import Mock
        trainer = self.trainer()
        trainer.wandb_manager.enabled = True
        trainer.wandb_manager.save_artifacts = True
        trainer.wandb_manager.log_artifact = Mock()
        with patch.object(trainer, 'validate', return_value=(.5, [.5])):
            trainer.run_training_loop(max_iterations=1)
        names = [call.kwargs['name'] for call in trainer.wandb_manager.log_artifact.call_args_list]
        self.assertTrue(names)
        self.assertTrue(all('_fold0_' in name for name in names))

    def test_pick_transfer_resets_optimizer_without_loading_pretrain_state(self):
        trainer = self.trainer(PICKMethod)
        loss = trainer.model(self.batch['image']).square().mean()
        trainer.apply_gradients(loss, 0)
        self.assertTrue(trainer.optimizer.state)
        path = self.root / 'pretrained.pth'
        save_checkpoint(path, trainer.model, trainer.optimizer)
        trainer.pretrain_checkpoint = str(path)
        trainer.load_pretrain_weights_for_selftrain()
        self.assertFalse(trainer.optimizer.state)

    def test_typeerror_inside_callback_is_not_retried(self):
        trainer = self.trainer()
        calls = []
        def broken(*args):
            calls.append(1)
            raise TypeError('internal defect')
        with self.assertRaisesRegex(TypeError, 'internal defect'):
            trainer.optimize_batch(self.batch, 0, step_fn=broken)
        self.assertEqual(len(calls), 1)

    def test_nonfinite_loss_does_not_mutate_weights(self):
        trainer = self.trainer()
        before = copy.deepcopy(trainer.model.state_dict())
        with self.assertRaises(FloatingPointError):
            trainer.apply_gradients(trainer.model(self.batch['image']).mean() * float('nan'), 0)
        for key, value in trainer.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_mean_teacher_updates_after_optimizer_and_copies_buffers(self):
        trainer = self.trainer(MeanTeacherMethod, torch.nn.Sequential(
            torch.nn.Conv3d(1, 2, 1), torch.nn.BatchNorm3d(2)))
        before = copy.deepcopy(trainer.ema_model.state_dict())
        loss, _ = trainer.train_step(self.batch, 0)
        for key, value in trainer.ema_model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        trainer.apply_gradients(loss, 0)
        self.assertFalse(torch.equal(trainer.ema_model[0].weight, before['0.weight']))
        self.assertTrue(torch.equal(trainer.ema_model[1].running_mean, trainer.model[1].running_mean))
        self.assertEqual(trainer.ema_model[1].num_batches_tracked, 1)

    def test_cps_both_networks_update_and_resume(self):
        trainer = self.trainer(CPSMethod)
        before = trainer.model2.weight.detach().clone()
        trainer.optimize_batch(self.batch, 0)
        self.assertFalse(torch.equal(trainer.model2.weight, before))
        self.assertEqual(trainer.update_learning_rate(1, 2), trainer.optimizer2.param_groups[0]['lr'])
        path = self.root / 'cps.pth'
        trainer.save_checkpoint(path, 1, .2)
        expected = copy.deepcopy(trainer.model2.state_dict())
        with torch.no_grad(): trainer.model2.weight.zero_()
        self.assertEqual(trainer.load_checkpoint(path), (1, .2))
        self.assertTrue(torch.equal(trainer.model2.weight, expected['weight']))
        self.assertTrue(trainer.optimizer2.state)
        legacy = self.root / 'legacy.pth'
        save_checkpoint(legacy, trainer.model)
        with self.assertRaisesRegex(ValueError, 'requires net2'):
            trainer.load_checkpoint(legacy)

    def test_lefed_each_feedback_backward_exactly_once(self):
        trainer = self.trainer(LeFeDMethod, FeedbackNet())
        with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step:
            loss, logs = trainer.optimize_batch(self.batch, 0)
            self.assertEqual(step.call_count, 3)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(logs['total'], 0)

    def test_final_validation_and_no_overwrite(self):
        trainer = self.trainer()
        with patch.object(trainer, 'validate', return_value=(.5, [.5])) as validate:
            trainer.run_training_loop(max_iterations=1, val_interval=5)
            self.assertEqual(validate.call_count, 1)
        best = Path(trainer.snapshot_path) / 'best_model.pth'
        state = torch.load(best, weights_only=True)
        self.assertEqual(state['checkpoint_kind'], 'best')
        with self.assertRaises(FileExistsError):
            trainer.run_training_loop(max_iterations=1, val_interval=5)

    def test_masked_ce_is_invariant_to_batch_duplication(self):
        logits = torch.randn(1, 2, 2, 4, 4)
        target = self.batch['label'][:1]
        mask = torch.ones(2, 4, 4)
        loss = DiceCELoss(2)
        self.assertTrue(torch.allclose(loss(logits, target, mask),
            loss(logits.repeat(3, 1, 1, 1, 1), target.repeat(3, 1, 1, 1), mask)))
        self.assertTrue(torch.allclose(compute_masked_ce_loss(logits, target, mask),
            compute_masked_ce_loss(logits.repeat(3, 1, 1, 1, 1), target.repeat(3, 1, 1, 1), mask)))

    def test_output_extraction_main_scale_and_pick(self):
        main = torch.randn(1, 2, 4, 4, 4)
        aux = torch.randn(1, 2, 2, 2, 2)
        self.assertIs(extract_logits([main, aux]), main)
        pick = ([main], main, aux)
        self.assertIs(extract_logits(pick), main)
        result, scales = extract_multiscale_outputs(pick)
        self.assertIs(result, main)
        self.assertEqual(scales, [])

    def test_inference_rejects_uncovered_stride(self):
        with self.assertRaisesRegex(ValueError, 'cover every voxel'):
            sliding_window_inference_3d(torch.nn.Identity(), np.zeros((8, 8, 8)), 2,
                                        patch_size=(2, 2, 2), stride_z=3)

    def test_inference_constant_predictions_with_padding_and_tta(self):
        class Constant(torch.nn.Module):
            def forward(self, image): return torch.cat([image * 0, image * 0 + 2], 1)
        for tta in (False, True):
            pred = sliding_window_inference_3d(Constant(), np.zeros((1, 3, 3), np.float32), 2,
                patch_size=(2, 2, 2), stride_z=1, stride_xy=1, use_tta=tta, device=torch.device('cpu'))
            self.assertEqual(pred.shape, (1, 3, 3))
            self.assertTrue((pred == 1).all())

    def test_fold_split_cannot_fallback_and_rejects_overlap(self):
        (self.root / 'val.txt').write_text('generic\n')
        with self.assertRaises(FileNotFoundError):
            BHSDDataset(str(self.root), split='val', fold=0)
        split = self.root / 'splits'
        split.mkdir()
        (split / 'fold0_train.txt').write_text('shared\n')
        (split / 'fold0_val.txt').write_text('shared\n')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            BHSDDataset(str(self.root), split='val', fold=0)

    def test_unlabeled_annotations_hidden_before_augmentation(self):
        (self.root / 'train.txt').write_text('case\n')
        (self.root / 'data').mkdir()
        with h5py.File(self.root / 'data/case.h5', 'w') as stream:
            stream['image'] = np.zeros((2, 4, 4), np.float32)
            stream['label'] = np.ones((2, 4, 4), np.uint8)
        seen = []
        def transform(sample):
            seen.append(sample['label'].copy())
            return sample
        dataset = BHSDDataset(str(self.root), transform=transform)
        dataset.labeled_indices = set()
        dataset[0]
        self.assertFalse(seen[0].any())
        dataset.labeled_indices = {0}
        dataset[0]
        self.assertTrue(seen[1].all())

    def test_task_override_and_missing_config(self):
        cfg = load_config(dataset_config='bhsd_multiclass', cli_args=['--task', 'binary'])
        self.assertEqual((cfg.data.task, cfg.experiment.task, cfg.data.num_classes), ('binary', 'binary', 2))
        self.assertEqual(len(cfg.data.class_names), 2)
        with self.assertRaises(FileNotFoundError):
            load_config(dataset_config='typo_dataset')
        with self.assertRaises(FileNotFoundError):
            load_config(config_path='/missing/experiment.yaml')

    def test_explicit_model_parameters_not_silently_ignored(self):
        import models
        spec = model_spec_from_config({'model': {'name': 'nnunet3d', 'num_stages': 5,
            'anisotropic_kernels': False, 'use_strided_conv': False}}, 2)
        self.assertFalse(spec['kwargs']['anisotropic_kernels'])
        self.assertEqual(spec['kwargs']['num_stages'], 5)
        with self.assertRaises(ValueError):
            model_spec_from_config({'model': {'name': 'unet3d'}, 'method': {'name': 'pick'}}, 2)

    def test_corrupt_sample_fallback_preserves_partitions(self):
        """DATA-02: Fallback on corrupt volumes must never cross labeled/unlabeled partition boundary."""
        split_dir = self.root / 'splits'
        split_dir.mkdir(exist_ok=True)
        # 4 cases: 0, 1 (labeled), 2, 3 (unlabeled)
        cases = ['case_l0', 'case_l1', 'case_u2', 'case_u3']
        (split_dir / 'fold0_train.txt').write_text('\n'.join(cases) + '\n')
        (split_dir / 'fold0_val.txt').write_text('case_val\n')
        data_dir = self.root / 'data'
        data_dir.mkdir(exist_ok=True)

        # case_l0: valid labeled
        with h5py.File(data_dir / 'case_l0.h5', 'w') as f:
            f['image'] = np.ones((2, 4, 4), dtype=np.float32)
            f['label'] = np.ones((2, 4, 4), dtype=np.uint8)

        # case_l1: corrupt labeled (0-slice)
        with h5py.File(data_dir / 'case_l1.h5', 'w') as f:
            f['image'] = np.zeros((0, 4, 4), dtype=np.float32)
            f['label'] = np.zeros((0, 4, 4), dtype=np.uint8)

        # case_u2: valid unlabeled
        with h5py.File(data_dir / 'case_u2.h5', 'w') as f:
            f['image'] = np.full((2, 4, 4), 2.0, dtype=np.float32)
            f['label'] = np.ones((2, 4, 4), dtype=np.uint8)  # raw file has ground truth

        # case_u3: corrupt unlabeled (0-slice)
        with h5py.File(data_dir / 'case_u3.h5', 'w') as f:
            f['image'] = np.zeros((0, 4, 4), dtype=np.float32)
            f['label'] = np.zeros((0, 4, 4), dtype=np.uint8)

        dataset = BHSDDataset(str(self.root), split='train', fold=0)
        dataset.labeled_indices = {0, 1}

        # 1. Access corrupt unlabeled sample at index 3 (boundary wrap-around):
        # Must fallback to case_u2 (index 2), NOT wrap to index 0 (case_l0)!
        sample_u = dataset[3]
        self.assertEqual(sample_u['case'], 'case_u2')
        # Annotations for unlabeled must remain completely zeroed out:
        self.assertFalse(sample_u['label'].any())

        # 2. Access corrupt labeled sample at index 1:
        # Must fallback to case_l0 (index 0), retaining ground truth:
        sample_l = dataset[1]
        self.assertEqual(sample_l['case'], 'case_l0')
        self.assertTrue(sample_l['label'].all())


if __name__ == '__main__':
    unittest.main()
