"""Validation rendering, metric semantics, and W&B lifecycle regression coverage."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import h5py
import numpy as np
import torch
from core.config_parser import Config
from data.dataset_bhsd import BHSDDataset
from evaluation.metrics import case_segmentation_stats, aggregate_segmentation_stats, surface_distances
from evaluation.sliding_window import evaluate_dataset_sliding_window
from losses.deep_supervision import resolve_ds_weights
from utils.visualizer import display_intensity, prepare_visual_record, comparison_panel, standardize_2d_slice
from utils.wandb_manager import WandbManager, redact_config


class ValidationTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = Config({'experiment': {'exp_dir': str(self.root), 'name': 'experiment', 'fold': 0},
            'logging': {'use_wandb': True, 'wandb_project': 'test', 'wandb_mode': 'offline'},
            'method': {'name': 'mean_teacher'}})
        self.run = SimpleNamespace(id='stable-run', summary={}, log=Mock(), finish=Mock(), define_metric=Mock())
        self.sdk = SimpleNamespace(run=None, init=Mock(return_value=self.run))
        self.env = patch.dict(os.environ, {'WANDB_MODE': 'offline'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def manager(self):
        with patch.dict(sys.modules, {'wandb': self.sdk}):
            return WandbManager(self.cfg)

    def test_hu_display_requires_units_and_normalization(self):
        image = np.array([-1., 0., 1.], np.float32)
        with self.assertRaisesRegex(ValueError, 'declared HU'):
            display_intensity(image, settings={'intensity_mode': 'hu'})
        result = display_intensity(image, {'normalization': 'zscore_hu', 'norm_params': {'mean': 40, 'std': 20}},
                                   {'intensity_mode': 'hu', 'window_center': 40, 'window_width': 80})
        np.testing.assert_allclose(result, [.25, .5, .75])
        hu = display_intensity(np.array([0., 40., 80.]), {'intensity_units': 'HU'}, {'intensity_mode': 'hu'})
        np.testing.assert_allclose(hu, [0, .5, 1])

    def test_negative_case_slice_is_prediction_independent(self):
        image = np.ones((5, 8, 16), np.float32)
        gt = np.zeros_like(image, dtype=np.uint8)
        pred = gt.copy();pred[0] = 1
        a = prepare_visual_record({'image': image, 'label': gt, 'pred': pred})
        pred[:] = 0;pred[4] = 1
        b = prepare_visual_record({'image': image, 'label': gt, 'pred': pred})
        self.assertEqual(a['z'], 2)
        self.assertEqual(a['z'], b['z'])
        self.assertEqual(a['image'].ndim, 2)

    def test_aspect_ratio_is_preserved(self):
        mask = np.ones((10, 20), np.uint8)
        result = standardize_2d_slice(mask, (100, 100), True)
        self.assertEqual(np.count_nonzero(result), 50 * 100)
        physical = standardize_2d_slice(mask, (100, 100), True, (2., 1.))
        self.assertTrue(physical.all())

    def test_panel_background_and_wrong_class_error(self):
        record = {'image': np.full((8, 8), .5), 'label': np.zeros((8, 8), np.uint8),
                  'pred': np.zeros((8, 8), np.uint8)}
        record['label'][2, 2] = 1;record['pred'][2, 2] = 2
        panel = comparison_panel(record, {0:'BG',1:'A',2:'B'})
        np.testing.assert_array_equal(panel[24,0], panel[24,8])
        np.testing.assert_array_equal(panel[26,26], [220,40,220])

    def test_metric_positive_dice_and_negative_fp_are_separate(self):
        positive = case_segmentation_stats(np.array([1,0]),np.array([1,1]),2)
        negative = case_segmentation_stats(np.array([1,0]),np.array([0,0]),2)
        row = aggregate_segmentation_stats([positive,negative])['per_class'][0]
        self.assertAlmostEqual(row['dice_positive_mean'], 2/3)
        self.assertAlmostEqual(row['dice_case_mean'], 1/3)
        self.assertEqual(row['negative_fp_case_rate'], 1.)
        self.assertEqual(row['positive_cases'], 1)
        self.assertIsNone(row['negative_fp_ml_mean'])

    def test_physical_volume_and_undefined_precision(self):
        row = case_segmentation_stats(np.ones((1,1,2),int),np.zeros((1,1,2),int),2,(5,.5,.5))
        self.assertAlmostEqual(row['classes'][0]['fp_ml'], .0025)
        empty = aggregate_segmentation_stats([case_segmentation_stats(np.zeros(2),np.zeros(2),2)])
        self.assertIsNone(empty['per_class'][0]['dice_positive_mean'])
        self.assertIsNone(empty['per_class'][0]['precision_micro'])
        json.dumps(empty, allow_nan=False)

    def test_surface_distance_uses_spacing_and_empty_policy(self):
        p = np.zeros((3,3,3),bool);g=p.copy()
        p[0,1,1]=True;g[1,1,1]=True
        result = surface_distances(p,g,(5.,1.,1.))
        self.assertEqual(result['hd95_mm'],5.)
        self.assertEqual(result['asd_mm'],5.)
        self.assertIsNone(surface_distances(p,np.zeros_like(p),(5.,1.,1.))['hd95_mm'])
        with self.assertRaisesRegex(ValueError, 'spacing'):
            case_segmentation_stats(p.astype(int),g.astype(int),2,compute_surface=True)

    def test_configured_deep_supervision_weights_are_respected(self):
        self.assertEqual(resolve_ds_weights(2,[1,3]),[.25,.75])
        with self.assertRaises(ValueError):resolve_ds_weights(3,[1,3])
        with self.assertRaises(ValueError):resolve_ds_weights(2,[0,0])

    def test_run_identity_persists_and_resume_uses_same_id(self):
        first = self.manager()
        identity = json.loads(first.identity_path.read_text())
        self.assertEqual(identity['id'], 'stable-run')
        first.finish()
        self.cfg.experiment.resume = True
        second = self.manager()
        self.assertEqual(self.sdk.init.call_args.kwargs['id'], 'stable-run')
        second.finish()

    def test_online_resume_is_strict(self):
        first=self.manager();first.finish()
        self.cfg.experiment.resume=True
        with patch.dict(os.environ, {'WANDB_MODE':'online'}):self.manager()
        self.assertEqual(self.sdk.init.call_args.kwargs['resume'],'must')

    def test_resume_identity_can_come_from_checkpoint(self):
        import torch
        folder=self.root/'experiment/fold_0';folder.mkdir(parents=True)
        torch.save({'wandb_run':{'id':'from-checkpoint','project':'test','entity':None}},folder/'checkpoint_latest.pth')
        self.cfg.experiment.resume=True
        self.manager()
        self.assertEqual(self.sdk.init.call_args.kwargs['id'],'from-checkpoint')

    def test_fresh_run_preserves_identity_when_checkpoints_exist(self):
        for directory in ('', 'pre_train', 'self_train'):
            with self.subTest(directory=directory):
                self.cfg.method.name = 'pick'
                folder = self.root / 'experiment/fold_0'
                checkpoint = folder / directory / 'checkpoint_latest.pth'
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
                identity_path = folder / 'wandb_run.json'
                original = json.dumps({'id': 'original-run', 'project': 'test'})
                identity_path.write_text(original)
                with self.assertRaises(FileExistsError):
                    self.manager()
                self.sdk.init.assert_not_called()
                self.assertEqual(identity_path.read_text(), original)
                checkpoint.unlink()

    def test_fresh_self_train_can_use_existing_pretraining(self):
        self.cfg.method.name = 'pick'
        self.cfg.experiment.phase = 'self_train'
        folder = self.root / 'experiment/fold_0/pre_train'
        folder.mkdir(parents=True)
        (folder / 'checkpoint_latest.pth').touch()
        self.assertTrue(self.manager().enabled)

    def test_resume_rejects_identity_conflicting_with_checkpoint(self):
        first = self.manager()
        first.finish()
        self.sdk.init.reset_mock()
        torch.save({'wandb_run': {'id': 'different-run', 'project': 'test', 'entity': None}},
                   first.identity_path.parent / 'checkpoint_latest.pth')
        self.cfg.experiment.resume = True
        with self.assertRaisesRegex(ValueError, 'conflicts with checkpoint'):
            self.manager()
        self.sdk.init.assert_not_called()

    def test_resume_without_identity_fails(self):
        self.cfg.experiment.resume=True
        with self.assertRaisesRegex(ValueError,'persisted run identity'):self.manager()

    def test_log_uses_explicit_axis_not_rewriting_sdk_step(self):
        manager=self.manager()
        manager.log_metrics({'train/loss':1.},step=10)
        manager.log_metrics({'val/mDice':.5},step=10)
        self.assertEqual(self.run.log.call_count,2)
        self.assertEqual(self.run.log.call_args.args[0]['global_step'],10)
        self.assertNotIn('step',self.run.log.call_args.kwargs)
        self.assertTrue(self.run.log.call_args.kwargs['commit'])

    def test_failure_status_and_config_redaction(self):
        self.cfg.logging.api_key='test-secret'
        manager=self.manager()
        self.assertEqual(self.sdk.init.call_args.kwargs['config']['logging']['api_key'],'[REDACTED]')
        manager.finish(exit_code=1)
        self.run.finish.assert_called_once_with(exit_code=1)

    def test_log_failure_warns_and_required_mode_raises(self):
        manager=self.manager();self.run.log.side_effect=RuntimeError('storage unavailable')
        with self.assertLogs(level='WARNING'):manager.log_metrics({'x':1},1)
        self.cfg.logging.wandb_required=True
        with self.assertRaises(RuntimeError):manager.log_metrics({'x':1},2)

    def test_dataset_preserves_metadata_only_for_validation(self):
        (self.root/'data').mkdir();(self.root/'val.txt').write_text('case\n')
        with h5py.File(self.root/'data/case.h5','w') as stream:
            stream['image']=np.zeros((2,3,4),np.float32);stream['label']=np.zeros((2,3,4),np.uint8)
            stream.attrs['target_spacing']='(5.0, 0.5, 0.5)'
            stream.attrs['norm_params']="{'mean': 40.0, 'std': 20.0}"
        dataset=BHSDDataset(str(self.root),split='val',normalization='zscore_hu')
        sample=dataset[0]
        self.assertEqual(sample['metadata']['norm_params']['std'],20)
        self.assertEqual(sample['metadata']['spacing'],(5.,.5,.5))

    def test_validation_details_and_fixed_cohort(self):
        class Dataset:
            image_list=['case_b','case_a']
            def __len__(self):return 2
            def __getitem__(self,index):return {'image':np.zeros((2,4,4),np.float32),
                'label':np.zeros((2,4,4),np.uint8),'case':self.image_list[index]}
        class Net(torch.nn.Module):
            def forward(self,x):return torch.cat([x+1,x],1)
        details={}
        mean,classes,images=evaluate_dataset_sliding_window(Net(),Dataset(),num_classes=2,is_binary=True,
            patch_size=(2,4,4),stride_xy=2,stride_z=1,device=torch.device('cpu'),return_visual_samples=1,
            details_sink=details,visual_settings={'target_shape':[16,16]})
        self.assertEqual(details['visual_cohort'],['case_a'])
        self.assertEqual(images[0]['case'],'case_a')
        self.assertEqual(images[0]['image'].shape,(16,16))
        self.assertEqual(details['per_class'][0]['negative_cases'],2)

    def test_wandb_metric_definitions_include_summary_and_hidden(self):
        def validate_glob(name, **kwargs):
            if '*' in name.rstrip('*'):
                raise ValueError('Only suffix globs are supported')
        self.run.define_metric.side_effect = validate_glob
        manager = self.manager()
        self.assertTrue(manager.enabled)
        counts = {f'val/CustomClass_{suffix}': 1 for suffix in (
            'tp_voxels', 'fp_voxels', 'fn_voxels', 'positive_cases',
            'negative_cases', 'surface_valid_pairs')}
        manager.log_metrics({'val/mDice': .5, **counts}, step=10)
        manager.log_metrics(counts, step=20)
        calls = self.run.define_metric.call_args_list
        defined = {call.args[0]: call.kwargs for call in calls if call.args}
        self.assertEqual(defined.get('val/mDice', {}).get('summary'), 'max')
        self.assertEqual(defined.get('val/macro_positive_dice', {}).get('summary'), 'max')
        self.assertEqual(defined.get('train/total', {}).get('summary'), 'min')
        for key in counts:
            self.assertTrue(defined[key]['hidden'])
            self.assertEqual(sum(call.args[0] == key for call in calls), 1)
        self.assertEqual(self.run.log.call_count, 2)

    @unittest.skipUnless(importlib.util.find_spec('wandb_workspaces') and importlib.util.find_spec('wandb'),
                         'wandb-workspaces not installed in test environment')
    def test_setup_wandb_dashboard_builders(self):
        from tools.setup_wandb_dashboard import (
            build_overview_workspace,
            build_evaluation_workspace,
            build_diagnostics_workspace,
            inspect_workspace
        )
        classes = ['Epidural', 'Subdural']
        overview = build_overview_workspace('entity', 'proj', classes)
        eval_ws = build_evaluation_workspace('entity', 'proj', classes)
        diag = build_diagnostics_workspace('entity', 'proj', classes)

        info_o = inspect_workspace(overview)
        info_e = inspect_workspace(eval_ws)
        info_d = inspect_workspace(diag)

        self.assertEqual(info_o['total_panels'], 7)
        self.assertFalse(info_o['auto_generate_panels'])
        self.assertEqual(info_e['total_panels'], 7)
        self.assertEqual(info_d['total_panels'], 6)


@unittest.skipUnless(os.environ.get('BHSD_TEST_WANDB_OFFLINE')=='1' and importlib.util.find_spec('wandb'),
                     'Enable explicitly with an installed SDK for real offline integration')
class WandbOfflineIntegration(unittest.TestCase):
    def test_real_sdk_logs_media_and_failed_exit_without_cloud(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
                'WANDB_MODE':'offline','WANDB_DIR':root,'WANDB_SILENT':'true'}):
            cfg=Config({'experiment':{'name':'offline-test','exp_dir':root},'method':{'name':'mean_teacher'},
                'logging':{'use_wandb':True,'wandb_mode':'offline','wandb_project':'local-validation','wandb_required':True}})
            import wandb
            self.addCleanup(wandb.teardown, exit_code=0)
            manager=WandbManager(cfg)
            self.assertEqual(manager.run.settings.mode,'offline')
            from utils.visualizer import create_wandb_visual_payload
            volume=np.zeros((2,8,8),np.float32)
            payload=create_wandb_visual_payload([{'case':'synthetic','image':volume,
                'label':volume.astype(np.uint8),'pred':volume.astype(np.uint8),'dice':[1.]}],
                {0:'Background',1:'Foreground'}, {'target_shape':[32,32],'interactive':True})
            manager.log_metrics({'train/loss':.5},step=10)
            manager.log_metrics({'val/mDice':1., 'val/Foreground_tp_voxels': 0,
                                 'val/Foreground_positive_cases': 0, **payload},step=10)
            self.assertEqual(manager.log_failures, 0)
            manager.update_summary({'best_mDice':1.})
            manager.finish(exit_code=1)
            self.assertTrue(list(Path(root).rglob('*.wandb')))
            self.assertTrue(list(Path(root).rglob('*.png')))
