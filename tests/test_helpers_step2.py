"""Unit tests for Step 2: W&B table builder, system_info, and logger formatting."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from core.config_parser import Config
from utils.wandb_manager import WandbManager
from utils.system_info import record_resolved_config
from utils.logger import format_iteration_progress_string


class TestHelpersStep2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_wandb_table_builder_disabled_returns_none_without_sdk(self):
        cfg = Config({'experiment': {'fold': 0, 'exp_dir': str(self.root), 'name': 'test'}, 'logging': {'use_wandb': False}})
        mgr = WandbManager(cfg)
        mgr.enabled = False
        res = mgr.build_evaluation_table({'per_class': [{'class_id': 0}]}, ['BG'])
        self.assertIsNone(res)
        self.assertIsNone(mgr.run)

    def test_wandb_table_builder_with_mock_sdk(self):
        cfg = Config({
            'experiment': {'fold': 0, 'exp_dir': str(self.root), 'name': 'test'},
            'logging': {'use_wandb': False, 'wandb_project': 'test', 'wandb_mode': 'offline'}
        })
        mgr = WandbManager(cfg)
        mgr.enabled = True
        mock_table = MagicMock()
        with patch.dict('sys.modules', {'wandb': SimpleNamespace(Table=mock_table)}):
            details = {
                'per_class': [{
                    'class_id': 1, 'positive_cases': 5, 'negative_cases': 2,
                    'dice_case_mean': 0.8, 'dice_positive_mean': 0.85,
                    'hd95_mm_mean': 2.5, 'asd_mm_mean': 1.1,
                    'surface_valid_pairs': 5, 'negative_fp_case_rate': 0.0,
                    'negative_fp_ml_mean': None, 'precision_micro': 0.9,
                    'recall_micro': 0.85, 'tp_voxels': 100, 'fp_voxels': 10, 'fn_voxels': 15
                }]
            }
            res = mgr.build_evaluation_table(details, ['BG', 'Hem'])
            self.assertIsNotNone(res)
            self.assertTrue(mock_table.called)
            cols = mock_table.call_args[1]['columns']
            self.assertEqual(len(cols), 16)
            self.assertEqual(cols[0], "class_id")
            self.assertEqual(cols[1], "class_name")
            self.assertEqual(mock_table.call_args.kwargs["data"], [[
                1, "Hem", 5, 2, 0.8, 0.85, 2.5, 1.1, 5, 0.0, None, 0.9, 0.85, 100, 10, 15
            ]])
            self.assertIsNone(mgr.run)

    def test_record_resolved_config(self):
        cfg = Config({
            'experiment': {'exp_dir': str(self.root), 'name': 'test_exp', 'fold': 0},
            'data': {'intensity_units': 'HU', 'normalization': 'zscore_hu'},
            'method': {'name': 'mean_teacher'},
            'logging': {'api_key': 'private-test-key'}
        })
        layout = SimpleNamespace(to_dict=lambda: {'batch_size': 4})
        db_train = SimpleNamespace(split_fingerprint='train_hash', labeled_indices={0, 1})
        db_val = SimpleNamespace(split_fingerprint='val_hash')

        with patch('utils.system_info.collect_git_info', return_value={
                'source_revision': 'test-revision', 'source_dirty': False}) as git_info:
            resolved = record_resolved_config(
                snapshot_path=self.root,
                cfg=cfg,
                batch_layout=layout,
                validation_options={'patch_size': (32, 64, 64)},
                db_train=db_train,
                db_val=db_val,
                repo_root=self.root
            )

        git_info.assert_called_once_with(self.root)
        self.assertEqual(resolved['source_revision'], 'test-revision')
        self.assertIn('python', resolved)
        self.assertIn('torch', resolved)
        self.assertIn('split_fingerprints', resolved)
        self.assertEqual(resolved['actual_labeled_cases'], 2)
        self.assertTrue((self.root / 'resolved_config.json').exists())

        saved = json.loads((self.root / 'resolved_config.json').read_text())
        self.assertEqual(saved['experiment']['name'], 'test_exp')
        self.assertNotIn('private-test-key', (self.root / 'resolved_config.json').read_text())
        self.assertEqual(saved['resolved'], json.loads(json.dumps(cfg.to_dict()['resolved'])))

    def test_format_iteration_progress_string(self):
        msg = format_iteration_progress_string(
            iter_num=50,
            max_iters=100,
            sec_per_it=0.25,
            loss_dict={'total': 0.35, 'dice': 0.2},
            phase='train',
            vram_used=4.5
        )
        self.assertIn("[train] Iter 50/100 (50.0%)", msg)
        self.assertIn("0.25s/it", msg)
        self.assertIn("total: 0.3500", msg)
        self.assertIn("VRAM: 4.5GB", msg)
