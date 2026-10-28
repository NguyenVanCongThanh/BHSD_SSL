"""
Unit tests for Phase 1: Config & Contract Bugfixes
Verifies:
- OUT-01: Multiscale output extraction with dictionary aux, lists, fallback, and validation.
- MOD-02: Model spec keyword argument validation against constructor signatures.
- CFG-01: Config precedence (selectors resolved, dataset does not overwrite method defaults).
- CFG-02: Dynamic task & class synchronization without BHSD imports; schema & range validation.
"""

import unittest
from pathlib import Path
import torch

from core.config_parser import load_config, validate_config
from core.model_builder import model_spec_from_config, build_model_from_spec
from losses.deep_supervision import extract_multiscale_outputs
from models.outputs import ModelOutput, wrap_model_output
from utils.model_utils import extract_logits
import models


class TestPhase1Contracts(unittest.TestCase):

    def test_out_01_multiscale_extraction_dict_formats(self):
        main = torch.randn(2, 6, 16, 32, 32)
        aux1 = torch.randn(2, 6, 8, 16, 16)
        aux2 = torch.randn(2, 6, 4, 8, 8)

        # 1. Standard {'logits': main, 'aux': [aux1, aux2]} (OUT-01 probe)
        out_dict = {'logits': main, 'aux': [aux1, aux2]}
        res_main, res_aux = extract_multiscale_outputs(out_dict)
        self.assertIs(res_main, main)
        self.assertEqual(len(res_aux), 2)
        self.assertIs(res_aux[0], aux1)
        self.assertIs(res_aux[1], aux2)

        # 2. Key 'seg' with single aux tensor
        out_seg = {'seg': main, 'aux': aux1}
        res_main2, res_aux2 = extract_multiscale_outputs(out_seg)
        self.assertIs(res_main2, main)
        self.assertEqual(len(res_aux2), 1)
        self.assertIs(res_aux2[0], aux1)

        # 3. Key 'out' without aux
        out_only = {'out': main}
        res_main3, res_aux3 = extract_multiscale_outputs(out_only)
        self.assertIs(res_main3, main)
        self.assertEqual(len(res_aux3), 0)

        # 4. Fallback when keys are arbitrary tensor names
        fallback_dict = {'pred_level0': main, 'pred_level1': aux1}
        res_main4, res_aux4 = extract_multiscale_outputs(fallback_dict)
        self.assertIs(res_main4, main)
        self.assertEqual(len(res_aux4), 1)
        self.assertIs(res_aux4[0], aux1)

    def test_out_01_multiscale_extraction_sequences_and_pick(self):
        main = torch.randn(2, 6, 16, 32, 32)
        aux = torch.randn(2, 6, 8, 16, 16)

        # Multi-scale list
        res_main, res_aux = extract_multiscale_outputs([main, aux])
        self.assertIs(res_main, main)
        self.assertEqual(res_aux, [aux])

        # PICK tuple: (features, seg, feat_de)
        pick_tuple = ([torch.randn(2, 16)], main, torch.randn(2, 32))
        res_pick, res_pick_aux = extract_multiscale_outputs(pick_tuple)
        self.assertIs(res_pick, main)
        self.assertEqual(res_pick_aux, [])

        # Pure tensor
        res_t, res_t_aux = extract_multiscale_outputs(main)
        self.assertIs(res_t, main)
        self.assertEqual(res_t_aux, [])

    def test_out_01_multiscale_extraction_validation_errors(self):
        # Empty dict
        with self.assertRaises(ValueError):
            extract_multiscale_outputs({})

        # Empty sequence
        with self.assertRaises(ValueError):
            extract_multiscale_outputs([])

        # Invalid main type
        with self.assertRaises(TypeError):
            extract_multiscale_outputs({'logits': "invalid_tensor"})

        # Invalid aux element
        with self.assertRaises(TypeError):
            extract_multiscale_outputs({'logits': torch.randn(1, 2, 4, 4, 4), 'aux': ["invalid"]})

        # Unsupported type
        with self.assertRaises(TypeError):
            extract_multiscale_outputs(12345)

    def test_mod_02_model_spec_validation(self):
        # Valid parameters pass cleanly
        spec = model_spec_from_config({
            'model': {
                'name': 'vnet',
                'normalization': 'batchnorm',
                'has_dropout': True,
                'n_filters': 16
            }
        }, num_classes=6)
        self.assertEqual(spec['name'], 'vnet')
        self.assertEqual(spec['kwargs']['normalization'], 'batchnorm')
        self.assertEqual(spec['kwargs']['num_classes'], 6)

        # Typo parameter in non-legacy mode raises ValueError
        with self.assertRaises(ValueError) as ctx:
            model_spec_from_config({
                'model': {
                    'name': 'vnet',
                    'has_dropuot': True  # Typo!
                }
            }, num_classes=6, legacy=False)
        self.assertIn('has_dropuot', str(ctx.exception))

        # Typo parameter in legacy mode is silently filtered (backward compatibility)
        spec_legacy = model_spec_from_config({
            'model': {
                'name': 'vnet',
                'has_dropuot': True,
                'normalization': 'batchnorm'
            }
        }, num_classes=6, legacy=True)
        self.assertNotIn('has_dropuot', spec_legacy['kwargs'])
        self.assertEqual(spec_legacy['kwargs']['normalization'], 'batchnorm')

        # Capability enforcement
        with self.assertRaises(ValueError):
            model_spec_from_config({'model': {'name': 'unet3d'}, 'method': {'name': 'pick'}}, num_classes=6)
        with self.assertRaises(ValueError):
            model_spec_from_config({'model': {'name': 'vnet'}, 'method': {'name': 'lefed'}}, num_classes=6)

    def test_cfg_01_precedence_probe_matrix(self):
        # Row 1: method_config='mean_teacher'
        c1 = load_config(method_config='mean_teacher')
        self.assertEqual(c1.method.name, 'mean_teacher')
        self.assertEqual(list(c1.data.patch_size), [32, 192, 192])
        self.assertEqual(c1.data.labelnum, 154)

        # Row 2: method_config='mean_teacher' + dataset_config='bhsd_multiclass'
        # Dataset must NOT overwrite method-specific patch_size or labelnum
        c2 = load_config(method_config='mean_teacher', dataset_config='bhsd_multiclass')
        self.assertEqual(c2.method.name, 'mean_teacher')
        self.assertEqual(list(c2.data.patch_size), [32, 192, 192])
        self.assertEqual(c2.data.labelnum, 154)

        # Row 3: config_path='configs/methods/mean_teacher.yaml' + dataset_config='bhsd_multiclass'
        c3 = load_config(config_path='configs/methods/mean_teacher.yaml', dataset_config='bhsd_multiclass')
        self.assertEqual(c3.method.name, 'mean_teacher')
        self.assertEqual(list(c3.data.patch_size), [32, 192, 192])
        self.assertEqual(c3.data.labelnum, 154)

        # Row 4: Conflicting selectors fail early
        with self.assertRaises(ValueError) as ctx:
            load_config(config_path='configs/methods/mean_teacher.yaml', method_config='cps')
        self.assertIn('Conflicting method specification', str(ctx.exception))

    def test_cfg_01_experiment_presets_load_cleanly(self):
        presets = [
            'configs/experiments/bhsd_pick_fold0.yaml',
            'configs/experiments/bhsd_mt_nnunet_fold0.yaml',
            'configs/experiments/bhsd_cps_fold0.yaml',
            'configs/experiments/bhsd_supervised_fold0.yaml'
        ]
        for p in presets:
            cfg = load_config(config_path=p)
            self.assertTrue(hasattr(cfg, 'method'))
            self.assertTrue(hasattr(cfg, 'model'))
            self.assertTrue(hasattr(cfg, 'data'))
            self.assertTrue(hasattr(cfg, 'experiment'))
            self.assertEqual(cfg.experiment.fold, 0)
            self.assertEqual(len(cfg.data.patch_size), 3)

    def test_cfg_02_validation_invariants(self):
        # Invalid labeled_bs exceeding batch_size
        with self.assertRaises(ValueError):
            validate_config({'data': {'batch_size': 2, 'labeled_bs': 4}})

        # Negative fold
        with self.assertRaises(ValueError):
            validate_config({'experiment': {'fold': -1}})

        # Zero or negative learning rate
        with self.assertRaises(ValueError):
            validate_config({'method': {'base_lr': -0.01}})

        # Mismatched class_names length and num_classes
        with self.assertRaises(ValueError):
            validate_config({'data': {'num_classes': 4, 'class_names': ['A', 'B']}})

        # Invalid task name
        with self.assertRaises(ValueError):
            validate_config({'experiment': {'task': 'invalid_task'}})

    def test_r02_mean_teacher_model_builder_clean_spec(self):
        cfg = load_config(method_config='mean_teacher', dataset_config='bhsd_multiclass')
        spec = model_spec_from_config(cfg, cfg.data.num_classes)
        self.assertEqual(spec['name'], 'nnunet3d')
        self.assertNotIn('normalization', spec['kwargs'])
        model = build_model_from_spec(spec)
        self.assertIsNotNone(model)

    def test_r02_all_presets_model_builder(self):
        presets = [
            'configs/experiments/bhsd_pick_fold0.yaml',
            'configs/experiments/bhsd_mt_nnunet_fold0.yaml',
            'configs/experiments/bhsd_cps_fold0.yaml',
            'configs/experiments/bhsd_supervised_fold0.yaml'
        ]
        for p in presets:
            cfg = load_config(config_path=p)
            spec = model_spec_from_config(cfg, cfg.data.num_classes)
            model = build_model_from_spec(spec)
            self.assertIsNotNone(model)

    def test_r02_method_model_override_clean_isolation(self):
        # Mean Teacher specifies nnunet3d, but user overrides with unet3d
        cfg = load_config(method_config='mean_teacher', model_config='unet3d', dataset_config='bhsd_multiclass')
        spec = model_spec_from_config(cfg, cfg.data.num_classes)
        self.assertEqual(spec['name'], 'unet3d')
        self.assertNotIn('num_stages', spec['kwargs'])
        self.assertNotIn('anisotropic_kernels', spec['kwargs'])
        model = build_model_from_spec(spec)
        self.assertIsNotNone(model)

    def test_r03_dataset_composition_contracts(self):
        # 1. Preset with dataset: bhsd_binary and data: {patch_size: ...}
        cfg_bin = load_config(dataset_config='bhsd_binary', cli_args=['--method', 'supervised', '--data.patch_size', '[16,32,32]'])
        self.assertEqual(cfg_bin.data.num_classes, 2)
        self.assertEqual(cfg_bin.experiment.task, 'binary')

        # 2. Preset with data.name: bhsd round-trip
        cfg_bhsd = load_config(dataset_config='bhsd')
        self.assertEqual(cfg_bhsd.data.num_classes, 6)

        # 3. Custom multiclass 4 classes with --task multiclass
        cfg_custom = load_config(cli_args=['--data.num_classes', '4', '--task', 'multiclass', '--data.class_names', '[\"c0\",\"c1\",\"c2\",\"c3\"]'])
        self.assertEqual(cfg_custom.data.num_classes, 4)

    def test_r06_model_output_gather_compatibility(self):
        from torch.nn.parallel.scatter_gather import gather
        out = ModelOutput(note=None)
        merged = gather([out, out], target_device='cpu')
        self.assertEqual(set(merged), {'note'})

        t1 = torch.tensor([1.0, 2.0])
        out_t = ModelOutput(logits=t1, aux=[t1])
        self.assertIs(out_t.logits, t1)
        self.assertEqual(len(out_t.aux), 1)

    def test_r08_schema_unification_and_no_feature_leakage(self):
        t = torch.randn(2, 6, 16, 16, 16)
        feat = torch.randn(2, 512, 4, 4, 4)
        out = ModelOutput(logits=t, features=feat)
        m, aux = extract_multiscale_outputs(out)
        self.assertTrue(torch.equal(m, t))
        self.assertEqual(len(aux), 0)

        # Custom logits with metadata dict
        d = {'metadata': {'meta': 1}, 'custom_logits': t}
        self.assertTrue(torch.equal(extract_logits(d), t))
        m2, aux2 = extract_multiscale_outputs(d)
        self.assertTrue(torch.equal(m2, t))
        self.assertEqual(len(aux2), 0)

        with self.assertRaises(TypeError):
            wrap_model_output(9999)

    def test_r09_strict_unknown_keys_and_conflict_rejection(self):
        with self.assertRaises(ValueError) as ctx:
            load_config(cli_args=['--method.bas_lr', '0.02'])
        self.assertIn('Unknown parameter', str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            load_config(dataset_config='bhsd_multiclass', cli_args=['--experiment.num_classes', '3'])
        self.assertIn('Mismatch between data.num_classes', str(ctx2.exception))

        with self.assertRaises(ValueError) as ctx3:
            load_config(cli_args=['--method.base_lr', 'inf'])
        self.assertIn('method.base_lr must be positive finite float', str(ctx3.exception))

    def test_r10_ddp_cli_override_and_validation(self):
        # DDP boolean flag from torchrun / train.py injection
        cfg = load_config(dataset_config='bhsd_multiclass', cli_args=['--experiment.ddp', 'true'])
        self.assertTrue(cfg.experiment.ddp)

        # Flat --ddp mapping
        cfg2 = load_config(dataset_config='bhsd_multiclass', cli_args=['--ddp'])
        self.assertTrue(cfg2.experiment.ddp)

        # Non-boolean ddp validation
        with self.assertRaises(ValueError) as ctx:
            validate_config({'experiment': {'ddp': 'not_a_bool'}})
        self.assertIn('experiment.ddp must be boolean', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()

