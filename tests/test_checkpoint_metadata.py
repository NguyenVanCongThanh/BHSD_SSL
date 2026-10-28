"""Unit tests for training/checkpoint_metadata.py."""
import unittest

from evaluation.metrics import METRIC_POLICY
from training.checkpoint_metadata import build_checkpoint_metadata


class TestCheckpointMetadata(unittest.TestCase):
    def test_build_checkpoint_metadata_keys_and_precedence(self):
        config_dict = {'model_spec': {'name': 'vnet'}, 'data': {'name': 'bhsd'}}
        wandb_ident = {'run_id': 'run_123', 'entity': 'test_entity'}
        val_opts = {'patch_size': (32, 64, 64), 'stride_xy': 16}
        val_details = {'macro_positive_dice': 0.85}
        extra = {'ema_net': {'weight': 1}, 'checkpoint_kind': 'best', 'successful_updates': 99}

        meta = build_checkpoint_metadata(
            config_dict=config_dict,
            wandb_identity=wandb_ident,
            validation_options=val_opts,
            successful_updates=42,
            validation_details=val_details,
            extra_state=extra
        )

        self.assertEqual(meta['config'], config_dict)
        self.assertEqual(meta['wandb_run'], wandb_ident)
        self.assertEqual(meta['metric_policy'], METRIC_POLICY)
        self.assertEqual(meta['validation_inference'], val_opts)
        self.assertEqual(meta['successful_updates'], 99)
        self.assertEqual(meta['validation_metrics'], val_details)
        self.assertEqual(meta['model_spec'], {'name': 'vnet'})
        self.assertEqual(meta['ema_net'], {'weight': 1})
        self.assertEqual(meta['checkpoint_kind'], 'best')

    def test_build_checkpoint_metadata_without_optional_fields(self):
        meta = build_checkpoint_metadata(
            config_dict=None,
            wandb_identity=None,
            validation_options=None,
            successful_updates=0,
            validation_details=None,
            extra_state=None
        )
        self.assertIsNone(meta['config'])
        self.assertNotIn('wandb_run', meta)
        self.assertNotIn('validation_metrics', meta)
        self.assertNotIn('model_spec', meta)
        self.assertEqual(meta['successful_updates'], 0)
        self.assertEqual(meta['validation_inference'], {})

    def test_build_checkpoint_metadata_does_not_mutate_inputs(self):
        extra = {'key': 'val'}
        extra_orig = dict(extra)
        val_opts = {'stride': 1}
        val_opts_orig = dict(val_opts)
        meta = build_checkpoint_metadata(
            config_dict={'a': 1},
            validation_options=val_opts,
            extra_state=extra
        )
        meta['new_field'] = 'mutation'
        self.assertEqual(extra, extra_orig)
        self.assertEqual(val_opts, val_opts_orig)

