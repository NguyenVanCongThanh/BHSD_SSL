"""Batch contract checks; arithmetic tests also run without PyTorch installed."""
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('batch_contract', ROOT / 'core/batching.py')
batching = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batching
spec.loader.exec_module(batching)
resolve_batch_layout = batching.resolve_batch_layout


class TestGlobalBatchContract(unittest.TestCase):
    def test_mean_teacher_two_ranks(self):
        layout = resolve_batch_layout(4, 2, 2)
        self.assertEqual((layout.local_batch_size, layout.local_labeled_bs,
                          layout.local_unlabeled_bs), (2, 1, 1))
        self.assertEqual(layout.global_batch_size, 4)

    def test_single_process_including_data_parallel(self):
        layout = resolve_batch_layout(4, 2)
        self.assertEqual((layout.local_batch_size, layout.local_labeled_bs), (4, 2))

    def test_rejects_rounding_and_missing_streams(self):
        for args in [(5, 2, 2), (4, 1, 2), (4, 4, 2), (4, 0, 2), (0, 0, 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_batch_layout(*args)

    def test_paired_methods_require_balanced_local_pairs(self):
        for method in ['pick', 'bcp']:
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, 'paired CutMix'):
                    resolve_batch_layout(4, 2, 2, method)
                with self.assertRaises(ValueError):
                    resolve_batch_layout(12, 4, 2, method)
                layout = resolve_batch_layout(8, 4, 2, method)
                self.assertEqual((layout.local_labeled_bs, layout.local_unlabeled_bs), (2, 2))

    def test_supervised_uses_full_labeled_batch(self):
        layout = resolve_batch_layout(4, 2, 2, 'supervised')
        self.assertEqual((layout.global_labeled_bs, layout.local_labeled_bs,
                          layout.local_unlabeled_bs), (4, 2, 0))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class TestBatchIntegration(unittest.TestCase):
    def test_trainers_use_local_counts_and_supervised_labeled_pool(self):
        import torch
        import tempfile
        from unittest.mock import patch
        from core.config_parser import Config
        from methods.base_method import BaseMethod
        from methods.mean_teacher import MeanTeacherMethod
        from methods.supervised import SupervisedMethod

        class CpuDDP(torch.nn.Module):
            def __init__(self, model, **kwargs):
                super().__init__()
                self.module = model

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        dataset = torch.utils.data.TensorDataset(torch.arange(20))
        for method, trainer_type in [('mean_teacher', MeanTeacherMethod),
                                     ('supervised', SupervisedMethod)]:
            with self.subTest(method=method):
                experiment_dir = tempfile.TemporaryDirectory()
                self.addCleanup(experiment_dir.cleanup)
                cfg = Config({'data': {'batch_size': 4, 'labeled_bs': 2,
                                       'labelnum': 8, 'extended_aug': False},
                              'method': {'name': method},
                              'model': {'deep_supervision': False},
                              'experiment': {'exp_dir': experiment_dir.name}, 'logging': {}})
                def fake_environment(trainer):
                    self.assertEqual(trainer.labeled_bs, 1 if method == 'mean_teacher' else 2)
                    trainer.device = torch.device('cpu')
                # Exercise real constructor + dataloader setup, with no process group/GPU.
                with patch('methods.base_method.is_dist_avail_and_initialized', return_value=True), \
                     patch('methods.base_method.get_world_size', return_value=2), \
                     patch('methods.base_method.get_rank', return_value=0), \
                     patch.object(BaseMethod, '_setup_environment', fake_environment), \
                     patch.object(torch.nn.parallel, 'DistributedDataParallel', CpuDDP), \
                     patch.object(BaseMethod, 'setup_optimizer'), \
                     patch.object(BaseMethod, 'setup_logging'), \
                     patch('data.dataloader_factory.DATASETS.build', return_value=dataset), \
                     patch('torch.cuda.is_available', return_value=False):
                    trainer = trainer_type(torch.nn.Conv3d(1, 2, 1), cfg)
                self.assertEqual((cfg.data.batch_size, cfg.data.labeled_bs), (4, 2))
                self.assertEqual(trainer.labeled_bs, 1 if method == 'mean_teacher' else 2)
                if method == 'supervised':
                    self.assertIsInstance(trainer.trainloader.dataset, torch.utils.data.Subset)
                    self.assertEqual(list(trainer.trainloader.dataset.indices), list(range(8)))
                    self.assertEqual(trainer.trainloader.batch_size, 2)
                else:
                    batch = next(iter(trainer.batch_sampler))
                    self.assertEqual(len(batch), 2)
                    self.assertLess(batch[0], 8)
                    self.assertGreaterEqual(batch[1], 8)

    def test_sampler_and_mean_teacher_labeled_slice(self):
        import torch
        from data.samplers import DistributedTwoStreamBatchSampler
        from methods.ssl_utils import split_ssl_batch
        layout = resolve_batch_layout(4, 2, 2)
        indices = []
        for rank in range(2):
            sampler = DistributedTwoStreamBatchSampler(
                list(range(154)), list(range(154, 2200)),
                batch_size=layout.local_batch_size,
                secondary_batch_size=layout.local_unlabeled_bs,
                num_replicas=2, rank=rank,
            )
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            batch = batches[0]
            self.assertEqual(len(batch), 2)
            self.assertLess(batch[0], 154)
            self.assertGreaterEqual(batch[1], 154)
            split = split_ssl_batch({'image': torch.tensor(batch), 'label': torch.tensor(batch)},
                                    layout.local_labeled_bs)
            self.assertEqual(len(split.labeled_volume), 1)
            self.assertEqual(len(split.unlabeled_volume), 1)
            indices.extend(batch)
        self.assertEqual(len(set(indices)), 4)

    def test_uneven_pool_has_equal_steps_on_all_ranks(self):
        from data.samplers import DistributedTwoStreamBatchSampler
        counts = []
        for rank in range(2):
            sampler = DistributedTwoStreamBatchSampler(
                list(range(155)), list(range(155, 200)), 2, 1,
                num_replicas=2, rank=rank,
            )
            counts.append(len(list(sampler)))
            self.assertEqual(counts[-1], len(sampler))
        self.assertEqual(counts, [77, 77])


if __name__ == '__main__':
    unittest.main()
