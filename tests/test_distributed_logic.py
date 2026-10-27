"""Exercise real CPU/Gloo collectives and two-network synchronization."""
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from core.config_parser import Config
from methods.base_method import BaseMethod
from methods.cps import CPSMethod
from methods.mean_teacher import MeanTeacherMethod
from methods.lefed import LeFeDMethod
from methods.bcp import BCPMethod
from methods.pick import PICKMethod
from tests.test_logic_audit import FeedbackNet
from evaluation.sliding_window import evaluate_dataset_sliding_window
from utils.checkpoint import get_raw_model


class TinyPick(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv3d(1, 2, 1)
        self.main = torch.nn.Conv3d(2, 2, 1)
        self.mim = torch.nn.Conv3d(2, 1, 1)
        self.aux = torch.nn.Conv3d(2, 2, 1)

    def forward(self, image, mode=None):
        features = self.encoder(image)
        if mode == 'mim': return self.mim(features)
        if mode == 'aux': return self.aux(features)
        return [features], self.main(features), features


def _worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(Path(root) / 'rendezvous'),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        results = {}
        for cls, method in ((MeanTeacherMethod, 'mean_teacher'), (CPSMethod, 'cps'), (LeFeDMethod, 'lefed'), (BCPMethod, 'bcp'), (PICKMethod, 'pick')):
            cfg = Config({'data': {'batch_size': 8 if method in ('bcp', 'pick') else 4, 'labeled_bs': 4 if method in ('bcp', 'pick') else 2, 'task': 'binary', 'num_classes': 2},
                'model': {}, 'method': {'name': method, 'max_iteration': 2, 'base_lr': .1},
                'experiment': {'exp_dir': root, 'name': method, 'amp': False},
                'logging': {'val_stride_z': 1, 'val_stride_xy': 1}})
            def loaders(trainer): trainer.trainloader = [0]
            def logging(trainer):
                trainer.writer = None
                trainer.wandb_manager = SimpleNamespace(enabled=False, save_artifacts=False,
                    log_metrics=lambda *a, **k: None, update_summary=lambda *a, **k: None)
            with patch.object(BaseMethod, 'setup_dataloaders', loaders), \
                 patch.object(BaseMethod, 'setup_logging', logging), \
                 patch('torch.cuda.is_available', return_value=False):
                trainer = cls(TinyPick() if method == 'pick' else FeedbackNet() if method == 'lefed' else torch.nn.Sequential(torch.nn.Conv3d(1, 2, 1), torch.nn.BatchNorm3d(2)), cfg)
            batch_size = 4 if method in ('bcp', 'pick') else 2
            batch = {'image': torch.randn(batch_size, 1, 2, 4, 4) + rank,
                     'label': torch.randint(0, 2, (batch_size, 2, 4, 4))}
            trainer.optimize_batch(batch, 0)
            trainer.optimize_batch(batch, 1)
            for model in [trainer.model] + ([trainer.model2] if method == 'cps' else [trainer.ema_model] if method == 'mean_teacher' else []):
                for param in model.parameters():
                    expected = param.detach().clone()
                    dist.broadcast(expected, src=0)
                    assert torch.equal(param, expected), 'DDP parameters diverged'
            # Rank 1 has no validation cases; all ranks must still gather and agree.
            class Dataset:
                def __len__(self): return 1
                def __getitem__(self, index):
                    return {'image': np.zeros((2, 4, 4), np.float32),
                            'label': np.zeros((2, 4, 4), np.uint8)}
            trainer.db_val = Dataset()
            trainer.patch_size = (2, 4, 4)
            trainer.class_names = ['Background', 'Hemorrhage']
            with patch.object(trainer, 'save_checkpoint'):
                best = trainer._handle_periodic_validation(root, 'train', 2, 2, -.1,
                    'best.pth', None, method, str(Path(root) / 'latest.pth'), False)
            scores = [None, None]
            dist.all_gather_object(scores, best)
            assert scores[0] == scores[1]
            results[method] = {'parameters_equal': True, 'validation_best_equal': True}
            if method == 'mean_teacher':
                class CohortDataset:
                    image_list = ['case_b', 'case_a']
                    def __len__(self): return 2
                    def __getitem__(self, index): return {
                        'image': np.zeros((2,4,4), np.float32),
                        'label': np.zeros((2,4,4), np.uint8), 'case': self.image_list[index]}
                details = {}
                output = evaluate_dataset_sliding_window(trainer.ema_model, CohortDataset(),
                    num_classes=2, is_binary=True, patch_size=(2,4,4), stride_z=1, stride_xy=2,
                    device=torch.device('cpu'), return_visual_samples=1 if rank == 0 else 0,
                    visual_settings={'target_shape': [16,16]}, details_sink=details)
                assert details['num_cases'] == 2
                if rank == 0:
                    assert output[2][0]['case'] == 'case_a', 'Cohort must include selected case owned by rank 1'
                    assert output[2][0]['image'].shape == (16,16)

            # Nonfinite loss on one rank must fail on both before backward.
            from utils.model_utils import extract_logits
            loss = extract_logits(trainer.model(batch['image'])).sum()
            if rank == 1:
                loss = loss * float('nan')
            try:
                trainer.apply_gradients(loss, 2)
            except FloatingPointError:
                results[method]['nonfinite_rejected'] = True
            else:
                raise AssertionError('Nonfinite loss was not rejected')
        (Path(root) / f'rank{rank}.json').write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


class DistributedLogicTests(unittest.TestCase):
    def test_two_rank_training_and_validation(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(_worker, args=(root,), nprocs=2, join=True)
            for rank in range(2):
                results = json.loads((Path(root) / f'rank{rank}.json').read_text())
                self.assertTrue(results['cps']['parameters_equal'])
                self.assertTrue(results['mean_teacher']['validation_best_equal'])
                self.assertTrue(results['cps']['nonfinite_rejected'])


if __name__ == '__main__':
    unittest.main()
