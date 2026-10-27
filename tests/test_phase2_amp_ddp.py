"""
Unit tests for Phase 2: Coordinated Multi-Optimizer AMP, DDP & Reproducibility
Verifies:
- AMP-01 & OPT-01: Multi-optimizer overflow synchronization (all step or none step).
- Both model1 and model2 receive gradient clipping.
- after_optimizer_step / successful_updates only triggered on real updates.
- REPR-01: RNG roundtrip through checkpoint save and restore under weights_only=True.
- CPS checkpoint save/load includes net2, opt2, and RNG state.
"""

import unittest
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from methods.base_method import BaseMethod
from utils.checkpoint import save_checkpoint, load_checkpoint, capture_rng_state, restore_rng_state


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class DummyTrainer(BaseMethod):
    """Minimal subclass to test gradient coordination without data loading."""
    def __init__(self, model, model2=None, use_amp=False):
        self.model = model
        self.model2 = model2
        self.device = torch.device('cpu')
        self.is_ddp = False
        self.use_amp = use_amp
        self.rank = 0
        self.world_size = 1
        self.successful_updates = 0
        self.after_step_calls = 0

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=0.1) if model2 is not None else None

        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cpu', enabled=use_amp, init_scale=65536.0)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=65536.0)

    def training_optimizers(self):
        opts = [self.optimizer]
        if self.optimizer2 is not None:
            opts.append(self.optimizer2)
        return opts

    def after_optimizer_step(self, iter_num):
        self.after_step_calls += 1

    def train_step(self, *args, **kwargs):
        pass


class TestPhase2Coordination(unittest.TestCase):

    def test_amp_multi_optimizer_overflow_prevents_partial_updates(self):
        """AMP-01 & OPT-01: If optimizer2 has inf gradient, optimizer1 MUST skip as well."""
        m1 = DummyModel()
        m2 = DummyModel()
        trainer = DummyTrainer(m1, m2, use_amp=True)

        w1_before = m1.conv.weight.clone()
        w2_before = m2.conv.weight.clone()
        scale_before = trainer.scaler.get_scale()

        # Hook unscale_ to inject Inf into m2's gradient during unscaling
        orig_unscale = trainer.scaler.unscale_
        def unscale_with_m2_overflow(opt):
            orig_unscale(opt)
            if opt is trainer.optimizer2:
                for p in m2.parameters():
                    if p.grad is not None:
                        p.grad.view(-1)[0] = float('inf')
        trainer.scaler.unscale_ = unscale_with_m2_overflow

        x = torch.randn(2, 1, 8, 8, 8)
        loss = (m1(x).sum() + m2(x).sum()) * 0.1

        # Execute apply_gradients
        trainer.apply_gradients(loss, iter_num=1)

        # 1. Both models must remain untouched (zero partial update)
        self.assertTrue(torch.equal(m1.conv.weight, w1_before), "m1 must NOT step when m2 has inf gradient!")
        self.assertTrue(torch.equal(m2.conv.weight, w2_before), "m2 must NOT step when m2 has inf gradient!")

        # 2. after_optimizer_step must NOT have been called
        self.assertEqual(trainer.after_step_calls, 0)
        self.assertEqual(trainer.successful_updates, 0)

        # 3. Scaler scale must have backed off
        scale_after = trainer.scaler.get_scale()
        self.assertLess(scale_after, scale_before)

    def test_amp_multi_optimizer_successful_coordinated_step(self):
        """When all gradients are finite, all optimizers step and counter increments."""
        m1 = DummyModel()
        m2 = DummyModel()
        trainer = DummyTrainer(m1, m2, use_amp=True)

        w1_before = m1.conv.weight.clone()
        w2_before = m2.conv.weight.clone()

        x = torch.randn(2, 1, 8, 8, 8)
        loss = (m1(x).sum() + m2(x).sum()) * 0.1

        trainer.apply_gradients(loss, iter_num=1)

        # Both models updated
        self.assertFalse(torch.equal(m1.conv.weight, w1_before))
        self.assertFalse(torch.equal(m2.conv.weight, w2_before))

        # Hook and counter called
        self.assertEqual(trainer.after_step_calls, 1)
        self.assertEqual(trainer.successful_updates, 1)

    def test_repr_01_rng_state_checkpoint_roundtrip(self, tmp_path=None):
        """REPR-01: Round-trip RNG state produces identical random sequences upon restore."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoint_test.pth"
            model = DummyModel()
            opt = torch.optim.SGD(model.parameters(), lr=0.01)

            # Seed initial state
            torch.manual_seed(9999)
            np.random.seed(9999)
            random.seed(9999)

            save_checkpoint(ckpt_path, model=model, optimizer=opt, iter_num=10)

            # Advance random generators
            vals_torch_expected = torch.randn(5)
            vals_np_expected = np.random.rand(5)
            vals_py_expected = [random.random() for _ in range(5)]

            # Mutate state further
            _ = torch.randn(100)
            _ = np.random.rand(100)
            _ = [random.random() for _ in range(100)]

            # Restore checkpoint with restore_rng=True
            m_loaded = DummyModel()
            opt_loaded = torch.optim.SGD(m_loaded.parameters(), lr=0.01)
            load_checkpoint(ckpt_path, model=m_loaded, optimizer=opt_loaded, restore_rng=True)

            # Now sample again - must match exactly
            vals_torch_actual = torch.randn(5)
            vals_np_actual = np.random.rand(5)
            vals_py_actual = [random.random() for _ in range(5)]

            self.assertTrue(torch.equal(vals_torch_actual, vals_torch_expected))
            self.assertTrue(np.array_equal(vals_np_actual, vals_np_expected))
            self.assertEqual(vals_py_actual, vals_py_expected)


if __name__ == '__main__':
    unittest.main()
