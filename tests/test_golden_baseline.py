"""
Golden baseline regression tests for MedSSL-BHSD.
Freezes mathematical baselines for loss functions, consistency regularization,
and evaluation metrics to ensure zero regression across architectural refactoring.
"""
import unittest
import numpy as np
import torch

from utils.reproducibility import set_seed
from losses.dice import MaskDiceLoss
from losses.compound import DiceCELoss
from losses.consistency import ConsistencyLoss
from evaluation.metrics import (
    cal_multiclass_dice,
    cal_binary_dice,
    case_segmentation_stats,
    aggregate_segmentation_stats,
)


class TestGoldenBaseline(unittest.TestCase):
    def setUp(self):
        set_seed(1337, deterministic=True)
        torch.set_num_threads(1)
        # Deterministic synthetic logits and target
        torch.manual_seed(1337)
        self.logits = torch.randn(2, 6, 8, 16, 16, dtype=torch.float32)
        self.target = torch.randint(0, 6, (2, 8, 16, 16), dtype=torch.long)
        self.pred_np = torch.argmax(self.logits[0], dim=0).numpy().astype(np.uint8)
        self.gt_np = self.target[0].numpy().astype(np.uint8)

    def test_golden_mask_dice_loss(self):
        """Verify MaskDiceLoss numerical value matches frozen baseline."""
        dice_loss_fn = MaskDiceLoss(nclass=6, include_background=False)
        val = dice_loss_fn(self.logits, self.target).item()
        self.assertAlmostEqual(val, 0.830743, places=5)

    def test_golden_dice_ce_compound_loss(self):
        """Verify DiceCELoss numerical value matches frozen baseline."""
        dice_ce_fn = DiceCELoss(nclass=6, weight_dice=0.5, weight_ce=0.5, include_background=False)
        val = dice_ce_fn(self.logits, self.target).item()
        self.assertAlmostEqual(val, 1.498944, places=5)

    def test_golden_consistency_hybrid_loss(self):
        """Verify ConsistencyLoss hybrid mode matches frozen baseline."""
        cons_fn = ConsistencyLoss(consistency_type='hybrid', num_classes=6, is_binary=False)
        val = cons_fn(self.logits, self.logits * 0.9).item()
        self.assertAlmostEqual(val, 0.001025, places=5)

    def test_golden_evaluation_metrics(self):
        """Verify multiclass Dice, binary Dice, and aggregation match frozen baseline."""
        mc_dice = cal_multiclass_dice(self.pred_np, self.gt_np, num_classes=6)
        expected_mc = [0.176066, 0.186916, 0.160819, 0.164074, 0.161832]
        for got, exp in zip(mc_dice, expected_mc):
            self.assertAlmostEqual(got, exp, places=5)

        bin_dice = cal_binary_dice((self.pred_np > 0).astype(np.uint8), (self.gt_np > 0).astype(np.uint8))
        self.assertAlmostEqual(bin_dice, 0.830454, places=5)

        stats = case_segmentation_stats(self.pred_np, self.gt_np, num_classes=6, spacing=(1.0, 1.0, 1.0))
        agg = aggregate_segmentation_stats([stats])
        self.assertAlmostEqual(agg['per_class'][0]['dice_case_mean'], 0.176066, places=5)


if __name__ == '__main__':
    unittest.main()

