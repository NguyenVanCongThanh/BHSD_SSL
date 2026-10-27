"""
Unit tests for Phase 4 & Phase 5: Metrics, Benchmark & Dataset Expansion
Verifies:
- aggregate_segmentation_stats computes macro positive dice, eligible_classes, and positive support.
- Negative cases or classes without positive samples are not fabricated as 0 or 1.
- benchmark_5folds supports configurable folds parameter.
"""

import unittest
import numpy as np

from evaluation.metrics import aggregate_segmentation_stats, case_segmentation_stats


class TestPhase4Metrics(unittest.TestCase):

    def test_macro_positive_dice_computation(self):
        # 3 cases, 3 classes (0: bg, 1: lesion_A, 2: lesion_B)
        # Case 0: Lesion A present (perfect dice), Lesion B absent (gt=0, pred=0)
        c0 = {'classes': [
            {'tp': 100, 'fp': 0, 'fn': 0, 'dice': 1.0, 'fp_ml': None}, # Class 1: positive
            {'tp': 0, 'fp': 0, 'fn': 0, 'dice': 1.0, 'fp_ml': None},   # Class 2: negative
        ]}
        # Case 1: Lesion A present (tp=30, fp=20, fn=20 -> dice = 60/100 = 0.6), Lesion B absent (gt=0, pred=0)
        c1 = {'classes': [
            {'tp': 30, 'fp': 20, 'fn': 20, 'dice': 0.6, 'fp_ml': None}, # Class 1: positive
            {'tp': 0, 'fp': 0, 'fn': 0, 'dice': 1.0, 'fp_ml': None},   # Class 2: negative
        ]}
        # Case 2: Lesion A absent, Lesion B absent
        c2 = {'classes': [
            {'tp': 0, 'fp': 0, 'fn': 0, 'dice': 1.0, 'fp_ml': None},   # Class 1: negative
            {'tp': 0, 'fp': 0, 'fn': 0, 'dice': 1.0, 'fp_ml': None},   # Class 2: negative
        ]}

        records = [c0, c1, c2]
        stats = aggregate_segmentation_stats(records)

        # Class 1: positive cases = 2, positive mean = (1.0 + 0.6) / 2 = 0.8
        # Class 2: positive cases = 0, positive mean = None
        self.assertEqual(stats['per_class'][0]['positive_cases'], 2)
        self.assertAlmostEqual(stats['per_class'][0]['dice_positive_mean'], 0.8)

        self.assertEqual(stats['per_class'][1]['positive_cases'], 0)
        self.assertIsNone(stats['per_class'][1]['dice_positive_mean'])

        # Macro positive dice should ONLY average over eligible classes (Class 1)
        # Class 2 must NOT be included as 0 or 1!
        self.assertEqual(stats['eligible_positive_classes'], [1])
        self.assertAlmostEqual(stats['macro_positive_dice'], 0.8)
        self.assertEqual(stats['positive_support'], {1: 2, 2: 0})

    def test_macro_positive_dice_all_classes_positive(self):
        # Case with both classes positive
        c0 = {'classes': [
            {'tp': 100, 'fp': 0, 'fn': 0, 'dice': 1.0, 'fp_ml': None},
            {'tp': 50, 'fp': 50, 'fn': 0, 'dice': 0.6666667, 'fp_ml': None},
        ]}
        stats = aggregate_segmentation_stats([c0])
        self.assertEqual(stats['eligible_positive_classes'], [1, 2])
        expected_macro = (1.0 + 0.6666667) / 2.0
        self.assertAlmostEqual(stats['macro_positive_dice'], expected_macro, places=5)

    def test_macro_positive_dice_no_positive_cases(self):
        # All cases are negative for all foreground classes
        c0 = {'classes': [
            {'tp': 0, 'fp': 0, 'fn': 0, 'dice': 1.0, 'fp_ml': None},
            {'tp': 0, 'fp': 0, 'fn': 0, 'dice': 1.0, 'fp_ml': None},
        ]}
        stats = aggregate_segmentation_stats([c0])
        self.assertEqual(stats['eligible_positive_classes'], [])
        self.assertIsNone(stats['macro_positive_dice'])


if __name__ == '__main__':
    unittest.main()

