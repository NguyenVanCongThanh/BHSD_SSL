"""Export training iterations independently of W&B's append-only history index."""
from types import SimpleNamespace
import unittest

from tools.fetch_wandb import extract_metrics


class WandbExportTests(unittest.TestCase):
    def extract(self, history):
        run = SimpleNamespace(
            id='test', name='test', summary={}, config={}, state='finished',
            created_at='', url='', scan_history=lambda: iter(history),
            logged_artifacts=lambda: [],
        )
        return extract_metrics(run)['history']

    def test_global_step_joins_split_rows_in_either_order(self):
        train = {'_step': 24, 'global_step': 250, 'train/total': .5}
        validation = {'_step': 25, 'global_step': 250, 'val/mDice': .8}
        for history in ([train, validation], [validation, train]):
            with self.subTest(history=history):
                rows = self.extract(history)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['iter'], 250)
                self.assertEqual(rows[0]['global_step'], 250)
                self.assertEqual(rows[0]['train/total'], .5)
                self.assertEqual(rows[0]['val/mDice'], .8)
        self.assertNotIn('iter', validation)

    def test_does_not_fill_losses_from_other_iterations(self):
        rows = self.extract([
            {'_step': 1, 'global_step': 240, 'train/total': .5},
            {'_step': 2, 'global_step': 250, 'val/mDice': .8},
        ])
        self.assertNotIn('train/total', rows[0])

    def test_legacy_iterations_and_sdk_step_fallback(self):
        rows = self.extract([
            {'_step': 1, 'iter': 500, 'train/total': .4, 'val/mDice': .7},
            {'_step': 2, 'val/mDice': .8},
        ])
        self.assertEqual([row['iter'] for row in rows], [500, 2])
        self.assertEqual(rows[0]['train/total'], .4)

    def test_global_step_zero_takes_precedence(self):
        rows = self.extract([{'_step': 9, 'iter': 100, 'global_step': 0, 'val/mDice': .1}])
        self.assertEqual(rows[0]['iter'], 0)
