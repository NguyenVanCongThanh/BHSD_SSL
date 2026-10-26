import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.crawl_storage import new_snapshot, comparison_file


class TestCrawlStorage(unittest.TestCase):
    def test_repeated_crawls_preserve_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'BHSD_EXPERIMENT_DATA': tmp}):
            first = new_snapshot('wandb', 'run1', run_id='run1')
            (first / 'summary.json').write_text('{"iter": 340}')
            second = new_snapshot('wandb', 'run1', run_id='run1')
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads((first / 'summary.json').read_text()), {'iter': 340})
            self.assertEqual(json.loads((second / 'manifest.json').read_text())['run_id'], 'run1')

    def test_snapshot_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            new_snapshot('wandb', '../outside')

    def test_offline_comparison_selection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'BHSD_EXPERIMENT_DATA': tmp}):
            first = new_snapshot('wandb', '_collections')
            (first / 'wandb_results.json').write_text('{}')
            second = new_snapshot('wandb', '_collections')
            (second / 'wandb_results.json').write_text('{}')
            with patch('sys.argv', ['compare_runs.py']):
                self.assertEqual(comparison_file(), second / 'wandb_results.json')
            with patch('sys.argv', ['compare_runs.py', '--input', str(first / 'wandb_results.json')]):
                self.assertEqual(comparison_file(), first / 'wandb_results.json')


if __name__ == '__main__':
    unittest.main()
