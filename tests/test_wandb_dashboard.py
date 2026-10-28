"""
Unit tests for tools/setup_wandb_dashboard.py
"""
import unittest
import subprocess
import sys
from pathlib import Path

from tools.setup_wandb_dashboard import (
    build_overview_workspace,
    build_evaluation_workspace,
    build_diagnostics_workspace,
    inspect_workspace,
    BHSD_FOREGROUND_CLASSES,
    BHSD_BINARY_CLASSES
)


class TestWandbDashboardSetup(unittest.TestCase):
    def test_overview_workspace_layout(self):
        ws = build_overview_workspace("entity", "PICK_BHSD", BHSD_FOREGROUND_CLASSES)
        summary = inspect_workspace(ws)
        self.assertEqual(summary["view_name"], "Overview")
        self.assertEqual(summary["total_sections"], 2)
        self.assertEqual(summary["total_panels"], 7)
        self.assertFalse(summary["auto_generate_panels"])

        # Check section names
        sec_names = [s["section_name"] for s in summary["sections"]]
        self.assertIn("1. Core Validation Performance", sec_names)
        self.assertIn("2. Training Dynamics", sec_names)

        # Ensure learning rate and rampup weight are separated
        titles = [p["title"] for s in summary["sections"] for p in s["panels"]]
        self.assertIn("Learning Rate", titles)
        self.assertIn("Consistency Loss Ramp-up Weight", titles)

    def test_evaluation_workspace_layout(self):
        ws = build_evaluation_workspace("entity", "PICK_BHSD", BHSD_FOREGROUND_CLASSES)
        summary = inspect_workspace(ws)
        self.assertEqual(summary["view_name"], "Evaluation")
        self.assertEqual(summary["total_sections"], 3)
        self.assertEqual(summary["total_panels"], 7)

        titles = [p["title"] for s in summary["sections"] for p in s["panels"]]
        self.assertTrue(any("HD95" in t for t in titles))
        self.assertTrue(any("ASD" in t for t in titles))
        self.assertTrue(any("Gallery" in t for t in titles))

    def test_diagnostics_workspace_layout(self):
        ws = build_diagnostics_workspace("entity", "PICK_BHSD", BHSD_FOREGROUND_CLASSES)
        summary = inspect_workspace(ws)
        self.assertEqual(summary["view_name"], "Diagnostics")
        self.assertEqual(summary["total_sections"], 2)
        self.assertEqual(summary["total_panels"], 6)

        titles = [p["title"] for s in summary["sections"] for p in s["panels"]]
        self.assertTrue(any("Positive Case Count" in t for t in titles))
        self.assertTrue(any("True Positive Voxels" in t for t in titles))

    def test_binary_classes_support(self):
        ws = build_overview_workspace("entity", "PICK_BHSD", BHSD_BINARY_CLASSES)
        summary = inspect_workspace(ws)
        self.assertEqual(summary["total_panels"], 7)
        # Class dice panel should only have 1 series
        dice_panel = summary["sections"][0]["panels"][2]
        self.assertEqual(len(dice_panel["y_series"]), 1)
        self.assertEqual(dice_panel["y_series"][0], "val/dice_Hemorrhage")

    def test_cli_dry_run(self):
        script_path = Path(__file__).resolve().parents[1] / "tools" / "setup_wandb_dashboard.py"
        res = subprocess.run(
            [sys.executable, str(script_path), "--dry-run", "--project", "TEST_PROJ"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0, f"CLI dry-run failed with:\n{res.stderr}")
        self.assertIn("VIEW: Overview (Panels: 7", res.stdout + res.stderr)
        self.assertIn("VIEW: Evaluation (Panels: 7", res.stdout + res.stderr)
        self.assertIn("VIEW: Diagnostics (Panels: 6", res.stdout + res.stderr)
        self.assertIn("[DRY-RUN] Skipped saving", res.stdout + res.stderr)


if __name__ == "__main__":
    unittest.main()

