"""
Unit & Integration Tests for the MedSSL-BHSD Research Method Visualization System.
Compatible with both unittest and pytest.
"""

import unittest
from pathlib import Path

from visualizer.bridge import (
    build_config_presets,
    collect_code_snippets,
    load_empirical_runs,
)
from visualizer.schemas.backbones import get_backbone_specs
from visualizer.schemas.methods import get_method_specs
from tools.build_visualizer import build_data_bundle, compile_html


class TestVisualizerSystem(unittest.TestCase):
    def test_config_presets_match_repo_yamls(self):
        presets = build_config_presets()
        resolved = presets["resolved"]
        mt_cfg = resolved["mean_teacher_default"]
        mt_fold0 = resolved["mean_teacher_fold0_preset"]

        self.assertEqual(mt_cfg["method"]["name"], "mean_teacher")
        self.assertEqual(mt_cfg["data"]["patch_size"], [28, 256, 256])
        self.assertEqual(mt_cfg["data"]["batch_size"], 8)
        self.assertEqual(mt_cfg["data"]["labeled_bs"], 4)
        self.assertEqual(mt_cfg["method"]["consistency"], 0.05)
        self.assertEqual(mt_cfg["method"]["consistency_rampup"], 4000.0)
        self.assertEqual(mt_cfg["method"]["ema_decay"], 0.999)
        self.assertEqual(mt_cfg["method"]["ema_schedule"], "cosine")

        # Verify config inheritance nuance in bhsd_mt_nnunet_fold0.yaml
        self.assertEqual(mt_fold0["data"]["patch_size"], [32, 192, 192])
        self.assertEqual(mt_fold0["data"]["batch_size"], 4)
        self.assertEqual(mt_fold0["method"]["consistency"], 1.0)
        self.assertEqual(mt_fold0["method"]["consistency_rampup"], 40.0)
        self.assertEqual(mt_fold0["method"]["ema_decay"], 0.99)
        self.assertEqual(mt_fold0["method"]["ema_schedule"], "cosine")

    def test_code_snippets_and_empirical_runs(self):
        snippets = collect_code_snippets()
        required_keys = {
            "mt_init",
            "mt_consistency_schedule",
            "mt_train_step",
            "mt_ema_hook",
            "ema_update_fn",
            "consistency_loss",
            "nnunet_arch",
            "gradient_handler",
            "anti_leakage_crop",
        }
        self.assertTrue(required_keys.issubset(set(snippets.keys())))
        for k, item in snippets.items():
            self.assertGreater(len(item["code"]), 20, f"Snippet {k} should not be empty")
            self.assertIn("|", item["code"], f"Snippet {k} should contain line numbers")

        runs = load_empirical_runs()
        self.assertIn("o6s4pex3", runs)
        self.assertIn("9tvq8qld", runs)
        self.assertEqual(len(runs["o6s4pex3"]["validation_trajectory"]), 42)
        self.assertEqual(len(runs["9tvq8qld"]["validation_trajectory"]), 64)

    def test_declarative_method_and_backbone_schemas(self):
        backbones = get_backbone_specs()
        self.assertTrue({"nnunet3d", "vnet", "unet3d"}.issubset(set(backbones.keys())))
        self.assertEqual(
            backbones["nnunet3d"]["stage_rules"]["channels_by_stages"]["7"],
            [32, 64, 128, 256, 320, 320, 320],
        )

        methods = get_method_specs()
        expected_methods = {"mean_teacher", "cps", "bcp", "pick", "lefed", "supervised"}
        self.assertTrue(expected_methods.issubset(set(methods.keys())))

        mt = methods["mean_teacher"]
        self.assertEqual(len(mt["nodes"]), 8)
        self.assertEqual(len(mt["edges"]), 12)
        self.assertEqual(len(mt["execution_steps"]), 8)

        for m_id, m_spec in methods.items():
            node_ids = {n["id"] for n in m_spec["nodes"]}
            for edge in m_spec["edges"]:
                self.assertIn(edge["from"], node_ids, f"{m_id}: edge.from {edge['from']} missing")
                self.assertIn(edge["to"], node_ids, f"{m_id}: edge.to {edge['to']} missing")

    def test_compile_html_bundle(self):
        bundle = build_data_bundle()
        html = compile_html(bundle)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("window.MEDSSL_BUNDLE =", html)
        self.assertIn("window.MedSSLCore =", html)
        self.assertIn("window.MedSSLViews =", html)
        self.assertIn("MeanTeacherMethod", html)


if __name__ == "__main__":
    unittest.main()
