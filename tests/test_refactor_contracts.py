"""
Regression tests verifying refactored constant externalization, YAML/env interpolation,
parameterized architectures/losses/augmentations, AST code extraction, tooling flexibility,
and Kaggle notebook preset generation.
"""

import ast
import importlib.util
import io
import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from core.config_parser import load_config, validate_config
from data.constants import (
    BHSD_BINARY_NAMES,
    BHSD_MULTICLASS_NAMES,
    DEFAULT_CLIP_NORM,
    get_dataset_class_names,
    get_foreground_class_names,
)
from data.dataloader_factory import _extract_aug_config, _resolve_data_root, build_bhsd_transforms
from data.transforms3d import GammaTransform, GaussianBlur3D, RandomCrop, RandomScaling3D
from losses.consistency import ConsistencyLoss
from methods.base_method import BaseMethod
from models.backbones.nnunet3d import nnUNet3D
from models.backbones.vnet import VNet7Stage
from tools.compare_runs import print_classes, print_summary
from tools.generate_kaggle_notebook import build_notebook_dict
from tools.generate_mean_teacher_notebook import build_mean_teacher_notebook
from tools.generate_supervised_notebook import build_supervised_notebook
from tools.kaggle.notebook_templates import load_notebook_presets
from visualizer.bridge import CODE_SNIPPET_SPECS, collect_code_snippets, load_empirical_runs


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRefactorContracts(unittest.TestCase):
    def test_config_env_interpolation_and_standalone_parser(self):
        with patch.dict(
            os.environ,
            {
                "MEDSSL_EXP_DIR": "/tmp/custom_exp",
                "MEDSSL_DATA_ROOT": "/tmp/custom_bhsd",
            },
            clear=False,
        ):
            cfg = load_config(config_path=str(REPO_ROOT / "configs" / "base.yaml"))
            self.assertEqual(cfg.experiment.exp_dir, "/tmp/custom_exp")
            self.assertEqual(cfg.data.root_path, "/tmp/custom_bhsd")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEDSSL_EXP_DIR", None)
            os.environ.pop("MEDSSL_DATA_ROOT", None)
            cfg_default = load_config(config_path=str(REPO_ROOT / "configs" / "base.yaml"))
            self.assertFalse(cfg_default.experiment.exp_dir.startswith("~"))
            self.assertTrue(cfg_default.experiment.exp_dir.endswith("/experiments"))
            self.assertEqual(_resolve_data_root(cfg_default), os.path.expanduser("~/notebook/preprocessed_bhsd_pick/"))

        # Validate standalone import isolation of core/config_parser.py (no internal imports)
        with tempfile.TemporaryDirectory() as tmpdir:
            copied = Path(tmpdir) / "config_parser.py"
            copied.write_text((REPO_ROOT / "core" / "config_parser.py").read_text(encoding="utf-8"), encoding="utf-8")
            spec = importlib.util.spec_from_file_location("standalone_config_parser", copied)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.validate_config({
                "data": {"foreground_prob": 0.5, "aug_scale_prob": 0.1},
                "method": {"clip_norm": 8.0, "sup_ce_weight": 0.4, "sup_dice_weight": 0.6},
            })

    def test_dataset_class_registry_and_resolution(self):
        self.assertEqual(get_dataset_class_names("bhsd", "multiclass"), BHSD_MULTICLASS_NAMES)
        self.assertEqual(get_dataset_class_names("bhsd", "binary"), BHSD_BINARY_NAMES)
        self.assertEqual(get_foreground_class_names("bhsd", "binary"), ["Hemorrhage"])
        self.assertEqual(get_foreground_class_names("acdc"), ["RightVentricle", "Myocardium", "LeftVentricle"])
        self.assertEqual(get_foreground_class_names("la"), ["LeftAtrium"])
        self.assertEqual(get_foreground_class_names("pancreas"), ["Pancreas"])
        self.assertEqual(get_dataset_class_names("unknown_ds", num_classes=4), ["Background", "Class_1", "Class_2", "Class_3"])

    def test_consistency_loss_parameterized_weights_and_threshold(self):
        torch.manual_seed(0)
        s_logits = torch.randn(2, 6, 4, 8, 8)
        t_logits = torch.randn(2, 6, 4, 8, 8)

        loss_default = ConsistencyLoss(consistency_type="hybrid", num_classes=6)
        loss_dice_only = ConsistencyLoss(consistency_type="hybrid", num_classes=6, weight_dice=1.0, weight_mse=0.0)
        loss_mse_only = ConsistencyLoss(consistency_type="hybrid", num_classes=6, weight_dice=0.0, weight_mse=1.0)

        v_def = loss_default(s_logits, t_logits).item()
        v_dice = loss_dice_only(s_logits, t_logits).item()
        v_mse = loss_mse_only(s_logits, t_logits).item()
        self.assertAlmostEqual(v_def, 0.5 * v_dice + 0.5 * v_mse, places=5)

    def test_backbones_parameterized_with_identical_default_state_dict(self):
        m_nn = nnUNet3D(in_channels=1, num_classes=2, num_stages=7, deep_supervision=True)
        m_nn_explicit = nnUNet3D(
            in_channels=1,
            num_classes=2,
            num_stages=7,
            deep_supervision=True,
            dropout_prob=0.2,
            max_filters=320,
            stage_multipliers=(1, 2, 4, 8, 10, 10, 10),
        )
        m_nn_custom = nnUNet3D(
            in_channels=1,
            num_classes=2,
            num_stages=7,
            deep_supervision=True,
            has_dropout=True,
            dropout_prob=0.1,
            max_filters=64,
            stage_multipliers=(1, 1, 2, 2, 2, 2, 2),
        )
        self.assertEqual(set(m_nn.state_dict().keys()), set(m_nn_explicit.state_dict().keys()))
        self.assertEqual(set(m_nn.state_dict().keys()), set(m_nn_custom.state_dict().keys()))
        for key in m_nn.state_dict():
            self.assertEqual(m_nn.state_dict()[key].shape, m_nn_explicit.state_dict()[key].shape)
        self.assertLess(m_nn_custom.enc5.conv[0].out_channels, m_nn.enc5.conv[0].out_channels)
        self.assertAlmostEqual(m_nn_custom.dropout.p, 0.1)

        m_vn = VNet7Stage(n_channels=1, n_classes=2, has_dropout=True)
        m_vn_explicit = VNet7Stage(
            n_channels=1,
            n_classes=2,
            has_dropout=True,
            dropout_prob=0.2,
            stage_multipliers=(1, 2, 4, 8, 10, 10, 10),
        )
        m_vn_custom = VNet7Stage(n_channels=1, n_classes=2, has_dropout=True, dropout_prob=0.35, stage_multipliers=(1, 1, 2, 2, 2, 2, 2))
        self.assertEqual(set(m_vn.state_dict().keys()), set(m_vn_explicit.state_dict().keys()))
        for key in m_vn.state_dict():
            self.assertEqual(m_vn.state_dict()[key].shape, m_vn_explicit.state_dict()[key].shape)
        self.assertAlmostEqual(m_vn_custom.dropout.p, 0.35)
        self.assertEqual(m_vn_custom.enc5.conv1.out_channels, 64)

    def test_base_method_clip_norm_and_dummy_subclass(self):
        class DummyMethod(BaseMethod):
            def __init__(self, cfg_obj=None):
                if cfg_obj is not None:
                    self.cfg = cfg_obj
                self.model = torch.nn.Linear(4, 2)
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
                self.use_amp = False
                self.is_ddp = False
                self.device = torch.device("cpu")

            def train_step(self, *args, **kwargs):
                return torch.tensor(0.0), {}

        dummy_no_cfg = DummyMethod()
        loss1 = dummy_no_cfg.model(torch.ones(2, 4)).sum()
        self.assertTrue(dummy_no_cfg.apply_gradients(loss1, iter_num=1))
        self.assertEqual(dummy_no_cfg._gradient_handler.clip_norm, DEFAULT_CLIP_NORM)

        cfg_custom = types.SimpleNamespace(method=types.SimpleNamespace(clip_norm=5.5))
        dummy_with_cfg = DummyMethod(cfg_obj=cfg_custom)
        loss2 = dummy_with_cfg.model(torch.ones(2, 4)).sum()
        self.assertTrue(dummy_with_cfg.apply_gradients(loss2, iter_num=1))
        self.assertEqual(dummy_with_cfg._gradient_handler.clip_norm, 5.5)

    def test_aug_config_and_new_yaml_keys_accepted(self):
        validate_config({
            "data": {
                "foreground_prob": 0.45,
                "strong_voxel_thresh": 500,
                "aug_scale_range": [0.9, 1.1],
                "aug_scale_prob": 0.15,
                "aug_gamma_range": [0.8, 1.2],
                "aug_gamma_prob": 0.25,
                "aug_blur_sigma_range": [0.4, 0.8],
                "aug_blur_prob": 0.1,
            },
            "method": {
                "clip_norm": 6.0,
                "consistency_dice_weight": 0.7,
                "consistency_mse_weight": 0.3,
                "fg_confidence_max_thresh": 0.5,
                "sup_ce_weight": 0.4,
                "sup_dice_weight": 0.6,
                "pretrain_pseudo_thresh_init": 0.7,
                "pretrain_pseudo_thresh_late": 0.8,
                "pretrain_pseudo_switch_iter": 1000,
                "selftrain_pseudo_thresh": 0.75,
            },
        })
        fake_cfg = types.SimpleNamespace(
            data=types.SimpleNamespace(
                aug_scale_range=[0.9, 1.1],
                aug_scale_prob=0.15,
                aug_gamma_range=[0.8, 1.2],
                aug_gamma_prob=0.25,
                aug_blur_sigma_range=[0.4, 0.8],
                aug_blur_prob=0.1,
            )
        )
        aug_cfg = _extract_aug_config(fake_cfg)
        self.assertEqual(aug_cfg["scale_range"], (0.9, 1.1))
        self.assertAlmostEqual(aug_cfg["blur_prob"], 0.1)

        pipeline = build_bhsd_transforms((16, 64, 64), extended_aug=True, foreground_prob=0.45, aug_config=aug_cfg)
        t_types = {type(t): t for t in pipeline.transforms}
        self.assertAlmostEqual(t_types[RandomCrop].foreground_prob, 0.45)
        self.assertEqual(t_types[RandomScaling3D].scale_range, (0.9, 1.1))
        self.assertAlmostEqual(t_types[GammaTransform].prob, 0.25)
        self.assertEqual(t_types[GaussianBlur3D].sigma_range, (0.4, 0.8))

    def test_visualizer_ast_snippets_and_empirical_catalog(self):
        snippets = collect_code_snippets()
        self.assertEqual(set(snippets.keys()), set(CODE_SNIPPET_SPECS.keys()))
        for key, item in snippets.items():
            self.assertFalse(item["code"].startswith("# Symbols not found"), f"Missing symbol for {key}: {item}")
            self.assertFalse(item["code"].startswith("# File not found"), f"Missing file for {key}")
            self.assertGreater(item["end_line"], item["start_line"] - 1)

        runs = load_empirical_runs()
        self.assertIn("o6s4pex3", runs)
        self.assertIn("9tvq8qld", runs)
        self.assertIn("class_breakdown_final", runs["o6s4pex3"])

    def test_compare_runs_dynamic_class_discovery(self):
        payload = {
            "target_recent": {
                "name": "binary_run",
                "id": "r1",
                "state": "finished",
                "config": {"method": {"name": "mean_teacher"}},
                "summary": {
                    "val/best_mDice": 0.72,
                    "best_iter": 1000,
                    "val/mDice": 0.70,
                    "val/dice_Hemorrhage": 0.72,
                },
                "history": [
                    {"iter": 500, "val/mDice": 0.61, "val/dice_Hemorrhage": 0.61},
                    {"iter": 1000, "val/mDice": 0.72, "val/dice_Hemorrhage": 0.72},
                ],
            },
            "target_prev": {"summary": {}, "history": []},
            "target_baseline": {"summary": {}, "history": []},
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_summary(payload)
            print_classes(payload)
        output = buf.getvalue()
        self.assertIn("Hemorrhage:", output)
        self.assertIn("0.7200", output)

    def test_kaggle_notebook_generators_and_presets(self):
        presets = load_notebook_presets()
        self.assertIn("mean_teacher", presets)
        self.assertIn("supervised", presets)
        self.assertIn("pick", presets)

        notebooks = [
            build_mean_teacher_notebook(setting="default"),
            build_mean_teacher_notebook(task="binary", setting="swdl"),
            build_mean_teacher_notebook(setting="vnet"),
            build_mean_teacher_notebook(setting="vnet_swdl"),
            build_supervised_notebook(setting="opt1_nnunet"),
            build_supervised_notebook(setting="vnet"),
            build_supervised_notebook(setting="vnet_legacy"),
            build_notebook_dict(),
        ]
        for nb in notebooks:
            self.assertEqual(nb["nbformat"], 4)
            self.assertGreater(len(nb["cells"]), 5)
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    lines = [
                        (line[: len(line) - len(line.lstrip())] + "pass\n")
                        if line.lstrip().startswith(("!", "%"))
                        else line
                        for line in cell["source"]
                    ]
                    ast.parse("".join(lines))

        mt_nb = build_mean_teacher_notebook()
        marker_matches = [
            c for c in mt_nb["cells"]
            if c["cell_type"] == "code"
            and "# Cấu hình Anti-Deadlock" in "".join(c["source"])
            and "base_args = [" in "".join(c["source"])
        ]
        self.assertEqual(len(marker_matches), 1)


if __name__ == "__main__":
    unittest.main()
