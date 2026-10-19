"""
Comprehensive Verification Tests for MedSSL-BHSD Benchmark Framework.
Tests Registries (DATASETS, MODELS, METHODS), Config Compositions, Optimizer Factories,
and Trainer Instantiations across multiple configurations.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.registry import MODELS, METHODS, DATASETS
from core.config_parser import load_config
from core.optimizers import build_optimizer, calculate_lr
import data
import models
import methods


class TestBenchmarkFramework(unittest.TestCase):

    def test_registry_population(self):
        """Verify all essential components are registered."""
        self.assertIn("bhsd", DATASETS.list_keys())
        self.assertIn("generic_h5", DATASETS.list_keys())
        self.assertIn("acdc", DATASETS.list_keys())

        self.assertIn("vnet", MODELS.list_keys())
        self.assertIn("pick_vnet", MODELS.list_keys())
        self.assertIn("unet3d", MODELS.list_keys())
        self.assertIn("unetr", MODELS.list_keys())

        self.assertIn("supervised", METHODS.list_keys())
        self.assertIn("mean_teacher", METHODS.list_keys())
        self.assertIn("pick", METHODS.list_keys())
        self.assertIn("cps", METHODS.list_keys())
        self.assertIn("bcp", METHODS.list_keys())

    def test_config_composition_multiclass(self):
        """Test multi-tier config merging for BHSD Multiclass."""
        cfg = load_config(
            dataset_config="bhsd_multiclass",
            model_config="unet3d",
            method_config="mean_teacher",
            cli_args=["--fold", "2", "--data.batch_size", "8"]
        )
        self.assertEqual(cfg.data.name, "bhsd")
        self.assertEqual(cfg.data.num_classes, 6)
        self.assertEqual(len(cfg.data.class_names), 6)
        self.assertEqual(cfg.data.batch_size, 8)
        self.assertEqual(cfg.experiment.fold, 2)
        self.assertEqual(cfg.model.name, "unet3d")
        self.assertEqual(cfg.method.name, "mean_teacher")

    def test_config_composition_binary(self):
        """Test multi-tier config merging for BHSD Binary."""
        cfg = load_config(
            dataset_config="bhsd_binary",
            model_config="vnet",
            method_config="supervised",
            cli_args=["--fold", "0"]
        )
        self.assertEqual(cfg.data.name, "bhsd")
        self.assertEqual(cfg.data.num_classes, 2)
        self.assertEqual(cfg.data.task, "binary")
        self.assertEqual(cfg.model.name, "vnet")
        self.assertEqual(cfg.method.name, "supervised")

    def test_model_building_and_shapes(self):
        """Verify building models with standard forward pass."""
        patch = torch.randn(1, 1, 32, 64, 64)
        
        # 1. Standard single VNet
        vnet = MODELS.build("vnet", in_channels=1, num_classes=6, has_dropout=False)
        out = vnet(patch)
        self.assertEqual(out.shape, (1, 6, 32, 64, 64))

        # 2. UNet3D
        unet = MODELS.build("unet3d", in_channels=1, num_classes=2, has_dropout=False)
        out = unet(patch)
        self.assertEqual(out.shape, (1, 2, 32, 64, 64))

        # 3. PICK composite model
        pick_vnet = MODELS.build("pick_vnet", in_channels=1, num_classes=6, has_dropout=False)
        out_main, out_seg, feat = pick_vnet(patch)
        self.assertEqual(out_seg.shape, (1, 6, 32, 64, 64))

    def test_optimizer_factory(self):
        """Verify building different optimizers and LR schedules."""
        toy_model = nn.Linear(10, 2)
        
        # SGD
        cfg_sgd = load_config(cli_args=["--method.optimizer", "sgd", "--method.base_lr", "0.02"])
        opt_sgd = build_optimizer(toy_model, cfg_sgd)
        self.assertIsInstance(opt_sgd, torch.optim.SGD)
        self.assertAlmostEqual(opt_sgd.param_groups[0]['lr'], 0.02)

        # AdamW
        cfg_adamw = load_config(cli_args=["--method.optimizer", "adamw", "--method.base_lr", "0.001"])
        opt_adamw = build_optimizer(toy_model, cfg_adamw)
        self.assertIsInstance(opt_adamw, torch.optim.AdamW)
        self.assertAlmostEqual(opt_adamw.param_groups[0]['lr'], 0.001)

        # Schedulers
        lr_poly = calculate_lr(iter_num=5000, max_iterations=10000, base_lr=0.01, scheduler_type='poly')
        self.assertLess(lr_poly, 0.01)
        self.assertGreater(lr_poly, 0.0)

        lr_warmup = calculate_lr(iter_num=50, max_iterations=1000, base_lr=0.01, warmup_iters=100)
        self.assertAlmostEqual(lr_warmup, 0.005)

    def test_all_trainers_instantiation(self):
        """Test building trainer instances for all SSL methods via registry."""
        for method_name in ["supervised", "mean_teacher", "cps", "bcp", "pick"]:
            cfg = load_config(
                dataset_config="bhsd_multiclass",
                model_config="vnet",
                method_config=method_name,
                cli_args=["--fold", "0"]
            )
            # Pick uses pick_vnet
            model_key = "pick_vnet" if method_name == "pick" else "vnet"
            model = MODELS.build(model_key, in_channels=1, num_classes=6, has_dropout=False)
            trainer = METHODS.build(method_name, model=model, cfg=cfg)
            self.assertEqual(trainer.num_classes, 6)
            self.assertEqual(trainer.fold, 0)

    def test_train_steps(self):
        """Test train_step forward and loss calculation on synthetic batch."""
        cfg = load_config(
            dataset_config="bhsd_multiclass",
            model_config="vnet",
            method_config="supervised",
            cli_args=["--fold", "0"]
        )
        model = MODELS.build("vnet", in_channels=1, num_classes=6, has_dropout=False)
        trainer = METHODS.build("supervised", model=model, cfg=cfg)

        synthetic_batch = {
            'image': torch.randn(4, 1, 16, 32, 32),
            'label': torch.randint(0, 6, (4, 16, 32, 32), dtype=torch.long),
            'case': 'test_case'
        }
        loss, loss_dict = trainer.train_step(synthetic_batch, iter_num=0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIn("total", loss_dict)
        self.assertIn("ce", loss_dict)
        self.assertIn("dice", loss_dict)


if __name__ == "__main__":
    unittest.main()

