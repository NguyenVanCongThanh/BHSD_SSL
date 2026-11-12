import json
import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from core.config_parser import load_config
from data.constants import DEFAULT_FOREGROUND_OVERSAMPLE_PROB, resolve_sliding_window_strides
from data.dataloader_factory import build_bhsd_transforms
from data.transforms3d import GammaTransform, RandomScaling3D
from evaluation.sliding_window import sliding_window_inference_3d
from losses import ConsistencyLoss, DiceCELoss, MaskDiceLoss, TverskyLoss
from losses.deep_supervision import compute_ds_weights, downsample_target_for_ds
from methods.supervised import SupervisedMethod


class _DummyMultiScaleModel(nn.Module):
    CAPABILITIES = {"segmentation", "deep_supervision"}

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.head0 = nn.Conv3d(1, num_classes, kernel_size=1)
        self.head1 = nn.Conv3d(1, num_classes, kernel_size=1)
        self.head2 = nn.Conv3d(1, num_classes, kernel_size=1)

    def forward(self, x):
        main = self.head0(x)
        aux1 = self.head1(x[:, :, ::2, ::2, ::2])
        aux2 = self.head2(x[:, :, ::4, ::4, ::4])
        return [main, aux1, aux2]


class TestNNUNetAlignment(unittest.TestCase):
    def test_sliding_window_strides_50_percent_overlap(self):
        self.assertEqual(resolve_sliding_window_strides((28, 256, 256)), (14, 128))
        self.assertEqual(resolve_sliding_window_strides((32, 256, 256)), (16, 128))
        self.assertEqual(resolve_sliding_window_strides((16, 64, 64)), (8, 32))
        self.assertEqual(resolve_sliding_window_strides((28, 256, 256), stride_z=7, stride_xy=64), (7, 64))
        with self.assertRaises(ValueError):
            resolve_sliding_window_strides((28, 256, 256), step_size=0.0)
        with self.assertRaises(ValueError):
            resolve_sliding_window_strides((28, 256, 256), step_size=1.2)

        model = _DummyMultiScaleModel(num_classes=2).eval()
        vol = np.random.randn(20, 48, 48).astype(np.float32)
        pred = sliding_window_inference_3d(
            model=model,
            image=vol,
            num_classes=2,
            patch_size=(16, 32, 32),
            use_amp=False,
            device=torch.device("cpu"),
        )
        self.assertEqual(pred.shape, vol.shape)

    def test_batch_dice_differs_from_per_sample_dice_on_empty_fg(self):
        # Sample 0 has strong FG overlap; sample 1 has no FG in target or pred.
        logits = torch.full((2, 2, 4, 8, 8), -4.0)
        logits[:, 0] = 4.0
        logits[0, 1, 1:3, 2:6, 2:6] = 4.0
        logits[0, 0, 1:3, 2:6, 2:6] = -4.0

        targets = torch.zeros((2, 4, 8, 8), dtype=torch.long)
        targets[0, 1:3, 2:6, 2:6] = 1

        loss_sample = MaskDiceLoss(nclass=2, include_background=False, batch_dice=False)
        loss_batch = MaskDiceLoss(nclass=2, include_background=False, batch_dice=True)

        logits_s = logits.clone().requires_grad_(True)
        logits_b = logits.clone().requires_grad_(True)
        v_sample = loss_sample(logits_s, targets)
        v_batch = loss_batch(logits_b, targets)
        self.assertNotAlmostEqual(float(v_sample.item()), float(v_batch.item()), places=4)
        self.assertLess(float(v_batch.item()), float(v_sample.item()))
        v_batch.backward()
        self.assertTrue(torch.isfinite(logits_b.grad).all())

        dice_ce = DiceCELoss(nclass=2, include_background=False, batch_dice=True)
        tversky = TverskyLoss(nclass=2, include_background=False, batch_dice=True)
        cons = ConsistencyLoss(consistency_type="hybrid", num_classes=2, is_binary=True, batch_dice=True)
        self.assertTrue(torch.isfinite(dice_ce(logits, targets)))
        self.assertTrue(torch.isfinite(tversky(logits, targets)))
        self.assertTrue(torch.isfinite(cons(logits, logits.detach())))

    def test_deep_supervision_zero_lowest_and_native_downsample(self):
        weights = compute_ds_weights(4, zero_lowest_scale=True)
        self.assertEqual(len(weights), 4)
        self.assertEqual(weights[-1], 0.0)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)

        target = torch.zeros((2, 16, 32, 32), dtype=torch.long)
        target[:, 4:12, 8:24, 8:24] = 1
        down = downsample_target_for_ds(target, (8, 16, 16))
        self.assertEqual(down.shape, (2, 8, 16, 16))
        self.assertEqual(down.dtype, torch.long)
        self.assertEqual(int(down.max().item()), 1)

    def test_supervised_method_wires_batch_dice_and_ds_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = load_config(
                dataset_config="bhsd_binary",
                method_config="supervised",
                cli_args=[
                    "--experiment.exp_dir", tmpdir,
                    "--data.max_samples", "2",
                    "--data.labelnum", "2",
                    "--data.batch_size", "2",
                    "--data.labeled_bs", "2",
                    "--data.patch_size", "[16, 32, 32]",
                    "--logging.use_wandb", "false",
                    "--logging.use_hf_storage", "false",
                ],
            )
            # Prevent real filesystem dependency in setup_dataloaders by patching only the factory symbol
            from unittest.mock import patch
            dummy_bundle = type(
                "Bundle",
                (),
                {
                    "db_train": [1, 2],
                    "db_val": [1],
                    "trainloader": [1],
                    "train_sampler": None,
                    "batch_sampler": None,
                },
            )()
            with patch("methods.base_method.build_bhsd_dataloaders", return_value=dummy_bundle):
                method = SupervisedMethod(_DummyMultiScaleModel(num_classes=2), cfg)
                # Simulate DS model capability cleanly
                method.ds_enabled = True
                self.assertTrue(method.batch_dice)
                self.assertTrue(method.dice_loss.batch_dice)
                self.assertTrue(method.ds_zero_lowest_scale)
                self.assertTrue(method.ds_downsample_target)

                opts = method.validation_options()
                self.assertEqual((opts["stride_z"], opts["stride_xy"]), (8, 16))
                self.assertEqual(opts["step_size"], 0.5)

                batch = {
                    "image": torch.randn(2, 1, 16, 32, 32),
                    "label": torch.randint(0, 2, (2, 16, 32, 32)),
                }
                loss, logs = method.train_step(batch, iter_num=1)
                self.assertTrue(torch.isfinite(loss))
                self.assertIn("deep_supervision", logs)

    def test_nnunet_augmentation_pipeline_and_retain_stats(self):
        self.assertAlmostEqual(DEFAULT_FOREGROUND_OVERSAMPLE_PROB, 0.50, places=6)
        pipeline = build_bhsd_transforms(
            patch_size=(16, 32, 32),
            extended_aug=True,
            aug_preset="nnunet",
        )
        vol = np.random.randn(20, 40, 40).astype(np.float32) - 0.5
        lbl = np.zeros((20, 40, 40), dtype=np.int64)
        lbl[5:12, 10:25, 10:25] = 1
        out = pipeline({"image": vol.copy(), "label": lbl.copy()})
        self.assertEqual(tuple(out["image"].shape), (1, 16, 32, 32))
        self.assertEqual(tuple(out["label"].shape), (16, 32, 32))
        self.assertEqual(out["image"].dtype, torch.float32)
        self.assertEqual(out["label"].dtype, torch.int64)
        self.assertTrue(torch.isfinite(out["image"]).all())

        gamma = GammaTransform(gamma_range=(1.4, 1.4), prob=1.0, retain_stats=True)
        g_out = gamma({"image": vol.copy(), "label": lbl.copy()})["image"]
        self.assertAlmostEqual(float(g_out.mean()), float(vol.mean()), places=3)
        self.assertAlmostEqual(float(g_out.std()), float(vol.std()), places=3)

        scaler = RandomScaling3D(scale_range=(0.5, 0.5), prob=1.0, pad_to_min=True)
        neg_vol = np.full((12, 24, 24), -2.0, dtype=np.float32)
        neg_vol[4:8, 8:16, 8:16] = 1.0
        s_out = scaler({"image": neg_vol, "label": np.zeros((12, 24, 24), dtype=np.int64)})["image"]
        self.assertLessEqual(float(s_out[0, 0, 0]), -1.9)

    def test_experiment_configs_and_kaggle_presets_aligned(self):
        presets = [
            "configs/experiments/bhsd_supervised_vnet_opt1_fold0.yaml",
            "configs/experiments/bhsd_supervised_vnet_opt1_v2_fold0.yaml",
            "configs/experiments/bhsd_supervised_nnunet_opt1_fold0.yaml",
            "configs/experiments/bhsd_mt_nnunet_fold0.yaml",
            "configs/experiments/bhsd_pick_fold0.yaml",
        ]
        for path in presets:
            cfg = load_config(config_path=path)
            self.assertTrue(cfg.method.batch_dice, msg=path)
            self.assertTrue(cfg.method.ds_zero_lowest_scale, msg=path)
            self.assertTrue(cfg.method.ds_downsample_target, msg=path)
            self.assertAlmostEqual(float(cfg.method.momentum), 0.99, places=6, msg=path)
            self.assertTrue(bool(cfg.method.nesterov), msg=path)
            self.assertAlmostEqual(float(cfg.method.weight_decay), 3e-5, places=8, msg=path)
            self.assertEqual(str(cfg.data.aug_preset), "nnunet", msg=path)
            self.assertAlmostEqual(float(cfg.logging.val_step_size), 0.5, places=6, msg=path)
            self.assertIsNone(cfg.logging.val_stride_xy, msg=path)
            self.assertIsNone(cfg.logging.val_stride_z, msg=path)

        with open(os.path.join("configs", "kaggle", "notebook_presets.json"), "r", encoding="utf-8") as f:
            kp = json.load(f)
        for family in ("mean_teacher", "supervised"):
            for name, setting in kp[family]["settings"].items():
                pz, py, px = [int(x) for x in setting["patch_size_eval"]]
                self.assertEqual(setting["val_stride_z"], pz // 2, msg=f"{family}.{name}")
                self.assertEqual(setting["val_stride_xy"], min(py, px) // 2, msg=f"{family}.{name}")
        self.assertEqual(kp["pick"]["defaults"]["val_stride_xy"], 80)
        self.assertEqual(kp["pick"]["defaults"]["val_stride_z"], 16)


if __name__ == "__main__":
    unittest.main()

