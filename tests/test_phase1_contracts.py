"""Phase 1 critical correctness, crash-fix, telemetry, ramp-up, native-space, and cloud-default contract tests."""
import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from core.config_parser import Config, load_config, validate_config
from core.model_builder import model_spec_from_config
from methods.base_method import BaseMethod


class _TinySegNet(nn.Module):
    CAPABILITIES = {"segmentation"}

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class _TinyMultiScaleNet(nn.Module):
    CAPABILITIES = {"segmentation", "deep_supervision"}

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.deep_supervision = True
        self.conv1 = nn.Conv3d(in_channels, num_classes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, num_classes, kernel_size=3, stride=2, padding=1)

    def forward(self, x, return_all_scales=False):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        if self.training or return_all_scales:
            return [out1, out2]
        return out1


def _make_batch(batch_size=4, num_classes=2, shape=(8, 16, 16)):
    return {
        "image": torch.randn(batch_size, 1, *shape),
        "label": torch.randint(0, num_classes, (batch_size, *shape), dtype=torch.long),
    }


def _build_trainer(cls, cfg, model=None, batch_size=4, num_classes=2, shape=(8, 16, 16)):
    def loaders(trainer):
        trainer.trainloader = [_make_batch(batch_size, num_classes, shape)] * 10
        trainer.train_sampler = None
        trainer.batch_sampler = None

    def logging_setup(trainer):
        trainer.writer = MagicMock()
        trainer.wandb_manager = SimpleNamespace(
            enabled=False,
            save_artifacts=False,
            update_summary=MagicMock(),
            log_metrics=MagicMock(),
            restore_artifact=MagicMock(return_value=False),
            finish=MagicMock(),
            identity=lambda: {"id": "test_run"},
        )
        trainer.use_wandb = False

    with patch.object(BaseMethod, "setup_dataloaders", loaders), \
         patch.object(BaseMethod, "setup_logging", logging_setup), \
         patch("torch.cuda.is_available", return_value=False):
        return cls(model or _TinySegNet(num_classes=num_classes), cfg)


class TestBCPContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = Config({
            "data": {"batch_size": 4, "labeled_bs": 2, "task": "binary", "num_classes": 2, "patch_size": [8, 16, 16]},
            "model": {"name": "vnet"},
            "method": {"name": "bcp", "pre_max_iteration": 2, "self_max_iteration": 4, "ema_decay": 0.99},
            "experiment": {"exp_dir": str(self.root), "name": "bcp_test", "amp": False, "resume": False, "phase": "all"},
            "logging": {"val_interval": 2, "save_interval": 2},
        })

    def test_bcp_pretrain_transfer_loads_net_resets_optimizer_and_syncs_ema(self):
        from methods.bcp import BCPMethod
        from utils.checkpoint import save_checkpoint

        trainer = _build_trainer(BCPMethod, self.cfg)
        # Create a distinct pretrain checkpoint with known weights under standard 'net' key
        source_model = _TinySegNet(num_classes=2)
        with torch.no_grad():
            source_model.conv.weight.fill_(0.42)
            source_model.conv.bias.fill_(0.13)
        pre_best = os.path.join(trainer.pre_snapshot_path, "best_model.pth")
        save_checkpoint(pre_best, source_model, trainer.optimizer, iter_num=2, best_dice=0.9, phase="pre_train")

        # Populate optimizer state to verify setup_optimizer resets it
        old_opt = trainer.optimizer
        trainer.load_pretrain_weights_for_selftrain()

        self.assertIsNot(trainer.optimizer, old_opt, "Stage 2 must reinitialize optimizer")
        self.assertTrue(torch.allclose(trainer.model.conv.weight, torch.full_like(trainer.model.conv.weight, 0.42)))
        self.assertTrue(torch.allclose(trainer.ema_model.conv.weight, torch.full_like(trainer.ema_model.conv.weight, 0.42)))

    def test_bcp_ema_updates_only_after_optimizer_step_and_persists_in_checkpoint(self):
        from methods.bcp import BCPMethod

        trainer = _build_trainer(BCPMethod, self.cfg)
        trainer._active_phase = "self_train"
        # Perturb student away from EMA
        with torch.no_grad():
            trainer.model.conv.weight.fill_(1.0)
            trainer.ema_model.conv.weight.fill_(0.0)

        ema_before = trainer.ema_model.conv.weight.detach().clone()
        loss, loss_dict = trainer.selftrain_step(_make_batch(), iter_num=1)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("total", loss_dict)
        self.assertIn("supervised", loss_dict)
        self.assertIn("consistency", loss_dict)
        self.assertIn("weight", loss_dict)
        # EMA must NOT be mutated during selftrain_step (pre-backward)
        self.assertTrue(torch.equal(trainer.ema_model.conv.weight, ema_before))

        # EMA must update in after_optimizer_step
        trainer.after_optimizer_step(iter_num=1)
        self.assertFalse(torch.equal(trainer.ema_model.conv.weight, ema_before))

        # Checkpoint round-trip must preserve ema_net
        ckpt_path = os.path.join(trainer.self_snapshot_path, "checkpoint_latest.pth")
        saved_ema = trainer.ema_model.conv.weight.detach().clone()
        trainer.save_checkpoint(ckpt_path, iter_num=1, best_dice=0.5, phase="self_train")

        with torch.no_grad():
            trainer.ema_model.conv.weight.fill_(-9.0)
        trainer.load_checkpoint(ckpt_path)
        self.assertTrue(torch.allclose(trainer.ema_model.conv.weight, saved_ema))

    def test_bcp_fresh_run_directory_collision_detected(self):
        from utils.wandb_manager import WandbManager

        pre_dir = self.root / "bcp_test" / "fold_0" / "pre_train"
        pre_dir.mkdir(parents=True, exist_ok=True)
        (pre_dir / "best_model.pth").touch()
        with self.assertRaises(FileExistsError):
            WandbManager.check_fresh_run_directory(self.cfg, fold=0)

    def test_bcp_and_pick_cloud_resume_uses_restore_artifact(self):
        from methods.bcp import BCPMethod
        from methods.pick import PICKMethod

        for method_name, cls in (("bcp", BCPMethod), ("pick", PICKMethod)):
            with self.subTest(method=method_name):
                cfg = copy.deepcopy(self.cfg)
                cfg.experiment.name = f"{method_name}_resume_test"
                cfg.experiment.resume = True
                cfg.method.name = method_name
                trainer = _build_trainer(cls, cfg)
                trainer.wandb_manager.enabled = True
                # WandbManager does NOT have restore_checkpoint; ensure restore_artifact is called
                self.assertFalse(hasattr(trainer.wandb_manager, "restore_checkpoint"))
                with patch.object(trainer, "load_pretrain_weights_for_selftrain"), \
                     patch.object(trainer, "run_training_loop", return_value=(1, 0.5)):
                    trainer.run_self_train()
                trainer.wandb_manager.restore_artifact.assert_called_once()


class TestUAMTContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config({
            "data": {"batch_size": 4, "labeled_bs": 2, "task": "binary", "num_classes": 2, "patch_size": [8, 16, 16]},
            "model": {"name": "vnet"},
            "method": {
                "name": "uamt",
                "max_iteration": 100,
                "ema_decay": 0.99,
                "mc_t": 2,
                "consistency": 0.1,
                "consistency_rampup": 40.0,
                "consistency_rampup_unit": "epoch",
            },
            "experiment": {"exp_dir": self.tmp.name, "name": "uamt_test", "amp": False, "resume": False},
            "logging": {"val_interval": 10, "save_interval": 10},
        })

    def test_uamt_does_not_double_update_ema(self):
        from methods.uamt import UAMTMethod

        trainer = _build_trainer(UAMTMethod, self.cfg)
        with torch.no_grad():
            trainer.model.conv.weight.fill_(1.0)
            trainer.ema_model.conv.weight.fill_(0.0)

        ema_before = trainer.ema_model.conv.weight.detach().clone()
        loss, loss_dict = trainer.train_step(_make_batch(), iter_num=10)
        self.assertTrue(torch.isfinite(loss))
        # Canonical telemetry keys
        for key in ("total", "supervised", "consistency", "weight"):
            self.assertIn(key, loss_dict)
        # EMA must NOT be mutated inside train_step
        self.assertTrue(torch.equal(trainer.ema_model.conv.weight, ema_before))
        # Single update via inherited after_optimizer_step
        trainer.after_optimizer_step(iter_num=10)
        self.assertFalse(torch.equal(trainer.ema_model.conv.weight, ema_before))


class TestSWDLContracts(unittest.TestCase):
    def test_swdl_decoder1_uses_encoder_skip_connections_and_finest_first_ds(self):
        from models.backbones.swdl_net import SWDLNet, LaplacianUpsample3D

        torch.manual_seed(0)
        net = SWDLNet(in_channels=1, num_classes=2, n_filters=4)
        net.eval()
        x = torch.randn(1, 1, 16, 16, 16)
        feats = net.encoder(x)
        out_base, _ = net.decoder1(feats)
        # Perturb finest encoder skip x1; if skip connections are wired, out_seg1 MUST change
        feats_mod = [feats[0] + 5.0, feats[1], feats[2], feats[3], feats[4]]
        out_mod, _ = net.decoder1(feats_mod)
        self.assertFalse(
            torch.allclose(out_base, out_mod),
            "SWDLDecoder(f1=None) must use encoder skip connections x1..x4",
        )

        # Verify SWDLSideConv outputs are ordered finest -> coarsest [out1..out5]
        net.train()
        out1, out2, masks, deep_outs = net(x)
        self.assertEqual(len(deep_outs), 5)
        # Check sideconv index 0 comes from side1 (finest, x8_up) by zeroing side1 weights
        with torch.no_grad():
            net.sideconv.side1.weight.zero_()
            net.sideconv.side1.bias.zero_()
        _, _, _, deep_outs_zeroed = net(x)
        self.assertTrue(
            torch.allclose(deep_outs_zeroed[0], torch.zeros_like(deep_outs_zeroed[0])),
            "deep_outs[0] must correspond to finest scale side1 so ds_weights[0]=0.8 weights finest scale",
        )

        # Verify LaplacianUpsample3D works on min_dim == 128 without kernel reshape crash
        lap = LaplacianUpsample3D(scale_factor=2.0, mu=1.5)
        self.assertEqual(lap.get_kernel_size(128) % 2, 1)

    def test_swdl_method_canonical_keys(self):
        from methods.swdl import SWDLMethod
        from models.backbones.swdl_net import SWDLNet

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config({
                "data": {"batch_size": 4, "labeled_bs": 2, "task": "binary", "num_classes": 2, "patch_size": [16, 16, 16]},
                "model": {"name": "swdl_net"},
                "method": {"name": "swdl", "feedback_steps": 2, "consistency_rampup": 40.0, "consistency_rampup_unit": "epoch"},
                "experiment": {"exp_dir": tmp, "name": "swdl_test", "amp": False, "resume": False},
                "logging": {"val_interval": 10, "save_interval": 10},
            })
            model = SWDLNet(in_channels=1, num_classes=2, n_filters=4)
            trainer = _build_trainer(SWDLMethod, cfg, model=model, shape=(16, 16, 16))
            loss, logs = trainer.train_step(_make_batch(shape=(16, 16, 16)), iter_num=1)
            self.assertTrue(torch.isfinite(loss))
            for key in ("total", "supervised", "consistency", "weight"):
                self.assertIn(key, logs)


class TestURPCAndRampupContracts(unittest.TestCase):
    def test_urpc_rejects_single_scale_output_and_5stage_vnet(self):
        from methods.urpc import URPCMethod

        # Static/config check: 5-stage vnet or deep_supervision=False must be rejected for urpc
        with self.assertRaises(ValueError):
            model_spec_from_config(
                {"model": {"name": "vnet", "num_stages": 5}, "method": {"name": "urpc"}},
                num_classes=2,
            )
        with self.assertRaises(ValueError):
            model_spec_from_config(
                {"model": {"name": "nnunet3d", "deep_supervision": False}, "method": {"name": "urpc"}},
                num_classes=2,
            )

        # Runtime check: URPCMethod.train_step raises ValueError if backbone returns 0 aux heads
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config({
                "data": {"batch_size": 4, "labeled_bs": 2, "task": "binary", "num_classes": 2, "patch_size": [8, 16, 16]},
                "model": {"name": "nnunet3d", "deep_supervision": True},
                "method": {"name": "urpc", "consistency_rampup": 40.0, "consistency_rampup_unit": "epoch"},
                "experiment": {"exp_dir": tmp, "name": "urpc_test", "amp": False, "resume": False},
                "logging": {"val_interval": 10, "save_interval": 10},
            })
            trainer_bad = _build_trainer(URPCMethod, cfg, model=_TinySegNet(num_classes=2))
            with self.assertRaises(ValueError):
                trainer_bad.train_step(_make_batch(), iter_num=1)

            trainer_ok = _build_trainer(URPCMethod, cfg, model=_TinyMultiScaleNet(num_classes=2))
            loss, logs = trainer_ok.train_step(_make_batch(), iter_num=1)
            self.assertTrue(torch.isfinite(loss))
            for key in ("total", "supervised", "consistency", "weight"):
                self.assertIn(key, logs)

    def test_consistency_rampup_unit_validation(self):
        validate_config({"method": {"consistency_rampup_unit": "iter"}})
        validate_config({"method": {"consistency_rampup_unit": "epoch"}})
        validate_config({"method": {"consistency_rampup_unit": None}})
        with self.assertRaises(ValueError):
            validate_config({"method": {"consistency_rampup_unit": "steps"}})


class TestNativeSpaceAndResumeAndCloudContracts(unittest.TestCase):
    def test_native_space_nifti_axis_order_and_strict_metadata(self):
        import nibabel as nib
        import SimpleITK as sitk
        from evaluation.native_space import (
            load_raw_nifti_ground_truth,
            restore_volume_to_native_space,
        )

        # Missing metadata must raise ValueError instead of silently returning un-restored volume
        with self.assertRaises(ValueError):
            restore_volume_to_native_space(np.zeros((4, 8, 8), dtype=np.uint8), {})

        # Round-trip NIfTI axis order: (Z, Y, X) saved via SimpleITK must load as identical (Z, Y, X)
        vol_zyx = np.zeros((6, 10, 12), dtype=np.uint8)
        vol_zyx[1:3, 2:5, 3:9] = 2
        with tempfile.TemporaryDirectory() as tmp:
            nii_path = os.path.join(tmp, "case_01.nii.gz")
            sitk_img = sitk.GetImageFromArray(vol_zyx)
            sitk_img.SetSpacing((0.5, 0.5, 5.0))
            sitk.WriteImage(sitk_img, nii_path)

            loaded = load_raw_nifti_ground_truth("case_01", tmp)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, vol_zyx.shape)
            self.assertTrue(np.array_equal(loaded, vol_zyx))

    def test_prepare_resume_supports_lefed(self):
        from tools.kaggle.resume import prepare_resume

        config = {
            "experiment": {"name": "lefed_exp", "fold": 0, "task": "binary", "num_classes": 2},
            "method": {"name": "lefed", "max_iteration": 12000},
            "data": {
                "name": "bhsd", "task": "binary", "num_classes": 2,
                "class_names": ["Background", "Hemorrhage"], "root_path": "old", "batch_size": 4,
            },
            "logging": {},
            "model": {"name": "lefed_net"},
            "resolved": {"batch_layout": {"world_size": 2}},
        }

        class _Artifact:
            name = "lefed_exp_fold0_best:v1"
            metadata = {"iter_num": 4000}

            def download(self, root):
                Path(root).mkdir(parents=True)
                (Path(root) / "best_model.pth").write_bytes(b"ckpt")
                return root

        run = SimpleNamespace(config=config, state="finished", logged_artifacts=lambda: [_Artifact()])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules):
            sys.modules["wandb"] = SimpleNamespace(Api=lambda: SimpleNamespace(run=lambda p: run))
            args, out_cfg = prepare_resume("my_entity/BHSD_LeFeD/abc123", "best", "/data", tmp, 2)
            self.assertIn("--resume", args)
            self.assertEqual(out_cfg["method"]["name"], "lefed")

    def test_no_hardcoded_personal_cloud_ids_in_base_config_or_hf_storage(self):
        base_yaml = (Path(__file__).resolve().parents[1] / "configs" / "base.yaml").read_text(encoding="utf-8")
        self.assertNotIn("NguyenVanCongThanh", base_yaml)
        hf_src = (Path(__file__).resolve().parents[1] / "utils" / "hf_storage.py").read_text(encoding="utf-8")
        self.assertNotIn("NguyenVanCongThanh", hf_src)
        resume_src = (Path(__file__).resolve().parents[1] / "tools" / "kaggle" / "resume.py").read_text(encoding="utf-8")
        self.assertNotIn("thanhnguyen2005243", resume_src)

    def test_generate_kaggle_notebook_does_not_force_resume(self):
        from tools.generate_kaggle_notebook import build_notebook_dict

        nb = build_notebook_dict()
        train_cell = next(
            "".join(c["source"]) for c in nb["cells"]
            if c.get("cell_type") == "code" and "base_args = [" in "".join(c.get("source", []))
        )
        self.assertNotIn('"--resume"', train_cell)


if __name__ == "__main__":
    unittest.main()
