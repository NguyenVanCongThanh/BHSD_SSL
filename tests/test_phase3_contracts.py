"""Phase 3 contract tests: WandbManager deduplication, CheckpointManager, UNETR grad coverage, worker RNG seeding, and EDA guard."""
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from core.config_parser import Config


class TestPhase3Contracts(unittest.TestCase):
    def test_wandb_manager_unified_commit_artifact_hf_and_wandb_paths(self):
        from utils.wandb_manager import WandbManager

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules):
            ckpt = Path(tmp) / "best_model.pth"
            ckpt.write_bytes(b"weights")

            created_artifacts = []

            class _FakeArtifact:
                def __init__(self, name, type, description="", metadata=None):
                    self.name = name
                    self.type = type
                    self.description = description
                    self.metadata = metadata or {}
                    self.refs = []
                    self.files = []
                    self.id = "art-1"
                    self.waited = False
                    created_artifacts.append(self)

                def add_reference(self, url, checksum=False):
                    self.refs.append(url)

                def add_file(self, path, name=None):
                    self.files.append((path, name))

                def wait(self, timeout=120):
                    self.waited = True

            sys.modules["wandb"] = types.SimpleNamespace(Artifact=_FakeArtifact)

            mgr = WandbManager.__new__(WandbManager)
            mgr.enabled = True
            mgr.save_artifacts = True
            mgr.async_upload = False
            mgr._upload_queue = None
            mgr.mode = "online"
            mgr.run_name = "phase3_run"
            mgr.run = types.SimpleNamespace(log_artifact=MagicMock())
            mgr.artifact_backend = "huggingface"
            mgr.hf_manager = types.SimpleNamespace(
                upload_checkpoint=MagicMock(return_value="https://huggingface.co/org/repo/blob/main/phase3_run/best_model.pth")
            )
            mgr._confirmed_uploads = 0
            mgr.cache_cleanup_interval = 5
            mgr._cleanup_artifact_cache = MagicMock()
            mgr._maintain_artifact_storage = MagicMock()

            # 1) HF reference path (0 byte W&B quota)
            ok = mgr.log_artifact(str(ckpt), name="exp_best", artifact_type="model", aliases=["best"])
            self.assertTrue(ok)
            self.assertEqual(len(created_artifacts[-1].refs), 1)
            self.assertEqual(len(created_artifacts[-1].files), 0)

            # 2) Direct W&B file path + wait + storage maintenance
            mgr.artifact_backend = "wandb"
            ok2 = mgr._upload_and_prune_direct(str(ckpt), "best_model.pth", "exp_best", "model", "", ["best"], {})
            self.assertTrue(ok2)
            self.assertEqual(len(created_artifacts[-1].files), 1)
            self.assertTrue(created_artifacts[-1].waited)
            mgr._maintain_artifact_storage.assert_called_once()

    def test_checkpoint_manager_roundtrip_and_throttling(self):
        from training.checkpoint_manager import CheckpointManager

        cfg = Config({
            "experiment": {"resume_policy": "latest", "fold": 0, "task": "binary"},
            "logging": {"artifact_interval": 100, "epoch_artifact_interval": 5},
        })
        self.assertTrue(CheckpointManager.cloud_latest_due(cfg, 100, phase_max_iters=250))
        self.assertTrue(CheckpointManager.cloud_latest_due(cfg, 250, phase_max_iters=250))
        self.assertFalse(CheckpointManager.cloud_latest_due(cfg, 50, phase_max_iters=250))
        self.assertFalse(CheckpointManager.cloud_epoch_due(cfg, 5))

        cfg.experiment.resume_policy = "epoch_boundary"
        self.assertTrue(CheckpointManager.cloud_epoch_due(cfg, 5))
        self.assertFalse(CheckpointManager.cloud_epoch_due(cfg, 3))

        with tempfile.TemporaryDirectory() as tmp:
            model = nn.Conv3d(1, 2, 1)
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            ckpt_path = str(Path(tmp) / "ckpt.pth")
            wb = types.SimpleNamespace(enabled=False, save_artifacts=False, run_id=None, identity=lambda: {})

            CheckpointManager.save(
                path=ckpt_path,
                model=model,
                optimizer=opt,
                scaler=None,
                cfg=cfg,
                wandb_manager=wb,
                validation_options={},
                successful_updates=7,
                validation_details={"mDice": 0.75},
                iter_num=10,
                best_dice=0.75,
                phase="train",
            )
            res = CheckpointManager.load(
                path=ckpt_path,
                model=model,
                optimizer=opt,
                scaler=None,
                device=torch.device("cpu"),
                fold=0,
                task="binary",
                num_classes=2,
                dataset_name="bhsd",
            )
            self.assertEqual(res.iter_num, 10)
            self.assertAlmostEqual(res.best_dice, 0.75)
            self.assertEqual(res.successful_updates, 7)

    def test_unetr_has_no_unused_parameters(self):
        from models.backbones.unetr import UNETR

        model = UNETR(
            in_channels=1,
            num_classes=2,
            img_shape=(16, 16, 16),
            embed_dim=24,
            patch_size=16,
            num_heads=4,
        )
        self.assertFalse(hasattr(model.transformer, "encoder_norm"))
        x = torch.randn(1, 1, 16, 16, 16)
        out = model(x)
        out.sum().backward()
        unused = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(unused, [], f"UNETR has unused parameters: {unused}")

    def test_dataloader_worker_init_fn_is_deterministic_and_unique_per_worker(self):
        from data.dataloader_factory import make_worker_init_fn

        fn_r0 = make_worker_init_fn(base_seed=1337, rank=0)
        fn_r1 = make_worker_init_fn(base_seed=1337, rank=1)

        fn_r0(0)
        sample_r0_w0_a = (float(np.random.rand()), random.random(), float(torch.rand(1).item()))
        fn_r0(0)
        sample_r0_w0_b = (float(np.random.rand()), random.random(), float(torch.rand(1).item()))
        self.assertEqual(sample_r0_w0_a, sample_r0_w0_b)

        fn_r0(1)
        sample_r0_w1 = (float(np.random.rand()), random.random(), float(torch.rand(1).item()))
        fn_r1(0)
        sample_r1_w0 = (float(np.random.rand()), random.random(), float(torch.rand(1).item()))
        self.assertNotEqual(sample_r0_w0_a, sample_r0_w1)
        self.assertNotEqual(sample_r0_w0_a, sample_r1_w0)

    def test_eda_notebook_generator_guards_missing_bhsd_eda_vi(self):
        src = (Path(__file__).resolve().parents[1] / "tools" / "generate_bhsd_eda_notebook.py").read_text(encoding="utf-8")
        self.assertIn("except ImportError:", src)
        self.assertIn("from tools.bhsd_eda_vi import", src)


if __name__ == "__main__":
    unittest.main()
