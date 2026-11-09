"""Unit tests for Hugging Face Hub Storage and WandB Reference Artifacts integration."""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from pathlib import Path

from utils.hf_storage import HFStorageManager
from utils.wandb_manager import WandbManager


class TestHFStorage(unittest.TestCase):
    def test_hf_storage_manager_init(self):
        with patch("huggingface_hub.HfApi") as mock_api:
            mgr = HFStorageManager(repo_id="test/repo", token="dummy_token")
            self.assertTrue(mgr.enabled)
            self.assertEqual(mgr.repo_id, "test/repo")

    def test_hf_storage_upload_and_download(self):
        with patch("huggingface_hub.HfApi") as mock_api, patch("huggingface_hub.hf_hub_download") as mock_download:
            mock_api.return_value.upload_file.return_value = "https://huggingface.co/test/repo/commit/123"
            mock_download.return_value = "/tmp/downloaded.pth"
            mock_api.return_value.list_repo_files.return_value = ["run_1/best/best_model.pth"]

            mgr = HFStorageManager(repo_id="test/repo", token="dummy_token")
            with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
                url = mgr.upload_checkpoint(tmp.name, "run_1/best/best_model.pth", async_upload=False)
                self.assertIn("https://huggingface.co/test/repo/blob/main/run_1/best/best_model.pth", url)

            restored = mgr.restore_checkpoint("best_model.pth", "/tmp", run_name="run_1")
            self.assertEqual(restored, "/tmp/downloaded.pth")

    def test_wandb_manager_hf_reference_artifact(self):
        obj = WandbManager.__new__(WandbManager)
        obj.cfg = SimpleNamespace(
            logging=SimpleNamespace(
                artifact_cache_size='1GB',
                artifact_backend='huggingface',
                use_hf_storage=True,
                artifact_async_upload=False
            )
        )
        obj.enabled = obj.save_artifacts = True
        obj.mode = 'online'
        obj.run = MagicMock()
        obj.run_name = "test_run"
        obj.run_id, obj.entity, obj.project = 'mine', 'team', 'project'
        obj.artifact_keep_versions = 2
        obj.cache_cleanup_interval = 5
        obj._confirmed_uploads = 0
        obj.async_upload = False
        obj._upload_queue = None
        obj._upload_thread = None
        obj.artifact_backend = 'huggingface'
        obj.use_hf_storage = True

        mock_hf = MagicMock()
        mock_hf.enabled = True
        mock_hf.upload_checkpoint.return_value = "https://huggingface.co/test/repo/blob/main/test_run/best.pth"
        obj.hf_manager = mock_hf

        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp, patch("wandb.Artifact") as mock_art:
            logged = obj.log_artifact(tmp.name, "best_ckpt", aliases=["best"])
            self.assertTrue(logged)
            mock_hf.upload_checkpoint.assert_called_once()
            # Verify reference was added to WandB rather than add_file
            mock_art.return_value.add_reference.assert_called_once_with("https://huggingface.co/test/repo/blob/main/test_run/best.pth")
            mock_art.return_value.add_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()

