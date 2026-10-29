"""Storage cleanup must never delete a pending upload or another run's data."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from methods.base_method import BaseMethod
from utils.wandb_manager import WandbManager


def manager():
    obj = WandbManager.__new__(WandbManager)
    obj.cfg = SimpleNamespace(logging=SimpleNamespace(artifact_cache_size='1GB'))
    obj.enabled = obj.save_artifacts = True
    obj.mode = 'online'
    obj.run = MagicMock()
    obj.run_id, obj.entity, obj.project = 'mine', 'team', 'project'
    obj.artifact_keep_versions = 2
    obj.cache_cleanup_interval = 5
    obj._confirmed_uploads = 0
    return obj


def version(n, owner='mine', aliases=()):
    obj = MagicMock()
    obj.id, obj.version, obj.aliases = str(n), f'v{n}', list(aliases)
    obj.logged_by.return_value = SimpleNamespace(id=owner)
    return obj


def check_prunes_only_old_owned_versions_after_success():
    obj = manager()
    versions = [version(0), version(1, 'other'), version(2), version(3), version(4)]
    with patch('wandb.Api') as api:
        api.return_value.artifacts.return_value = versions
        obj._maintain_artifact_storage(versions[-1], 'checkpoint', 'checkpoint')
    for v in (versions[0], versions[2]):
        v.delete.assert_called_once_with(delete_aliases=True)
    for v in (versions[1], versions[3], versions[4]):
        v.delete.assert_not_called()


def check_stale_listing_does_not_prune():
    obj = manager()
    versions = [version(n) for n in range(3)]
    with patch('wandb.Api') as api:
        api.return_value.artifacts.return_value = versions
        obj._maintain_artifact_storage(version(4), 'checkpoint', 'checkpoint')
    for v in versions:
        v.delete.assert_not_called()


def check_failed_upload_confirmation_skips_cleanup(tmp_path):
    obj = manager()
    checkpoint = tmp_path / 'latest.pth'
    checkpoint.write_bytes(b'checkpoint')
    with patch('wandb.Artifact') as artifact, patch.object(obj, '_maintain_artifact_storage') as cleanup:
        artifact.return_value.wait.side_effect = TimeoutError('upload pending')
        assert obj.log_artifact(str(checkpoint), 'checkpoint')
        cleanup.assert_not_called()


def check_offline_does_not_wait_or_delete(tmp_path):
    obj = manager()
    obj.mode = 'offline'
    checkpoint = tmp_path / 'latest.pth'
    checkpoint.write_bytes(b'checkpoint')
    with patch('wandb.Artifact') as artifact, patch.object(obj, '_maintain_artifact_storage') as cleanup:
        assert obj.log_artifact(str(checkpoint), 'checkpoint')
        artifact.return_value.wait.assert_not_called()
        cleanup.assert_not_called()


def check_cache_cleanup_runs_periodically():
    obj = manager()
    obj.artifact_keep_versions = 0
    with patch.object(obj, '_cleanup_artifact_cache') as cleanup:
        for _ in range(5):
            obj._maintain_artifact_storage(None, 'checkpoint', 'checkpoint')
        cleanup.assert_called_once()


def check_resume_policy_selects_cloud_checkpoint():
    trainer = SimpleNamespace(cfg=SimpleNamespace(
        experiment=SimpleNamespace(resume_policy='latest'),
        logging=SimpleNamespace(artifact_interval=5000, epoch_artifact_interval=10)))
    assert BaseMethod.cloud_latest_due(trainer, 5000)
    assert not BaseMethod.cloud_latest_due(trainer, 500)
    assert not BaseMethod.cloud_epoch_due(trainer, 10)
    trainer.cfg.experiment.resume_policy = 'epoch_boundary'
    assert not BaseMethod.cloud_latest_due(trainer, 5000)
    assert not BaseMethod.cloud_epoch_due(trainer, 1)
    assert BaseMethod.cloud_epoch_due(trainer, 10)


class TestArtifactRetention(unittest.TestCase):
    def test_protected_alias_and_cleanup_failure(self):
        obj = manager()
        versions = [version(0, aliases=['best']), version(1), version(2), version(3)]
        with patch('wandb.Api') as api:
            api.return_value.artifacts.return_value = versions
            versions[1].delete.side_effect = RuntimeError('permission denied')
            obj._maintain_artifact_storage(versions[-1], 'checkpoint', 'checkpoint')
        versions[0].delete.assert_not_called()
        versions[1].delete.assert_called_once()
        self.assertEqual(obj._confirmed_uploads, 1)

    def test_retention(self):
        check_prunes_only_old_owned_versions_after_success()

    def test_stale_listing(self):
        check_stale_listing_does_not_prune()

    def test_pending_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            check_failed_upload_confirmation_skips_cleanup(Path(directory))

    def test_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            check_offline_does_not_wait_or_delete(Path(directory))

    def test_cache(self):
        check_cache_cleanup_runs_periodically()

    def test_policy(self):
        check_resume_policy_selects_cloud_checkpoint()
