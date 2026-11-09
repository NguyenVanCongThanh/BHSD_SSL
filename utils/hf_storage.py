"""
Hugging Face Hub Model & Checkpoint Storage Manager for MedSSL-BHSD.
Provides free, unlimited, private cloud storage for model weights,
supporting non-blocking asynchronous uploads, auto-resume downloads,
and seamless integration with Weights & Biases experiment tracking.
"""

from typing import Dict, Any, Optional, List
import os
import logging
from pathlib import Path
from concurrent.futures import Future

# Auto-load .env file if present
def _load_env():
    root = Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

_load_env()


class HFStorageManager:
    """
    Manages model checkpoint uploading and downloading to/from Hugging Face Model Hub.
    """
    def __init__(
        self,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        private: bool = True
    ):
        self.token = token or os.environ.get("HF_TOKEN")
        self.repo_id = repo_id or os.environ.get("HF_REPO_ID", "NguyenVanCongThanh/BHSD_checkpoints")
        self.private = private
        self.api = None
        self.enabled = False
        self._pending_futures: List[Future] = []

        self._init_hf()

    def _init_hf(self):
        try:
            from huggingface_hub import HfApi
            # HfApi will auto-resolve token from self.token, HF_TOKEN env, or ~/.cache/huggingface/token
            self.api = HfApi(token=self.token)
            # Verify or create the repository
            try:
                self.api.repo_info(repo_id=self.repo_id, repo_type="model")
            except Exception:
                logging.info(f"[HF HUB] Creating repository '{self.repo_id}' (private={self.private})...")
                self.api.create_repo(repo_id=self.repo_id, repo_type="model", private=self.private, exist_ok=True)

            self.enabled = True
            logging.info(f"[HF HUB] Initialized Hugging Face Model Storage for repo: {self.repo_id}")
        except Exception as e:
            logging.warning(f"[HF HUB] Failed to initialize Hugging Face Hub: {e}. Checkpoint cloud sync will degrade.")
            self.enabled = False

    def upload_checkpoint(
        self,
        file_path: str,
        path_in_repo: str,
        commit_message: Optional[str] = None,
        async_upload: bool = True
    ) -> Optional[str]:
        """
        Upload a checkpoint file to the Hugging Face model repository.
        
        Args:
            file_path: Local path to the checkpoint file (.pth)
            path_in_repo: Path where the file should be saved in the repo
            commit_message: Optional commit message describing the checkpoint
            async_upload: If True, executes asynchronously without blocking training loop
            
        Returns:
            The public/private URL to the file on Hugging Face Hub, or None on failure.
        """
        if not self.enabled or self.api is None:
            return None

        if not os.path.exists(file_path):
            logging.warning(f"[HF HUB] Checkpoint file does not exist: {file_path}")
            return None

        commit_msg = commit_message or f"Upload {Path(file_path).name}"
        file_url = f"https://huggingface.co/{self.repo_id}/blob/main/{path_in_repo}"

        try:
            if async_upload:
                # Clean up completed futures to keep memory clean
                self._pending_futures = [f for f in self._pending_futures if not f.done()]
                future = self.api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=commit_msg,
                    run_as_future=True
                )
                self._pending_futures.append(future)
                logging.info(f"[HF HUB ASYNC] Non-blocking upload queued: {path_in_repo} -> {file_url}")
            else:
                self.api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=commit_msg
                )
                logging.info(f"[HF HUB] Direct upload completed: {path_in_repo} -> {file_url}")

            return file_url
        except Exception as e:
            logging.warning(f"[HF HUB] Failed to upload {file_path} to {path_in_repo}: {e}")
            return None

    def restore_checkpoint(
        self,
        filename_or_pattern: str,
        dest_dir: str,
        run_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Download a checkpoint file from Hugging Face Hub to local directory.
        Supports fuzzy and canonical resolution for automated resume.
        
        Args:
            filename_or_pattern: File name or pattern to restore (e.g. 'checkpoint_latest.pth' or 'fold0_latest')
            dest_dir: Destination directory where checkpoint should be placed
            run_name: Optional run name subfolder in repo
            
        Returns:
            Local file path if downloaded successfully, None otherwise.
        """
        if not self.enabled or self.api is None:
            return None

        try:
            import shutil
            from huggingface_hub import hf_hub_download
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="model")

            is_best = "best" in filename_or_pattern.lower()
            is_latest = "latest" in filename_or_pattern.lower()

            candidates = []
            
            # 1. Direct candidate matching with run_name
            if run_name:
                if is_best:
                    candidates.extend([f"{run_name}/best_model.pth", f"{run_name}/VNet_best_model.pth"])
                if is_latest:
                    candidates.extend([f"{run_name}/checkpoint_latest.pth", f"{run_name}/checkpoint_epoch_latest.pth"])
                candidates.append(f"{run_name}/{filename_or_pattern}")
                if not filename_or_pattern.endswith(".pth"):
                    candidates.append(f"{run_name}/{filename_or_pattern}.pth")

                # Match runs that start with or contain the run_name/prefix
                run_prefix = run_name.split("_202")[0] if "_202" in run_name else run_name
                for f in files:
                    if f.startswith(f"{run_name}/") or (len(run_prefix) > 5 and f.startswith(f"{run_prefix}")):
                        if is_best and "best" in f.lower():
                            candidates.append(f)
                        elif is_latest and "latest" in f.lower():
                            candidates.append(f)
            else:
                # 2. Explicit pattern matching across repo (only when run_name is not specified)
                for f in files:
                    if filename_or_pattern in f:
                        candidates.append(f)

            # Filter candidates that actually exist in repo
            matched_candidates = [c for c in candidates if c in files]

            if not matched_candidates:
                logging.warning(f"[HF HUB] No matching checkpoint found for '{filename_or_pattern}' (run={run_name}) in {self.repo_id}")
                return None

            target_in_repo = matched_candidates[0]
            os.makedirs(dest_dir, exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=target_in_repo,
                local_dir=dest_dir,
                token=self.token,
                repo_type="model"
            )

            # Ensure file is copied directly into dest_dir with its standard filename
            dest_filename = Path(target_in_repo).name
            if is_best and dest_filename not in ("best_model.pth", "VNet_best_model.pth"):
                dest_filename = "best_model.pth"
            elif is_latest and dest_filename not in ("checkpoint_latest.pth", "checkpoint_epoch_latest.pth"):
                dest_filename = "checkpoint_latest.pth"

            final_path = Path(dest_dir) / dest_filename
            if Path(downloaded_path).exists() and Path(downloaded_path).resolve() != final_path.resolve():
                shutil.copy2(downloaded_path, final_path)
                return str(final_path)

            logging.info(f"[HF HUB] Successfully restored checkpoint '{target_in_repo}' -> {downloaded_path}!")
            return str(downloaded_path)
        except Exception as e:
            logging.warning(f"[HF HUB] Failed to restore checkpoint '{filename_or_pattern}': {e}")
            return None

    def wait_for_uploads(self, timeout: Optional[float] = 120):
        """Wait for any active background uploads to complete."""
        if not self._pending_futures:
            return
        logging.info(f"[HF HUB] Waiting for {len(self._pending_futures)} pending uploads to finalize...")
        for future in list(self._pending_futures):
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logging.warning(f"[HF HUB] Background upload finished with error: {e}")
        self._pending_futures.clear()

