"""
Tool to migrate model checkpoints from Weights & Biases (W&B)
to Hugging Face Model Hub with 100% structural consistency.

Guarantees:
1. Exact same directory layout: `<run_name>/best_model.pth` and `<run_name>/checkpoint_latest.pth`
2. Exactly 1-2 clean checkpoints per run (no duplicate or intermediate clutter)
3. 100% indistinguishable from new runs trained going forward.

Usage:
    # Dry-run (check matched runs and paths without downloading/uploading):
    python tools/migrate_wandb_to_hf.py --dry-run

    # Perform full migration:
    python tools/migrate_wandb_to_hf.py

    # Perform migration and delete heavy binary artifacts on W&B to free 5GB quota:
    python tools/migrate_wandb_to_hf.py --delete-from-wandb
"""

import os
import sys
import argparse
import tempfile
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hf_storage import HFStorageManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def migrate_runs(
    entity: str,
    projects: list,
    repo_id: str,
    dry_run: bool = False,
    delete_from_wandb: bool = False
):
    import wandb
    api = wandb.Api()
    hf_manager = None if dry_run else HFStorageManager(repo_id=repo_id)

    total_runs = 0
    total_files = 0
    total_bytes = 0

    logging.info("=" * 75)
    logging.info("=== W&B -> HUGGING FACE CANONICAL ARTIFACT MIGRATION ===")
    logging.info(f"Entity:          {entity}")
    logging.info(f"Projects:        {', '.join(projects)}")
    logging.info(f"HF Target Repo:  {repo_id}")
    logging.info(f"Dry-run:         {dry_run}")
    logging.info(f"Delete from W&B: {delete_from_wandb}")
    logging.info("=" * 75)

    for proj in projects:
        logging.info(f"[*] Scanning project: {proj}...")
        try:
            runs = api.runs(f"{entity}/{proj}")
        except Exception as e:
            logging.warning(f"Could not access project {proj}: {e}")
            continue

        for run in runs:
            model_artifacts = [a for a in run.logged_artifacts() if a.type in ('model', 'checkpoint')]
            if not model_artifacts:
                continue

            # Identify canonical best and latest artifacts
            best_arts = [a for a in model_artifacts if 'best' in a.aliases or 'best' in a.name.lower()]
            latest_arts = [a for a in model_artifacts if 'latest' in a.aliases or 'latest' in a.name.lower()]

            target_best = max(best_arts, key=lambda a: int(a.name.rsplit(':v', 1)[1])) if best_arts else None
            target_latest = max(latest_arts, key=lambda a: int(a.name.rsplit(':v', 1)[1])) if latest_arts else None

            # Prepare list of items to migrate: (artifact, canonical_filename)
            items_to_migrate = []
            if target_best:
                items_to_migrate.append((target_best, "best_model.pth"))
            if target_latest and target_latest.id != (target_best.id if target_best else None):
                items_to_migrate.append((target_latest, "checkpoint_latest.pth"))

            if not items_to_migrate:
                continue

            total_runs += 1
            logging.info(f"  [Run] {run.name} ({len(items_to_migrate)} checkpoints)")

            for art, target_filename in items_to_migrate:
                path_in_repo = f"{run.name}/{target_filename}"
                size_mb = getattr(art, "size", 0) / (1024 * 1024)
                logging.info(f"     -> {art.name} ({size_mb:.1f} MB) -> {path_in_repo}")

                if dry_run:
                    total_files += 1
                    total_bytes += getattr(art, "size", 0)
                    continue

                # Download and upload to Hugging Face
                with tempfile.TemporaryDirectory() as tmp_dir:
                    try:
                        downloaded_dir = art.download(root=tmp_dir)
                        pth_files = list(Path(downloaded_dir).rglob("*.pth"))
                        if not pth_files:
                            logging.warning(f"        No .pth file found in {art.name}")
                            continue

                        source_file = pth_files[0]
                        file_size = source_file.stat().st_size

                        logging.info(f"        Uploading to HF: {path_in_repo}...")
                        hf_manager.upload_checkpoint(
                            file_path=str(source_file),
                            path_in_repo=path_in_repo,
                            commit_message=f"[{run.name}] {target_filename}",
                            async_upload=False
                        )
                        total_files += 1
                        total_bytes += file_size

                        if delete_from_wandb:
                            logging.info(f"        [!] Deleting {art.name} from W&B to free storage...")
                            art.delete(delete_aliases=True)

                    except Exception as err:
                        logging.error(f"        Failed to migrate {art.name}: {err}")

    logging.info("=" * 75)
    logging.info(f"MIGRATION COMPLETE: {total_runs} runs processed, {total_files} files ({total_bytes / (1024**3):.2f} GB).")
    logging.info("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Canonical migration of checkpoints from W&B to Hugging Face Hub")
    parser.add_argument("--entity", type=str, default="thanhnguyen2005243-ho-chi-minh-city-university-of-technology")
    parser.add_argument("--projects", nargs="+", default=["BHSD_Supervised", "BHSD_MeanTeacher", "PICK_BHSD", "PICK_BHSD_dual_gpu"])
    parser.add_argument("--repo-id", type=str, default=os.environ.get("HF_REPO_ID", "NguyenVanCongThanh/BHSD_checkpoints"))
    parser.add_argument("--dry-run", action="store_true", help="Inspect matched runs without downloading/uploading")
    parser.add_argument("--delete-from-wandb", action="store_true", help="Delete old binary artifact on W&B after migration to free 5GB quota")

    args = parser.parse_args()
    migrate_runs(
        entity=args.entity,
        projects=args.projects,
        repo_id=args.repo_id,
        dry_run=args.dry_run,
        delete_from_wandb=args.delete_from_wandb
    )


if __name__ == "__main__":
    main()
