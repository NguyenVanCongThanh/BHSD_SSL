"""
Tool to purge heavy model and checkpoint artifacts from Weights & Biases (W&B)
to free up cloud storage quota, while keeping all run metrics, charts,
and evaluation tables intact on the W&B dashboard.
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def purge_wandb_models(
    entity: str = "thanhnguyen2005243-ho-chi-minh-city-university-of-technology",
    projects: list = None
):
    import wandb
    api = wandb.Api()

    if projects is None:
        projects = ["BHSD_Supervised", "BHSD_MeanTeacher", "PICK_BHSD", "PICK_BHSD_dual_gpu"]

    logging.info("=" * 75)
    logging.info("=== PURGING HEAVY MODEL ARTIFACTS ON WANDB ===")
    logging.info(f"Entity:   {entity}")
    logging.info(f"Projects: {', '.join(projects)}")
    logging.info("=" * 75)

    total_deleted = 0
    total_bytes = 0

    for proj in projects:
        logging.info(f"[*] Scanning project: {proj}...")
        try:
            artifact_types = api.artifact_types(f"{entity}/{proj}")
        except Exception as e:
            logging.warning(f"Could not access project {proj}: {e}")
            continue

        for atype in artifact_types:
            # Only target heavy weights, keep run_table and other metadata intact!
            if atype.name not in ("model", "checkpoint"):
                continue

            for coll in atype.collections():
                for art in list(coll.artifacts()):
                    size_mb = getattr(art, "size", 0) / (1024 * 1024)
                    size_bytes = getattr(art, "size", 0)
                    try:
                        logging.info(f"  -> Deleting {art.name} ({size_mb:.1f} MB)...")
                        art.delete(delete_aliases=True)
                        total_deleted += 1
                        total_bytes += size_bytes
                    except Exception as err:
                        logging.warning(f"     Failed to delete {art.name}: {err}")

    logging.info("=" * 75)
    logging.info(f"PURGE FINISHED: Deleted {total_deleted} heavy artifacts (~{total_bytes / (1024**3):.2f} GB freed).")
    logging.info("=" * 75)


if __name__ == "__main__":
    purge_wandb_models()

