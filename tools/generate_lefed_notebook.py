"""Thin backward-compatible wrapper around tools.kaggle.render_notebook for LeFeD notebooks."""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kaggle.render_notebook import NotebookRenderOptions, render_training_notebook


def build_lefed_notebook(
    task: str = "binary",
    labelnum: int = 8,
    max_samples: int = 154,
    labeled_sampling: str = "stratified",
    setting: str = "vnet5",
):
    options = NotebookRenderOptions.from_env(
        task=task,
        setting=setting,
        labelnum=labelnum,
        max_samples=max_samples,
        labeled_sampling=labeled_sampling,
    )
    return render_training_notebook("lefed", options=options)


def main():
    task = os.environ.get("BHSD_TASK", "binary").strip().lower()
    if task not in ("multiclass", "binary"):
        task = "binary"
    labelnum = int(os.environ.get("BHSD_LABELNUM", "8"))
    max_samples = int(os.environ.get("BHSD_MAX_SAMPLES", "154"))
    labeled_sampling = os.environ.get("BHSD_LABELED_SAMPLING", "stratified").strip().lower()
    if labeled_sampling not in ("first", "stratified"):
        labeled_sampling = "stratified"
    setting = os.environ.get("BHSD_SETTING", "vnet5").strip().lower() or "vnet5"
    if setting == "default":
        setting = "vnet5"

    nb = build_lefed_notebook(
        task=task,
        labelnum=labelnum,
        max_samples=max_samples,
        labeled_sampling=labeled_sampling,
        setting=setting,
    )
    target = "scripts/kaggle_bhsd_lefed_dual_gpu.ipynb"
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Generated {target} (setting={setting}, task={task})")


if __name__ == "__main__":
    main()
