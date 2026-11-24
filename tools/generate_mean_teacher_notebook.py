"""Thin backward-compatible wrapper around tools.kaggle.render_notebook for Mean Teacher notebooks."""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kaggle.render_notebook import NotebookRenderOptions, render_training_notebook


def build_mean_teacher_notebook(
    task: str = "multiclass",
    labelnum: int = 154,
    max_samples: int = 2200,
    labeled_sampling: str = "first",
    setting: str = "default",
):
    options = NotebookRenderOptions.from_env(
        task=task,
        setting=setting,
        labelnum=labelnum,
        max_samples=max_samples,
        labeled_sampling=labeled_sampling,
    )
    return render_training_notebook("mean_teacher", options=options)


def main():
    task = os.environ.get("BHSD_TASK", "multiclass").strip().lower()
    if task not in ("multiclass", "binary"):
        task = "multiclass"
    labelnum = int(os.environ.get("BHSD_LABELNUM", "154"))
    max_samples = int(os.environ.get("BHSD_MAX_SAMPLES", "2200"))
    labeled_sampling = os.environ.get("BHSD_LABELED_SAMPLING", "first").strip().lower()
    if labeled_sampling not in ("first", "stratified"):
        labeled_sampling = "first"
    setting = os.environ.get("BHSD_SETTING", "default").strip().lower() or "default"

    nb = build_mean_teacher_notebook(
        task=task,
        labelnum=labelnum,
        max_samples=max_samples,
        labeled_sampling=labeled_sampling,
        setting=setting,
    )
    target = "scripts/kaggle_bhsd_mean_teacher_dual_gpu.ipynb"
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Generated {target} (setting={setting}, task={task})")


if __name__ == "__main__":
    main()
