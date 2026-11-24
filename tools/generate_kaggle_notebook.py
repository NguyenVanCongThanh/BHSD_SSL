"""Thin backward-compatible wrapper around tools.kaggle.render_notebook for PICK notebooks."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kaggle.render_notebook import NotebookRenderOptions, render_training_notebook


def build_notebook_dict(options: NotebookRenderOptions = None):
    opts = options if options is not None else NotebookRenderOptions.from_env()
    return render_training_notebook("pick", options=opts)


def main():
    nb = build_notebook_dict()
    target = "scripts/kaggle_bhsd_pick_dual_gpu.ipynb"
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Updated {target} successfully.")


if __name__ == "__main__":
    main()
