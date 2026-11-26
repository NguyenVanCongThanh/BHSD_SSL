"""Thin backward-compatible wrapper around tools.kaggle.render_notebook for Supervised notebooks."""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kaggle.render_notebook import NotebookRenderOptions, render_training_notebook


def build_supervised_notebook(
    task: str = "multiclass",
    labelnum: int = 154,
    max_samples: int = 154,
    labeled_sampling: str = "first",
    setting: str = "opt1_nnunet",
):
    options = NotebookRenderOptions.from_env(
        task=task,
        setting=setting,
        labelnum=labelnum,
        max_samples=max_samples,
        labeled_sampling=labeled_sampling,
    )
    return render_training_notebook("supervised", options=options)


def main():
    task = os.environ.get("BHSD_TASK", "multiclass").strip().lower()
    if task not in ("multiclass", "binary"):
        task = "multiclass"
    labelnum = int(os.environ.get("BHSD_LABELNUM", "154"))
    max_samples = int(os.environ.get("BHSD_MAX_SAMPLES", "154"))
    labeled_sampling = os.environ.get("BHSD_LABELED_SAMPLING", "first").strip().lower()
    if labeled_sampling not in ("first", "stratified"):
        labeled_sampling = "first"

    setting_targets = {
        "opt1_nnunet": "scripts/kaggle_bhsd_supervised_dual_gpu.ipynb",
        "vnet": "scripts/kaggle_bhsd_supervised_vnet_dual_gpu.ipynb",
        "vnet_v2": "scripts/kaggle_bhsd_supervised_vnet_v2_dual_gpu.ipynb",
        "vnet7_plain": "scripts/kaggle_bhsd_supervised_vnet7_plain_dual_gpu.ipynb",
        "vnet5_plain": "scripts/kaggle_bhsd_supervised_vnet5_plain_dual_gpu.ipynb",
        "vnet5_res": "scripts/kaggle_bhsd_supervised_vnet5_res_dual_gpu.ipynb",
        "unet3d": "scripts/kaggle_bhsd_supervised_unet3d_dual_gpu.ipynb",
        "nnunet_v2_auto": "scripts/kaggle_bhsd_supervised_nnunet_v2_auto_dual_gpu.ipynb",
    }
    setting_env = os.environ.get("BHSD_SETTING", "").strip().lower()
    if setting_env and setting_env != "default" and setting_env in setting_targets:
        targets = [(setting_env, setting_targets[setting_env])]
    else:
        targets = list(setting_targets.items())

    for setting, target in targets:
        nb = build_supervised_notebook(
            task=task,
            labelnum=labelnum,
            max_samples=max_samples,
            labeled_sampling=labeled_sampling,
            setting=setting,
        )
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Generated {target} (setting={setting}, task={task})")


if __name__ == "__main__":
    main()
