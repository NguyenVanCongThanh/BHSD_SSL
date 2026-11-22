import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kaggle.notebook_templates import (
    build_notebook_envelope,
    check_gpu_cell,
    clone_repo_cell,
    code_cell,
    load_notebook_presets,
    locate_dataset_cell,
    markdown_cell,
    pip_install_cell,
    profile_model_cell,
    resolve_method_setting,
    wandb_setup_cell,
)


def build_lefed_notebook(
    task: str = "binary",
    labelnum: int = 8,
    max_samples: int = 154,
    labeled_sampling: str = "stratified",
    setting: str = "vnet5",
):
    presets = load_notebook_presets()
    repo = presets["repository"]
    lefed_defaults = presets["lefed"]["defaults"]
    spec = resolve_method_setting("lefed", setting, presets=presets, default_key="vnet5")

    model_name = spec["model_name"]
    batch_size = int(os.environ.get("BHSD_BATCH_SIZE") or spec["batch_size"])
    labeled_bs = int(os.environ.get("BHSD_LABELED_BS") or spec["labeled_bs"])
    max_iter = int(os.environ.get("BHSD_MAX_ITER") or spec["max_iter"])
    lr_max_iter = os.environ.get("BHSD_LR_MAX_ITER")
    base_lr_override = os.environ.get("BHSD_BASE_LR")
    single_gpu = os.environ.get("BHSD_SINGLE_GPU", "").lower() in ("1", "true", "yes")
    run_name_override = os.environ.get("BHSD_RUN_NAME", "").strip()
    extra_overrides_raw = os.environ.get("BHSD_EXTRA_OVERRIDES", "").strip()

    val_stride_xy = spec["val_stride_xy"]
    val_stride_z = spec["val_stride_z"]
    patch_size_eval = list(spec["patch_size_eval"])
    config_file = spec["config_file"]
    arch_desc = spec["arch_desc"]
    extra_cli_lines = list(spec.get("extra_cli_lines", []))

    def _append_cli_pair(flag: str, value: str) -> None:
        extra_cli_lines.append(f"    {json.dumps(flag)}, {json.dumps(str(value))},\n")

    if lr_max_iter:
        _append_cli_pair("--lr_max_iteration", int(lr_max_iter))
    if base_lr_override:
        _append_cli_pair("--base_lr", float(base_lr_override))
    if extra_overrides_raw:
        parsed_extra = json.loads(extra_overrides_raw)
        if not isinstance(parsed_extra, list):
            raise ValueError("BHSD_EXTRA_OVERRIDES must be a JSON list of [key, value] pairs")
        for pair in parsed_extra:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Invalid BHSD_EXTRA_OVERRIDES entry: {pair!r}")
            key = str(pair[0]).strip().lstrip("-")
            val = str(pair[1]).strip()
            if not key or not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
                raise ValueError(f"Invalid override key: {key!r}")
            _append_cli_pair(f"--{key}", val)

    fold_id = lefed_defaults["fold_id"]
    val_interval = lefed_defaults["val_interval"]
    save_interval = lefed_defaults["save_interval"]
    wandb_project = lefed_defaults["wandb_project"]
    exp_dir = repo["exp_dir"]

    if run_name_override:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name_override):
            raise ValueError(f"BHSD_RUN_NAME contains invalid characters: {run_name_override!r}")
        exp_name_line = f"EXP_NAME = {json.dumps(run_name_override)}\n"
    else:
        task_tag = "bin" if task == "binary" else "mc"
        gpu_tag = "1gpu" if single_gpu else "ddp"
        name_parts = ["lefed", setting, task_tag, f"f{fold_id}", gpu_tag, f"lbl{int(labelnum)}", f"i{max_iter}"]
        exp_name_line = f"EXP_NAME = f{json.dumps('_'.join(name_parts) + '_{RUN_TIMESTAMP}')}\n"

    gpu_cmd_block = (
        [
            "num_gpus = 1 if (torch.cuda.is_available() and True) else 0\n",
            "use_ddp = False\n",
        ]
        if single_gpu
        else [
            "num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0\n",
            "use_ddp = (num_gpus >= 2)\n",
        ]
    )

    cells = [
        markdown_cell([
            "# MedSSL-BHSD: LeFeD Semi-Supervised Segmentation on Dual GPU T4 (Kaggle)\n",
            "\n",
            "3D brain hemorrhage segmentation on 2x T4 GPUs using **LeFeD (Learning Feedback Dual-Branch)** adapted for BHSD.\n",
            "\n",
            arch_desc,
            "- **3D Data Augmentation**: nnU-Net style RandomScaling3D, GammaTransform, and GaussianBlur3D.\n",
            "- **Mixed Precision (AMP FP16)**: GradScaler optimization for T4 Tensor Cores.\n",
            "- **Multi-GPU DDP (`torchrun`)**: Distributed training with periodic 3D sliding-window validation.\n",
            f"- **Labeled Split**: {labelnum} labeled cases ({labeled_sampling}) plus unlabeled pool for dual-decoder feedback consistency.\n",
            "- **Cloud Sync**: Metrics & visuals on W&B Cloud; checkpoints stored on Hugging Face Hub.\n",
            "- **Model Profiling (`evaluate.py --profile_only`)**: Measures Params, GFLOPs, Latency, and Peak VRAM.",
        ]),
        markdown_cell(["### Step 1: Install Required Packages"]),
        pip_install_cell(presets["packages"], quiet=True),
        markdown_cell(["### Step 2: Clone / Sync MedSSL-BHSD Repository"]),
        clone_repo_cell(
            repo["repo_name"],
            repo["work_dir"],
            flush=True,
            with_magic_cd=True,
            timeout=120,
            non_interactive=True,
        ),
        markdown_cell(["### Step 3: Locate BHSD Dataset"]),
        locate_dataset_cell(repo["input_root"], repo["working_root"], mode="mt"),
        markdown_cell(["### Step 4: Verify GPU Hardware"]),
        check_gpu_cell(flush=True, detailed=True),
        markdown_cell(["### Step 5: Configure Hugging Face Storage & Weights & Biases (W&B)"]),
        wandb_setup_cell(wandb_project, verify_viewer=True),
        markdown_cell(["### Step 6: Train LeFeD End-to-End (DDP)"]),
        code_cell(
            [
                "import os\n",
                "import sys\n",
                "import subprocess\n",
                "import datetime\n",
                "import socket\n",
                "import torch\n",
                "\n",
                f"FOLD_ID = {fold_id}\n",
                f"TASK = \"{task}\"\n",
                f"MODEL_NAME = \"{model_name}\"\n",
                "METHOD_NAME = \"lefed\"\n",
                f"SETTING = \"{setting}\"\n",
                "RUN_TIMESTAMP = datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                exp_name_line,
                f"EXP_DIR = \"{exp_dir}\"\n",
                f"CONFIG_FILE = \"{config_file}\"\n",
                f"MAX_ITER = {max_iter}\n",
                f"VAL_INTERVAL = {val_interval}\n",
                f"SAVE_INTERVAL = {save_interval}\n",
                f"BATCH_SIZE = {batch_size}\n",
                f"LABELED_BS = {labeled_bs}\n",
                f"LABELNUM = {int(labelnum)}\n",
                f"MAX_SAMPLES = {int(max_samples)}\n",
                f"LABELED_SAMPLING = \"{labeled_sampling}\"\n",
                f"PATCH_SIZE_EVAL = {list(patch_size_eval)}\n",
                "\n",
            ]
            + gpu_cmd_block
            + [
                "\n",
                "base_args = [\n",
                "    \"--config\", CONFIG_FILE,\n",
                "    \"--fold\", str(FOLD_ID),\n",
                "    \"--task\", TASK,\n",
                "    \"--model\", MODEL_NAME,\n",
                "    \"--exp\", EXP_NAME,\n",
                "    \"--exp_dir\", EXP_DIR,\n",
                "    \"--amp\",\n",
                "    \"--root_path\", str(DATA_ROOT),\n",
                "    \"--labelnum\", str(LABELNUM),\n",
                "    \"--max_samples\", str(MAX_SAMPLES),\n",
                "    \"--labeled_sampling\", str(LABELED_SAMPLING),\n",
                "    \"--batch_size\", str(BATCH_SIZE),\n",
                "    \"--labeled_bs\", str(LABELED_BS),\n",
                "    \"--max_iteration\", str(MAX_ITER),\n",
                "    \"--val_interval\", str(VAL_INTERVAL),\n",
                "    \"--save_interval\", str(SAVE_INTERVAL),\n",
                "    \"--use_wandb\",\n",
                "    \"--save_wandb_model\",\n",
                "    \"--wandb_project\", str(WANDB_PROJECT),\n",
            ]
            + extra_cli_lines
            + [
                "]\n",
                "\n",
                "def find_free_port():\n",
                "    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:\n",
                "        s.bind(('', 0))\n",
                "        return s.getsockname()[1]\n",
                "master_port = str(find_free_port())\n",
                "\n",
                "env = os.environ.copy()\n",
                "env[\"PYTHONUNBUFFERED\"] = \"1\"\n",
                "env[\"OMP_NUM_THREADS\"] = \"1\"\n",
                "env[\"HF_HUB_DISABLE_PROGRESS_BARS\"] = \"1\"\n",
                "env[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n",
                "env[\"NCCL_DEBUG\"] = \"WARN\"\n",
                "env[\"NCCL_IB_DISABLE\"] = \"1\"\n",
                "env[\"NCCL_P2P_DISABLE\"] = \"1\"\n",
                "env[\"TORCH_NCCL_ENABLE_MONITORING\"] = \"0\"\n",
                "env[\"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\"] = \"120\"\n",
                "\n",
                "if use_ddp:\n",
                "    print(f\"[INFO] Launching DDP via torchrun on {num_gpus} GPUs (port={master_port})...\", flush=True)\n",
                "    ddp_flag = [] if \"--ddp\" in base_args else [\"--ddp\"]\n",
                "    train_cmd = [\n",
                "        \"torchrun\",\n",
                "        f\"--nproc_per_node={num_gpus}\",\n",
                "        f\"--master_port={master_port}\",\n",
                "        \"train.py\",\n",
                "    ] + ddp_flag + base_args\n",
                "else:\n",
                "    print(\"[INFO] Launching single-GPU training (--gpu auto)...\", flush=True)\n",
                "    train_cmd = [\"python\", \"-u\", \"train.py\", \"--gpu\", \"auto\"] + base_args\n",
                "\n",
                "print(\"=\" * 75, flush=True)\n",
                "print(f\"[INFO] Command:\\n{' '.join(train_cmd)}\", flush=True)\n",
                "print(f\"[INFO] W&B Project: https://wandb.ai/{WANDB_PROJECT}\", flush=True)\n",
                "print(\"=\" * 75, flush=True)\n",
                "\n",
                "process = subprocess.Popen(\n",
                "    train_cmd,\n",
                "    stdout=subprocess.PIPE,\n",
                "    stderr=subprocess.STDOUT,\n",
                "    text=True,\n",
                "    bufsize=1,\n",
                "    env=env,\n",
                ")\n",
                "for line in process.stdout:\n",
                "    print(line, end=\"\", flush=True)\n",
                "process.wait()\n",
                "\n",
                "print(\"\\n\" + \"=\" * 75, flush=True)\n",
                "if process.returncode != 0:\n",
                "    raise RuntimeError(f\"[ERR] Training failed with exit code {process.returncode}.\")\n",
                "print(f\"[OK] Training completed successfully for {EXP_NAME}.\", flush=True)\n",
                "print(\"=\" * 75, flush=True)",
            ]
        ),
        markdown_cell(["### Step 7: Model Complexity & Efficiency Benchmark (`evaluate.py --profile_only`)"]),
        profile_model_cell(
            patch_size_eval=patch_size_eval,
            checkpoint_candidates=("best_model.pth", "checkpoint_latest.pth"),
        ),
    ]

    return build_notebook_envelope(cells, presets=presets)


if __name__ == "__main__":
    import argparse

    defaults = load_notebook_presets()["lefed"]["defaults"]
    parser = argparse.ArgumentParser(description="Generate LeFeD Kaggle Notebook")
    parser.add_argument(
        "--task",
        default=os.environ.get("BHSD_TASK", os.environ.get("TASK", defaults["task"])),
        choices=["multiclass", "binary"],
        help="Task type: 'multiclass' or 'binary'",
    )
    parser.add_argument(
        "--labelnum",
        type=int,
        default=int(os.environ.get("BHSD_LABELNUM", str(defaults["labelnum"]))),
        help="Number of labeled samples (default: 8 for 5%% of 154)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ.get("BHSD_MAX_SAMPLES", str(defaults["max_samples"]))),
        help="Max training pool samples (default: 154)",
    )
    parser.add_argument(
        "--labeled-sampling",
        default=os.environ.get("BHSD_LABELED_SAMPLING", defaults["labeled_sampling"]),
        choices=["first", "stratified"],
        help="Labeled subset sampling strategy: 'first' or 'stratified'",
    )
    parser.add_argument(
        "--setting",
        default=os.environ.get("BHSD_SETTING", "vnet5"),
        choices=["default", "vnet5"],
        help="Hyperparameter setting: 'default' or 'vnet5'",
    )
    args = parser.parse_args()

    nb = build_lefed_notebook(
        task=args.task,
        labelnum=args.labelnum,
        max_samples=args.max_samples,
        labeled_sampling=args.labeled_sampling,
        setting=args.setting,
    )
    target = "scripts/kaggle_bhsd_lefed_dual_gpu.ipynb"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(
        f"Generated {target} successfully "
        f"(task={args.task}, setting={args.setting}, labelnum={args.labelnum}, "
        f"max_samples={args.max_samples}, labeled_sampling={args.labeled_sampling})."
    )
