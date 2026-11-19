import json
import os
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


def build_mean_teacher_notebook(
    task: str = "multiclass",
    labelnum: int = 154,
    max_samples: int = 2200,
    labeled_sampling: str = "first",
    setting: str = "default",
):
    presets = load_notebook_presets()
    repo = presets["repository"]
    mt_defaults = presets["mean_teacher"]["defaults"]
    spec = resolve_method_setting("mean_teacher", setting, presets=presets, default_key="default")

    model_name = spec["model_name"]
    batch_size = spec["batch_size"]
    labeled_bs = spec["labeled_bs"]
    max_iter = spec["max_iter"]
    val_stride_xy = spec["val_stride_xy"]
    val_stride_z = spec["val_stride_z"]
    patch_size_eval = spec["patch_size_eval"]
    config_file = spec["config_file"]
    arch_desc = spec["arch_desc"]
    extra_cli_lines = spec.get("extra_cli_lines", [])

    fold_id = mt_defaults["fold_id"]
    val_interval = mt_defaults["val_interval"]
    save_interval = mt_defaults["save_interval"]
    wandb_project = mt_defaults["wandb_project"]
    exp_dir = repo["exp_dir"]

    cells = [
        markdown_cell([
            "# MedSSL-BHSD: Mean Teacher Semi-Supervised Segmentation on Dual GPU T4 (Kaggle)\n",
            "\n",
            "3D brain hemorrhage segmentation on 2x T4 GPUs using **Mean Teacher (Tarvainen & Valpola, NeurIPS 2017)** optimized for BHSD.\n",
            "\n",
            arch_desc,
            "- **3D Data Augmentation**: nnU-Net style RandomScaling3D, GammaTransform, and GaussianBlur3D.\n",
            "- **Mixed Precision (AMP FP16)**: GradScaler optimization for T4 Tensor Cores.\n",
            "- **EMA Teacher & Balanced Consistency**: Exponential moving average teacher with class-balanced consistency loss.\n",
            "- **Multi-GPU DDP (`torchrun`)**: Distributed training with periodic 3D sliding-window validation on the EMA teacher.\n",
            f"- **Labeled Split**: {labelnum} labeled cases plus unlabeled pool for consistency regularization.\n",
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
        markdown_cell([
            "### Step 5: Configure Hugging Face Storage & Weights & Biases (W&B)",
        ]),
        wandb_setup_cell(wandb_project, verify_viewer=True),
        markdown_cell([
            "### Step 6: Train Mean Teacher End-to-End (DDP)",
        ]),
        code_cell(
            [
                "import os\n",
                "import sys\n",
                "import subprocess\n",
                "import datetime\n",
                "import torch\n",
                "\n",
                f"FOLD_ID = {fold_id}\n",
                f"TASK = \"{task}\"\n",
                f"MODEL_NAME = \"{model_name}\"\n",
                "METHOD_NAME = \"mean_teacher\"\n",
                f"SETTING = \"{setting}\"\n",
                "RUN_TIMESTAMP = datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                f"EXP_NAME = f\"BHSD_{{TASK}}_MeanTeacher_{{SETTING}}_fold{{FOLD_ID}}_{{RUN_TIMESTAMP}}\"\n",
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
                "num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0\n",
                "use_ddp = (num_gpus >= 2)\n",
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
                "# Cấu hình Anti-Deadlock cho môi trường Đa GPU (2x T4):\n",
                "import socket\n",
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
                "    train_cmd = [\n",
                "        \"python\", \"-u\", \"train.py\",\n",
                "        \"--gpu\", \"auto\"\n",
                "    ] + base_args\n",
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
                "\n",
                "for line in process.stdout:\n",
                "    print(line, end=\"\", flush=True)\n",
                "\n",
                "process.wait()\n",
                "\n",
                "print(\"\\n\" + \"=\" * 75, flush=True)\n",
                "if process.returncode != 0:\n",
                "    raise RuntimeError(f\"[ERR] Training failed with exit code {process.returncode}.\")\n",
                "else:\n",
                "    print(f\"[OK] Training completed successfully for {EXP_NAME}.\", flush=True)\n",
                "print(\"=\" * 75, flush=True)",
            ]
        ),
        markdown_cell([
            "### Step 7: Model Complexity & Efficiency Benchmark (`evaluate.py --profile_only`)",
        ]),
        profile_model_cell(
            patch_size_eval=patch_size_eval,
            checkpoint_candidates=("best_model.pth", "checkpoint_latest.pth"),
        ),
    ]

    return build_notebook_envelope(cells, presets=presets)


if __name__ == "__main__":
    import argparse

    defaults = load_notebook_presets()["mean_teacher"]["defaults"]
    parser = argparse.ArgumentParser(description="Generate Mean Teacher Kaggle Notebook")
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
        help="Number of labeled samples (default: 154; use 8 for 5%% of 154)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ.get("BHSD_MAX_SAMPLES", str(defaults["max_samples"]))),
        help="Max training pool samples (default: 2200; use 154 to exclude 1,978 unlabeled cases)",
    )
    parser.add_argument(
        "--labeled-sampling",
        default=os.environ.get("BHSD_LABELED_SAMPLING", defaults["labeled_sampling"]),
        choices=["first", "stratified"],
        help="Labeled subset sampling strategy: 'first' (default) or 'stratified'",
    )
    parser.add_argument(
        "--setting",
        default=os.environ.get("BHSD_SETTING", "default"),
        choices=["default", "swdl", "vnet", "vnet_swdl"],
        help="Hyperparameter setting: 'default' (nnUNet3D), 'swdl' (VNet SWDL), 'vnet' (7-Stage Residual V-Net Concat), or 'vnet_swdl' (7-Stage V-Net Add/ReLU)",
    )
    args = parser.parse_args()

    nb = build_mean_teacher_notebook(
        task=args.task,
        labelnum=args.labelnum,
        max_samples=args.max_samples,
        labeled_sampling=args.labeled_sampling,
        setting=args.setting,
    )
    targets = [
        "scripts/kaggle_bhsd_mean_teacher_dual_gpu.ipynb"
    ]
    for target in targets:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(
            f"Generated {target} successfully "
            f"(task={args.task}, labelnum={args.labelnum}, max_samples={args.max_samples}, "
            f"labeled_sampling={args.labeled_sampling})."
        )
