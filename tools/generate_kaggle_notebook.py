import json
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
    wandb_setup_cell,
)


def build_notebook_dict():
    presets = load_notebook_presets()
    repo = presets["repository"]
    pick = presets["pick"]["defaults"]
    wandb_project = pick["wandb_project"]
    exp_dir = repo["exp_dir"]

    cells = [
        markdown_cell([
            "# MedSSL-BHSD: PICK Semi-Supervised Segmentation on Dual GPU T4 (Kaggle)\n",
            "\n",
            "3D brain hemorrhage segmentation on 2x T4 GPUs using **PICK** within the unified MedSSL-BHSD framework.\n",
            "\n",
            "- **Multi-GPU DDP (`torchrun`)**: Distributed training with periodic 3D sliding-window validation.\n",
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
            timeout=None,
            non_interactive=False,
        ),
        markdown_cell(["### Step 3: Locate BHSD Dataset"]),
        locate_dataset_cell(repo["input_root"], repo["working_root"], mode="pick"),
        markdown_cell(["### Step 4: Verify GPU Hardware"]),
        check_gpu_cell(flush=True, detailed=True),
        markdown_cell([
            "### Step 5: Configure Hugging Face Storage & Weights & Biases (W&B)",
        ]),
        wandb_setup_cell(wandb_project, verify_viewer=True),
        markdown_cell([
            "### Step 6: Train PICK Multi-Class (DDP)",
        ]),
        code_cell([
            "import os\n",
            "import sys\n",
            "import subprocess\n",
            "import torch\n",
            "\n",
            f"FOLD_ID = {pick['fold_id']}\n",
            f"TASK = \"{pick['task']}\"\n",
            f"MODEL_NAME = \"{pick['model_name']}\"\n",
            "EXP_NAME = f\"BHSD_{TASK}_PICK_dual_gpu_fold{FOLD_ID}\"\n",
            f"EXP_DIR = \"{exp_dir}\"\n",
            f"CONFIG_FILE = \"{pick['config_file']}\"\n",
            f"VAL_INTERVAL = {pick['val_interval']}\n",
            f"SAVE_INTERVAL = {pick['save_interval']}\n",
            f"BATCH_SIZE = {pick['batch_size']}\n",
            f"LABELED_BS = {pick['labeled_bs']}\n",
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
            "    \"--root_path\", str(DATA_ROOT),\n",
            "    \"--batch_size\", str(BATCH_SIZE),\n",
            "    \"--labeled_bs\", str(LABELED_BS),\n",
            "    \"--val_interval\", str(VAL_INTERVAL),\n",
            "    \"--save_interval\", str(SAVE_INTERVAL),\n",
            "    \"--use_wandb\",\n",
            "    \"--save_wandb_model\",\n",
            "    \"--wandb_project\", str(WANDB_PROJECT)\n",
            "]\n",
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
            "    print(f\"[INFO] Launching DDP via torchrun on {num_gpus} GPUs...\", flush=True)\n",
            "    ddp_flag = [] if \"--ddp\" in base_args else [\"--ddp\"]\n",
            "    train_cmd = [\n",
            "        \"torchrun\",\n",
            "        f\"--nproc_per_node={num_gpus}\",\n",
            "        \"--master_port=29500\",\n",
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
        ]),
        markdown_cell([
            "### Step 7: Model Complexity & Efficiency Benchmark (`evaluate.py --profile_only`)",
        ]),
        profile_model_cell(
            patch_size_eval=None,
            checkpoint_candidates=(
                "self_train/best_model.pth",
                "self_train/VNet_best_model.pth",
                "pre_train/best_model.pth",
                "pre_train/VNet_best_model.pth",
            ),
        ),
    ]

    return build_notebook_envelope(cells, presets=presets)


def main():
    nb = build_notebook_dict()
    targets = [
        "scripts/kaggle_bhsd_pick_dual_gpu.ipynb"
    ]
    for target in targets:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Updated {target} successfully.")


if __name__ == "__main__":
    main()
