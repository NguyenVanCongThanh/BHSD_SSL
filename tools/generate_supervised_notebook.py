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
    package_results_cell,
    pip_install_cell,
    resolve_method_setting,
    wandb_setup_cell,
)


def build_supervised_notebook(
    task: str = "multiclass",
    labelnum: int = 154,
    max_samples: int = 154,
    setting: str = "opt1_nnunet",
):
    presets = load_notebook_presets()
    repo = presets["repository"]
    sup_defaults = presets["supervised"]["defaults"]
    spec = resolve_method_setting("supervised", setting, presets=presets, default_key="opt1_nnunet")

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

    fold_id = sup_defaults["fold_id"]
    val_interval = sup_defaults["val_interval"]
    save_interval = sup_defaults["save_interval"]
    wandb_project = sup_defaults["wandb_project"]
    exp_dir = repo["exp_dir"]

    cells = [
        markdown_cell([
            "# 🧠 MedSSL-BHSD: Full Supervised Baseline (100% Data) trên Dual GPU T4 (Kaggle)\n",
            "\n",
            "Huấn luyện baseline có giám sát toàn diện (**Full Supervised**) sử dụng 100% dữ liệu có nhãn (154 ca tập train Fold 0) để so sánh công bằng tuyệt đối với nnU-Net Option 1 và các phương pháp Semi-Supervised (Mean Teacher, PICK, CPS).\n",
            "\n",
            arch_desc,
            "- **Data Augmentation 3D Chuẩn nnU-Net**: Tự động áp dụng RandomScaling3D, GammaTransform và GaussianBlur3D chống overfitting.\n",
            "- **Automatic Mixed Precision (AMP FP16) & GradScaler**: Tự động tối ưu trên Tensor Cores của 2x GPU T4, tiết kiệm VRAM và tăng tốc độ huấn luyện.\n",
            "- **Polynomial Learning Rate Decay**: $lr = base\\_lr \\times (1 - iter/max\\_iter)^{0.9}$ giúp hội tụ mượt mà.\n",
            "- **Deep Supervision**: Kích hoạt loss đa tầng tại các decoder stage theo trọng số phân rã hàm mũ chuẩn nnU-Net ($w_i = 2^{-i} / \\sum 2^{-k}$).\n",
            "- **Đa GPU DDP (DistributedDataParallel)**: Tự động phân bổ đều 2 mẫu/GPU trên 2x GPU T4 qua `torchrun`.\n",
            "- **3D Sliding-Window Validation**: Định kỳ đánh giá toàn bộ 38 ca validation gốc bằng sliding-window và cập nhật `best_model.pth`.\n",
            "- **Benchmark độc lập (`evaluate.py`)**: Tự động đo FLOPs, Latency, Params và per-class Dice chi tiết.",
        ]),
        pip_install_cell(presets["packages"], quiet=True),
        clone_repo_cell(
            repo["repo_name"],
            repo["work_dir"],
            flush=True,
            with_magic_cd=False,
            timeout=None,
            non_interactive=False,
        ),
        locate_dataset_cell(repo["input_root"], repo["working_root"], mode="supervised"),
        check_gpu_cell(flush=True, detailed=False),
        wandb_setup_cell(wandb_project, verify_viewer=False),
        code_cell(
            [
                "import datetime\n",
                "import socket\n",
                "\n",
                f"FOLD_ID = {fold_id}\n",
                f"TASK = \"{task}\"\n",
                f"MODEL_NAME = \"{model_name}\"\n",
                "METHOD_NAME = \"supervised\"\n",
                f"CONFIG_FILE = \"{config_file}\"\n",
                f"MAX_ITER = {max_iter}\n",
                f"VAL_INTERVAL = {val_interval}\n",
                f"SAVE_INTERVAL = {save_interval}\n",
                f"BATCH_SIZE = {batch_size}\n",
                f"LABELED_BS = {labeled_bs}\n",
                f"LABELNUM = {labelnum}\n",
                f"MAX_SAMPLES = {max_samples}\n",
                f"PATCH_SIZE_EVAL = {patch_size_eval}\n",
                "\n",
                "RUN_TIMESTAMP = datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                f"EXP_NAME = f\"BHSD_{{TASK}}_Supervised_{{MODEL_NAME}}_fold{{FOLD_ID}}_{{RUN_TIMESTAMP}}\"\n",
                f"EXP_DIR = \"{exp_dir}\"\n",
                "\n",
                "num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0\n",
                "use_ddp = (num_gpus >= 2)\n",
                "\n",
                "base_args = [\n",
                "    \"--config\", CONFIG_FILE,\n",
                "    \"--fold\", str(FOLD_ID),\n",
                "    \"--task\", TASK,\n",
                "    \"--method\", METHOD_NAME,\n",
                "    \"--exp\", EXP_NAME,\n",
                "    \"--exp_dir\", EXP_DIR,\n",
                "    \"--amp\",\n",
                "    \"--root_path\", str(DATA_ROOT),\n",
                "    \"--labelnum\", str(LABELNUM),\n",
                "    \"--max_samples\", str(MAX_SAMPLES),\n",
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
                "env[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n",
                "\n",
                "if use_ddp:\n",
                "    print(f\"[*] Chạy chế độ DDP qua torchrun trên {num_gpus} GPUs...\", flush=True)\n",
                "    train_cmd = [\n",
                "        \"torchrun\",\n",
                "        f\"--nproc_per_node={num_gpus}\",\n",
                "        f\"--master_port={master_port}\",\n",
                "        \"train.py\",\n",
                "        \"--ddp\"\n",
                "    ] + base_args\n",
                "else:\n",
                "    print(\"[*] Chạy chế độ đơn GPU...\", flush=True)\n",
                "    train_cmd = [\"python\", \"-u\", \"train.py\", \"--gpu\", \"auto\"] + base_args\n",
                "\n",
                "print(\"=\" * 70, flush=True)\n",
                "print(f\"[*] Lệnh chạy:\\n{' '.join(train_cmd)}\", flush=True)\n",
                "print(\"=\" * 70, flush=True)\n",
                "\n",
                "process = subprocess.Popen(\n",
                "    train_cmd,\n",
                "    stdout=subprocess.PIPE,\n",
                "    stderr=subprocess.STDOUT,\n",
                "    text=True,\n",
                "    bufsize=1\n",
                ")\n",
                "for line in iter(process.stdout.readline, ''):\n",
                "    print(line, end='', flush=True)\n",
                "process.stdout.close()\n",
                "ret = process.wait()\n",
                "if ret != 0:\n",
                "    raise RuntimeError(f\"Huấn luyện thất bại với mã lỗi {ret}\")\n",
                "print(\"[✓] Huấn luyện Supervised Baseline hoàn tất thành công!\", flush=True)",
            ]
        ),
        code_cell([
            "snapshot_dir = os.path.join(EXP_DIR, EXP_NAME, f\"fold_{FOLD_ID}\")\n",
            "best_pth = os.path.join(snapshot_dir, \"best_model.pth\")\n",
            "if not os.path.exists(best_pth):\n",
            "    best_pth = os.path.join(snapshot_dir, \"self_train\", \"best_model.pth\")\n",
            "if not os.path.exists(best_pth):\n",
            "    best_pth = os.path.join(snapshot_dir, \"checkpoint_latest.pth\")\n",
            "\n",
            "if os.path.exists(best_pth):\n",
            "    print(f\"[✓] Đang đánh giá model: {best_pth}\")\n",
            "    eval_cmd = [\n",
            "        \"python\", \"evaluate.py\",\n",
            "        \"--checkpoint\", best_pth,\n",
            "        \"--root_path\", str(DATA_ROOT),\n",
            "        \"--task\", TASK,\n",
            "        \"--model\", MODEL_NAME,\n",
            "        \"--patch_size\", *PATCH_SIZE_EVAL,\n",
            "        \"--fold\", str(FOLD_ID),\n",
            f"        \"--stride_xy\", \"{val_stride_xy}\",\n",
            f"        \"--stride_z\", \"{val_stride_z}\",\n",
            "        \"--profile\"\n",
            "    ]\n",
            "    try:\n",
            "        subprocess.run(eval_cmd, check=True)\n",
            "    except Exception as e:\n",
            "        print(f\"[!] Cảnh báo: Đánh giá bằng evaluate.py gặp lỗi nhưng không ảnh hưởng tới kết quả huấn luyện: {e}\", flush=True)\n",
            "else:\n",
            "    print(f\"[!] Không tìm thấy checkpoint tại {best_pth}\")",
        ]),
        package_results_cell(repo["working_root"], sync_wandb=False, flush=False),
    ]

    return build_notebook_envelope(cells, presets=presets)


def main():
    import argparse

    defaults = load_notebook_presets()["supervised"]["defaults"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["multiclass", "binary"], default=os.environ.get("BHSD_TASK", "binary"))
    parser.add_argument("--setting", default=os.environ.get("BHSD_SETTING", "all"), help="'opt1_nnunet', 'vnet', or 'all' to build both")
    parser.add_argument("--labelnum", type=int, default=defaults["labelnum"])
    parser.add_argument("--max_samples", type=int, default=defaults["max_samples"])
    parser.add_argument("--output", default=None)
    args, _ = parser.parse_known_args()

    os.makedirs("scripts", exist_ok=True)

    if args.setting in ("all", "default"):
        nb_nnunet = build_supervised_notebook(
            task=args.task,
            labelnum=args.labelnum,
            max_samples=args.max_samples,
            setting="opt1_nnunet",
        )
        target_nnunet = "scripts/kaggle_bhsd_supervised_dual_gpu.ipynb"
        with open(target_nnunet, "w", encoding="utf-8") as f:
            json.dump(nb_nnunet, f, indent=1, ensure_ascii=False)
        print(f"Updated {target_nnunet} (task={args.task}, setting=opt1_nnunet) successfully.")

        nb_vnet = build_supervised_notebook(
            task=args.task,
            labelnum=args.labelnum,
            max_samples=args.max_samples,
            setting="vnet",
        )
        target_vnet = "scripts/kaggle_bhsd_supervised_vnet_dual_gpu.ipynb"
        with open(target_vnet, "w", encoding="utf-8") as f:
            json.dump(nb_vnet, f, indent=1, ensure_ascii=False)
        print(f"Updated {target_vnet} (task={args.task}, setting=vnet) successfully.")
    else:
        nb = build_supervised_notebook(
            task=args.task,
            labelnum=args.labelnum,
            max_samples=args.max_samples,
            setting=args.setting,
        )
        if args.output:
            target = args.output
        elif args.setting in ("vnet", "vnet_opt1"):
            target = "scripts/kaggle_bhsd_supervised_vnet_dual_gpu.ipynb"
        else:
            target = "scripts/kaggle_bhsd_supervised_dual_gpu.ipynb"

        with open(target, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Updated {target} (task={args.task}, setting={args.setting}) successfully.")


if __name__ == "__main__":
    main()
