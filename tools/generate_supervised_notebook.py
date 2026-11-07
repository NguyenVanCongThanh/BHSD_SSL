import json
import os
import sys

def build_supervised_notebook(
    task: str = "multiclass",
    labelnum: int = 154,
    max_samples: int = 154,
    setting: str = "opt1_nnunet",
):
    if setting == "vnet":
        model_name = "vnet_single"
        batch_size = 4
        labeled_bs = 4
        max_iter = 25000
        val_stride_xy = 32
        val_stride_z = 8
        patch_size_eval = ["28", "256", "256"]
        config_file = "configs/experiments/bhsd_supervised_fold0.yaml"
        arch_desc = (
            "- **Kiến trúc V-Net Baseline Supervised**: Huấn luyện có giám sát 100% dữ liệu với mô hình V-Net 3D.\n"
        )
        extra_cli_lines = [
            '    "--model", "vnet_single",\n',
        ]
    else:  # opt1_nnunet default
        model_name = "nnunet3d"
        batch_size = 4  # 2 samples/GPU on Dual GPU T4
        labeled_bs = 4
        max_iter = 25000
        val_stride_xy = 32
        val_stride_z = 8
        patch_size_eval = ["28", "256", "256"]
        config_file = "configs/experiments/bhsd_supervised_nnunet_opt1_fold0.yaml"
        arch_desc = (
            "- **Kiến trúc 7-Stage nnU-Net 3D (Option 1 Design)**: Sử dụng mô hình nnU-Net 3D 7 stages (`nnunet3d`) "
            "với InstanceNorm3d, LeakyReLU, kernel anisotropic (1, 3, 3) ở 2 tầng đầu bảo toàn chi tiết trục Z, "
            "patch size [28, 256, 256], tích hợp **Deep Supervision** đa tầng chuẩn nnU-Net v2.\n"
        )
        extra_cli_lines = [
            '    "--model", "nnunet3d",\n',
        ]

    cells = [
        # Cell 0: Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
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
                "- **Benchmark độc lập (`evaluate.py`)**: Tự động đo FLOPs, Latency, Params và per-class Dice chi tiết."
            ]
        },
        # Cell 1: Install packages
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%pip install -q medpy tensorboardX h5py SimpleITK nibabel \"wandb>=0.19.10\" pyyaml"
            ]
        },
        # Cell 2: Clone repository
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import subprocess\n",
                "\n",
                "REPO_NAME = \"NguyenVanCongThanh/BHSD_SSL\"\n",
                "GITHUB_TOKEN = \"\"\n",
                "WORK_DIR = \"/kaggle/working/BHSD_SSL\"\n",
                "\n",
                "if GITHUB_TOKEN.strip():\n",
                "    GITHUB_REPO_URL = f\"https://{GITHUB_TOKEN.strip()}@github.com/{REPO_NAME}.git\"\n",
                "else:\n",
                "    GITHUB_REPO_URL = f\"https://github.com/{REPO_NAME}.git\"\n",
                "\n",
                "if not os.path.exists(WORK_DIR):\n",
                "    print(f\"[*] Đang clone repository {REPO_NAME}...\", flush=True)\n",
                "    subprocess.run([\"git\", \"clone\", GITHUB_REPO_URL, WORK_DIR], check=True)\n",
                "else:\n",
                "    print(f\"[*] Đồng bộ mã nguồn mới nhất từ GitHub tại {WORK_DIR}...\", flush=True)\n",
                "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"fetch\", \"origin\"], check=True)\n",
                "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"checkout\", \".\"], check=True)\n",
                "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"clean\", \"-fd\"], check=True)\n",
                "    res = subprocess.run([\"git\", \"-C\", WORK_DIR, \"rev-parse\", \"--abbrev-ref\", \"HEAD\"], capture_output=True, text=True)\n",
                "    branch = res.stdout.strip() if res.returncode == 0 and res.stdout.strip() else \"main\"\n",
                "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"reset\", \"--hard\", f\"origin/{branch}\"], check=True)\n",
                "\n",
                "os.chdir(WORK_DIR)\n",
                "print(f\"[OK] Đã vào thư mục làm việc: {os.getcwd()}\", flush=True)"
            ]
        },
        # Cell 3: Locate Dataset
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import glob\n",
                "\n",
                "INPUT_DIR = \"/kaggle/input\"\n",
                "WORKING_DIR = \"/kaggle/working\"\n",
                "\n",
                "def find_bhsd_root(search_dir):\n",
                "    for root, dirs, files in os.walk(search_dir):\n",
                "        has_data = any(d.lower() == \"data\" for d in dirs)\n",
                "        has_splits = any(d.lower() == \"splits\" for d in dirs)\n",
                "        if has_data and has_splits:\n",
                "            return root\n",
                "    return None\n",
                "\n",
                "print(\"[*] Đang định vị dataset BHSD...\", flush=True)\n",
                "DATA_ROOT = find_bhsd_root(INPUT_DIR) or find_bhsd_root(WORKING_DIR)\n",
                "\n",
                "if not DATA_ROOT:\n",
                "    all_tars = glob.glob(os.path.join(INPUT_DIR, \"**\", \"*.tar*\"), recursive=True)\n",
                "    matching_tars = [p for p in all_tars if any(k in os.path.basename(p).lower() for k in [\"bhsd\", \"h5\", \"fold\"])]\n",
                "    tar_path = matching_tars[0] if matching_tars else (all_tars[0] if all_tars else None)\n",
                "    if not tar_path:\n",
                "        raise FileNotFoundError(f\"[X] Không tìm thấy dataset BHSD trong {INPUT_DIR}!\")\n",
                "    print(f\"[*] Đang giải nén {os.path.basename(tar_path)} vào {WORKING_DIR}...\", flush=True)\n",
                "    if tar_path.endswith(\".tar.gz\") or tar_path.endswith(\".tgz\"):\n",
                "        !tar -xzf \"{tar_path}\" -C {WORKING_DIR}\n",
                "    else:\n",
                "        !tar -xf \"{tar_path}\" -C {WORKING_DIR}\n",
                "    DATA_ROOT = find_bhsd_root(WORKING_DIR)\n",
                "\n",
                "print(f\"[SUCCESS] DATA_ROOT: {DATA_ROOT}\", flush=True)"
            ]
        },
        # Cell 4: Check GPUs
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "assert torch.cuda.is_available(), \"[CẢNH BÁO] Chưa bật GPU! Hãy bật GPU T4 x 2 trong Settings.\"\n",
                "num_gpus = torch.cuda.device_count()\n",
                "print(f\"[GPU] Phát hiện {num_gpus} GPU khả dụng:\", flush=True)\n",
                "for i in range(num_gpus):\n",
                "    print(f\"  [+] GPU {i}: {torch.cuda.get_device_name(i)}\", flush=True)"
            ]
        },
        # Cell 5: Setup W&B
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import wandb\n",
                "WANDB_API_KEY = \"\"\n",
                "WANDB_PROJECT = \"BHSD_Supervised\"\n",
                "if WANDB_API_KEY.strip():\n",
                "    os.environ[\"WANDB_API_KEY\"] = WANDB_API_KEY.strip()\n",
                "    os.environ[\"WANDB_MODE\"] = \"online\"\n",
                "    wandb.login(key=WANDB_API_KEY.strip(), relogin=True)\n",
                "else:\n",
                "    os.environ[\"WANDB_MODE\"] = \"offline\"\n",
                "os.environ[\"WANDB_PROJECT\"] = WANDB_PROJECT"
            ]
        },
        # Cell 6: Train Supervised Baseline
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import datetime\n",
                "import socket\n",
                "\n",
                "FOLD_ID = 0\n",
                f"TASK = \"{task}\"\n",
                f"MODEL_NAME = \"{model_name}\"\n",
                "METHOD_NAME = \"supervised\"\n",
                f"CONFIG_FILE = \"{config_file}\"\n",
                f"MAX_ITER = {max_iter}\n",
                "VAL_INTERVAL = 500\n",
                "SAVE_INTERVAL = 100\n",
                f"BATCH_SIZE = {batch_size}\n",
                f"LABELED_BS = {labeled_bs}\n",
                f"LABELNUM = {labelnum}\n",
                f"MAX_SAMPLES = {max_samples}\n",
                "\n",
                "RUN_TIMESTAMP = datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                f"EXP_NAME = f\"BHSD_{{TASK}}_Supervised_{{MODEL_NAME}}_fold{{FOLD_ID}}_{{RUN_TIMESTAMP}}\"\n",
                "EXP_DIR = \"/kaggle/working/experiments\"\n",
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
                "] + extra_cli_lines\n",
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
                "print(\"[✓] Huấn luyện Supervised Baseline hoàn tất thành công!\", flush=True)"
            ]
        },
        # Cell 7: Evaluate Best Model
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
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
                "        \"--fold\", str(FOLD_ID),\n",
                "        \"--stride_xy\", \"32\",\n",
                "        \"--stride_z\", \"8\",\n",
                "        \"--profile\"\n",
                "    ]\n",
                "    subprocess.run(eval_cmd, check=True)\n",
                "else:\n",
                "    print(f\"[!] Không tìm thấy checkpoint tại {best_pth}\")"
            ]
        },
        # Cell 8: Package Results
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "OUTPUT_ARCHIVE = f\"/kaggle/working/{EXP_NAME}_results.tar.gz\"\n",
                "if os.path.exists(EXP_DIR):\n",
                "    !tar -czf {OUTPUT_ARCHIVE} -C {EXP_DIR} .\n",
                "    print(f\"[✓] Đã tạo file kết quả {OUTPUT_ARCHIVE}\")"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "gpu": {
                "count": 2,
                "type": "T4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return notebook

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["multiclass", "binary"], default="binary")
    parser.add_argument("--setting", default="opt1_nnunet")
    parser.add_argument("--labelnum", type=int, default=154)
    parser.add_argument("--max_samples", type=int, default=154)
    args, _ = parser.parse_known_args()

    nb = build_supervised_notebook(
        task=args.task,
        labelnum=args.labelnum,
        max_samples=args.max_samples,
        setting=args.setting
    )
    target = "scripts/kaggle_bhsd_supervised_dual_gpu.ipynb"
    os.makedirs("scripts", exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Updated {target} (task={args.task}, setting={args.setting}) successfully.")

if __name__ == "__main__":
    main()

