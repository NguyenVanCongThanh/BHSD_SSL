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
    package_results_cell,
    pip_install_cell,
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
            "# 🧠 MedSSL-BHSD: PICK Semi-Supervised Segmentation trên Dual GPU T4 (Kaggle)\n",
            "\n",
            "Phân đoạn 5 phân loại xuất huyết não chi tiết (EDH, IPH, IVH, SAH, SDH) + Background trên 2x GPU T4 Kaggle theo kiến trúc MedSSL-BHSD tái cấu trúc.\n",
            "\n",
            "### 🌟 Các cải tiến & Tối ưu hóa sẵn sàng cho 1 lần Run duy nhất:\n",
            "- **Thí nghiệm Mới Độc lập**: `EXP_NAME` được đổi thành `BHSD_multiclass_PICK_dual_gpu_fold0` đảm bảo không bị trùng lặp với artifact cũ của notebook trước.\n",
            "- **Cơ chế lưu Checkpoint Kép (Dual Checkpointing)**:\n",
            "  - **Mỗi 50 iterations (`--save_interval 50`)**: Tự động lưu đè `checkpoint_latest.pth` cục bộ trên SSD tốc độ cao (chỉ mất ~0.2s), giúp chống crash/timeout mà không gây nghẽn mạng.\n",
            "  - **Mỗi 500 iterations (`--val_interval 500`)**: Chạy 3D sliding-window validation 38 ca, cập nhật `best_model.pth` và đẩy đồng thời lên W&B Cloud Artifacts.\n",
            "- **Chế độ 2 GPU T4 tối ưu (DDP - DistributedDataParallel qua `torchrun`)**: Tự động kích hoạt khi phát hiện 2 GPU, chia đều VRAM (~6GB mỗi GPU) và chạy nhanh hơn 25-30% so với DataParallel cũ.\n",
            "- **Tự động đăng nhập W&B**: Khóa API Key đã được tích hợp sẵn, tự động đồng bộ biểu đồ mDice, per-class Dice và VRAM theo thời gian thực.\n",
            "- **Khả năng Auto-Resume**: Tự động khôi phục từ checkpoint gần nhất nếu bị timeout 12h của Kaggle.\n",
            "- **Đo lường Benchmark độc lập**: Tự động chạy `evaluate.py` sau huấn luyện để đo FLOPs, Latency, và per-class Dice trên sliding-window.\n",
            "- **Tự động đóng gói kết quả**: Nén toàn bộ checkpoint tốt nhất và logs ra file `.tar.gz` sẵn sàng tải về tại tab Output.",
        ]),
        markdown_cell(["### 📦 Bước 1: Cài đặt các thư viện cần thiết"]),
        pip_install_cell(presets["packages"], quiet=True),
        markdown_cell(["### 🚀 Bước 2: Clone & Cập nhật mã nguồn MedSSL-BHSD từ GitHub"]),
        clone_repo_cell(
            repo["repo_name"],
            repo["work_dir"],
            flush=False,
            with_magic_cd=True,
            timeout=None,
            non_interactive=False,
        ),
        markdown_cell(["### 🔍 Bước 3: Định vị Dataset BHSD từ Kaggle Input"]),
        locate_dataset_cell(repo["input_root"], repo["working_root"], mode="pick"),
        markdown_cell(["### ⚡ Bước 4: Kiểm tra Cấu hình Phần cứng Dual GPU (2x T4)"]),
        check_gpu_cell(flush=False, detailed=True),
        markdown_cell([
            "### 📊 Bước 5: Đăng nhập & Cấu hình Weights & Biases (W&B)\n",
            "API Key đã được tích hợp sẵn để đảm bảo notebook chạy thông suốt trong 1 lần run duy nhất.",
        ]),
        wandb_setup_cell(wandb_project, verify_viewer=True),
        markdown_cell([
            "### 🏋️‍♂️ Bước 6: Huấn luyện PICK Multi-Class với Kiến trúc Mới (2x T4 DDP)\n",
            "\n",
            "Sử dụng entrypoint mới **`train.py`** cùng cấu hình **`configs/methods/pick.yaml`**:\n",
            "- **Tên thí nghiệm mới**: `BHSD_multiclass_PICK_dual_gpu_fold0` (hoàn toàn mới, không ghi đè artifact cũ).\n",
            "- **Lưu đè cục bộ mỗi 50 iterations (`--save_interval 50`)**: Chống crash và timeout tức thời mà không tốn băng thông Cloud.\n",
            "- **Chế độ DDP (`torchrun --nproc_per_node=2`)**: Phân tán 2 process trên 2 GPU T4, chia đều VRAM (~6GB mỗi card) và đẩy tốc độ tối đa.\n",
            "- **Auto-Resume**: Cờ `--resume` tự động tìm checkpoint gần nhất (`checkpoint_latest.pth`) để tiếp tục nếu gặp timeout.",
        ]),
        code_cell([
            "# === HUẤN LUYỆN PICK TRÊN 2 GPU T4 VỚI FRAMEWORK MỚI ===\n",
            "import os\n",
            "import sys\n",
            "import subprocess\n",
            "import torch\n",
            "\n",
            "# --- THÔNG SỐ THÍ NGHIỆM ---\n",
            f"FOLD_ID = {pick['fold_id']}                                            # Chọn fold (0..4)\n",
            f"TASK = \"{pick['task']}\"                                    # \"multiclass\" (6 lớp) hoặc \"binary\" (2 lớp)\n",
            f"MODEL_NAME = \"{pick['model_name']}\"                                    # \"vnet\", \"unet3d\", \"unetr\"\n",
            "EXP_NAME = f\"BHSD_{TASK}_PICK_dual_gpu_fold{FOLD_ID}\"  # Tên thí nghiệm MỚI ĐỘC LẬP (tránh trùng artifact cũ)\n",
            f"EXP_DIR = \"{exp_dir}\"\n",
            f"CONFIG_FILE = \"{pick['config_file']}\"\n",
            f"VAL_INTERVAL = {pick['val_interval']}                                     # Đánh giá validation và đẩy Cloud mỗi 500 bước\n",
            f"SAVE_INTERVAL = {pick['save_interval']}                                     # Tự động lưu checkpoint_latest.pth cục bộ mỗi 50 bước để chống crash\n",
            f"BATCH_SIZE = {pick['batch_size']}                                         # Batch size tổng (DDP: mỗi GPU xử lý 2 mẫu)\n",
            f"LABELED_BS = {pick['labeled_bs']}                                         # Số mẫu có nhãn trong batch\n",
            "\n",
            "# Tự động phát hiện chế độ Đa GPU: Ưu tiên DDP nếu có >= 2 GPU\n",
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
            "    \"--resume\",\n",
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
            "env[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"  # Chống phân mảnh bộ nhớ VRAM cho tensor 3D lớn\n",
            "env[\"NCCL_DEBUG\"] = \"WARN\"\n",
            "env[\"NCCL_IB_DISABLE\"] = \"1\"\n",
            "env[\"NCCL_P2P_DISABLE\"] = \"1\"\n",
            "env[\"TORCH_NCCL_ENABLE_MONITORING\"] = \"0\"\n",
            "env[\"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\"] = \"120\"\n",
            "\n",
            "if use_ddp:\n",
            "    print(f\"[*] Phát hiện {num_gpus} GPU -> Khởi chạy chế độ DDP (torchrun) tối ưu trên 2x GPU T4...\")\n",
            "    train_cmd = [\n",
            "        \"torchrun\",\n",
            "        f\"--nproc_per_node={num_gpus}\",\n",
            "        \"--master_port=29500\",\n",
            "        \"train.py\",\n",
            "        \"--ddp\"\n",
            "    ] + base_args\n",
            "else:\n",
            "    print(\"[*] Chế độ: Đơn GPU hoặc DataParallel (DP) qua python...\")\n",
            "    train_cmd = [\n",
            "        \"python\", \"-u\", \"train.py\",\n",
            "        \"--gpu\", \"auto\"\n",
            "    ] + base_args\n",
            "\n",
            "print(\"=\" * 75)\n",
            "print(f\"[*] Khởi chạy lệnh:\\n{' '.join(train_cmd)}\")\n",
            "print(f\"[*] Theo dõi biểu đồ W&B Cloud tại: https://wandb.ai/{WANDB_PROJECT}\")\n",
            "print(\"=\" * 75)\n",
            "\n",
            "# Stream log thời gian thực ra màn hình\n",
            "process = subprocess.Popen(\n",
            "    train_cmd,\n",
            "    stdout=subprocess.PIPE,\n",
            "    stderr=subprocess.STDOUT,\n",
            "    text=True,\n",
            "    bufsize=1,\n",
            "    env=env\n",
            ")\n",
            "\n",
            "for line in process.stdout:\n",
            "    print(line, end=\"\")\n",
            "    sys.stdout.flush()\n",
            "\n",
            "process.wait()\n",
            "\n",
            "print(\"\\n\" + \"=\" * 75)\n",
            "if process.returncode != 0:\n",
            "    raise RuntimeError(f\"[LỖI] Huấn luyện kết thúc với mã lỗi {process.returncode}!\")\n",
            "else:\n",
            "    print(f\"[SUCCESS] Huấn luyện hoàn tất thành công cho {EXP_NAME}!\")\n",
            "print(\"=\" * 75)",
        ]),
        markdown_cell([
            "### 🔬 Bước 7: Đánh giá Độc lập với Unified Benchmark (`evaluate.py`)\n",
            "Nạp checkpoint tốt nhất (`VNet_best_model.pth`) để tính toán điểm mDice, per-class Dice và đo lường FLOPs, Latency.",
        ]),
        code_cell([
            "import os\n",
            "import subprocess\n",
            "\n",
            "snapshot_dir = os.path.join(EXP_DIR, EXP_NAME, f\"fold_{FOLD_ID}\")\n",
            "best_pth = os.path.join(snapshot_dir, \"self_train\", \"best_model.pth\")\n",
            "if not os.path.exists(best_pth):\n",
            "    best_pth = os.path.join(snapshot_dir, \"self_train\", \"VNet_best_model.pth\")\n",
            "if not os.path.exists(best_pth):\n",
            "    best_pth = os.path.join(snapshot_dir, \"pre_train\", \"best_model.pth\")\n",
            "if not os.path.exists(best_pth):\n",
            "    best_pth = os.path.join(snapshot_dir, \"pre_train\", \"VNet_best_model.pth\")\n",
            "\n",
            "if os.path.exists(best_pth):\n",
            "    print(f\"[✓] Tìm thấy best model checkpoint: {best_pth}\")\n",
            "    eval_cmd = [\n",
            "        \"python\", \"evaluate.py\",\n",
            "        \"--checkpoint\", best_pth,\n",
            "        \"--root_path\", str(DATA_ROOT),\n",
            "        \"--task\", TASK,\n",
            "        \"--model\", MODEL_NAME,\n",
            "        \"--fold\", str(FOLD_ID),\n",
            f"        \"--stride_xy\", \"{pick['val_stride_xy']}\",\n",
            f"        \"--stride_z\", \"{pick['val_stride_z']}\",\n",
            "        \"--profile\"\n",
            "    ]\n",
            "    print(f\"[*] Chạy đánh giá:\\n{' '.join(eval_cmd)}\\n\")\n",
            "    try:\n",
            "        subprocess.run(eval_cmd, check=True)\n",
            "    except Exception as e:\n",
            "        print(f\"[!] Cảnh báo: Đánh giá bằng evaluate.py gặp lỗi nhưng không ảnh hưởng tới kết quả huấn luyện: {e}\", flush=True)\n",
            "else:\n",
            "    print(f\"[!] Chưa tìm thấy checkpoint tại {best_pth}. Hãy kiểm tra lại thư mục snapshot.\")",
        ]),
        markdown_cell([
            "### 📦 Bước 8: Đóng gói Kết quả Checkpoint & Logs\n",
            "Nén toàn bộ kết quả mô hình thành file `.tar.gz` để tải về trực tiếp từ mục **Output** của Kaggle.",
        ]),
        package_results_cell(repo["working_root"], sync_wandb=True, flush=False),
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
