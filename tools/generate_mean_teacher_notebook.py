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
            "# 🧠 MedSSL-BHSD: Mean Teacher Semi-Supervised Segmentation trên Dual GPU T4 (Kaggle)\n",
            "\n",
            "Phân đoạn xuất huyết não trên 2x GPU T4 Kaggle theo phương pháp **Mean Teacher (Tarvainen & Valpola, NeurIPS 2017)** đã được tối ưu hóa cho bài toán mất cân bằng dữ liệu 3D BHSD.\n",
            "\n",
            arch_desc,
            "- **Data Augmentation 3D Mở Rộng**: Tự động áp dụng RandomScaling3D, GammaTransform và GaussianBlur3D chuẩn nnUNet v2 nhằm chống overfitting trên tập nhãn hạn chế.\n",
            "- **Automatic Mixed Precision (AMP FP16) & GradScaler**: Tự động tăng tốc huấn luyện lên ~2x qua Tensor Cores của 2x GPU T4, tiết kiệm ~40% VRAM và chống tràn số (underflow/overflow).\n",
            "- **Polynomial Learning Rate Decay**: Learning rate suy giảm đa thức theo từng bước ($lr = base\\_lr \\times (1 - iter/max\\_iter)^{0.9}$), cải thiện độ hội tụ và độ mượt mà của mDice.\n",
            "- **EMA Teacher & Balanced Consistency**: Trọng số Teacher cập nhật liên tục qua Exponential Moving Average ổn định (EMA decay 0.999); Consistency Loss cân bằng theo từng lớp (class-balanced channel MSE), downweight background (x0.05) và loại bỏ threshold cứng để chống triệt để sụp đổ mô hình (collapse).\n",
            "- **Tối ưu 2 GPU T4 (DDP qua `torchrun`)**: Tự động nhận diện 2x GPU T4, chia đều VRAM (~5.5GB/GPU), tốc độ cao hơn và tận dụng tối đa băng thông NVLink/PCIe.\n",
            "- **Sliding-Window Validation định kỳ trên Teacher**: Mỗi 200 iters tự động đánh giá mô hình Teacher bằng kỹ thuật cửa sổ trượt 3D (AMP-accelerated), trực quan hóa mẫu dự đoán tương tác trên W&B và lưu `best_model.pth`.\n",
            f"- **Thiết lập mẫu có nhãn**: Sử dụng {labelnum} mẫu có nhãn cùng tập unlabeled cho consistency regularization.\n",
            "- **Checkpoint kép & W&B Artifacts**: Lưu `checkpoint_latest.pth` mỗi 50 iters để chống timeout 12h, đồng thời tích hợp đồng bộ hóa W&B Cloud.\n",
            "- **Benchmark độc lập (`evaluate.py`)**: Tự động nạp `best_model.pth` để đo FLOPs, Latency, Params và per-class Dice chi tiết.\n",
            "- **Tự động đóng gói kết quả**: Nén toàn bộ log và checkpoint thành `.tar.gz` sẵn sàng tải về từ tab Output.",
        ]),
        markdown_cell(["### 📦 Bước 1: Cài đặt các thư viện cần thiết"]),
        pip_install_cell(
            presets["packages"],
            quiet=False,
            comment="# Cài đặt thư viện với thanh tiến trình rõ ràng (không dùng -q để theo dõi realtime, tránh cảm giác bị đơ)",
        ),
        markdown_cell(["### 🚀 Bước 2: Clone & Cập nhật mã nguồn MedSSL-BHSD từ GitHub"]),
        clone_repo_cell(
            repo["repo_name"],
            repo["work_dir"],
            flush=True,
            with_magic_cd=True,
            timeout=120,
            non_interactive=True,
        ),
        markdown_cell(["### 🔍 Bước 3: Định vị Dataset BHSD từ Kaggle Input"]),
        locate_dataset_cell(repo["input_root"], repo["working_root"], mode="mt"),
        markdown_cell(["### ⚡ Bước 4: Kiểm tra Cấu hình Phần cứng Dual GPU (2x T4)"]),
        check_gpu_cell(flush=True, detailed=True),
        markdown_cell([
            "### 📊 Bước 5: Đăng nhập & Cấu hình Weights & Biases (W&B)\n",
            "Theo dõi loss components (supervised, consistency, consistency_weight) và biểu đồ mDice trên W&B Cloud.",
        ]),
        wandb_setup_cell(wandb_project, verify_viewer=True),
        markdown_cell([
            "### 🏋️‍♂️ Bước 6: Huấn luyện Mean Teacher End-to-End (2x T4 DDP)\n",
            "- **Model**: `nnunet3d` (7 resolution stages với InstanceNorm3d & LeakyReLU, kernel anisotropic (1, 3, 3), downsampling có học, chuẩn nnU-Net v2 Option 1 & Spacing 0.5).\n",
            "- **Patch size**: [28, 256, 256] (chuẩn 100% nnU-Net Option 1 CTClip).\n",
            "- **Data Augmentation**: Tự động kích hoạt Scaling3D, GammaTransform, GaussianBlur3D chuẩn nnUNet v2.\n",
            "- **Tối ưu hóa**: Mixed Precision (FP16 qua AMP & GradScaler) + Polynomial LR Decay per iter.\n",
            "- **Batch size**: 8 (mỗi GPU T4 xử lý 4 mẫu: 2 labeled, 2 unlabeled).\n",
            "- **Max iterations**: 12,000 (chạy thử nghiệm kiểm tra nhanh ~2h trên 2x T4).\n",
            "- **Consistency**: Hybrid (50% Soft Dice foreground + 50% Class-balanced MSE), cosine EMA schedule (0.99 -> 0.999), downweight background.",
        ]),
        code_cell(
            [
                "import os\n",
                "import sys\n",
                "import subprocess\n",
                "import datetime\n",
                "import torch\n",
                "\n",
                "# --- THÔNG SỐ THÍ NGHIỆM ---\n",
                f"FOLD_ID = {fold_id}                                                  # Chọn fold (0..4)\n",
                f"TASK = \"{task}\"                                          # \"multiclass\" (6 lớp) hoặc \"binary\" (2 lớp)\n",
                f"MODEL_NAME = \"{model_name}\"                                      # Backbone kiến trúc\n",
                "METHOD_NAME = \"mean_teacher\"\n",
                f"SETTING = \"{setting}\"                                            # Cấu hình: \"default\" hoặc \"swdl\"\n",
                "RUN_TIMESTAMP = datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                f"EXP_NAME = f\"BHSD_{{TASK}}_MeanTeacher_{{SETTING}}_fold{{FOLD_ID}}_{{RUN_TIMESTAMP}}\"  # Run mới độc lập mỗi lần chạy\n",
                f"EXP_DIR = \"{exp_dir}\"\n",
                f"CONFIG_FILE = \"{config_file}\"\n",
                f"MAX_ITER = {max_iter}                                             # iters chạy kiểm tra\n",
                f"VAL_INTERVAL = {val_interval}                                           # Đánh giá validation trượt 3D mỗi 200 bước\n",
                f"SAVE_INTERVAL = {save_interval}                                           # Lưu checkpoint_latest.pth mỗi 50 bước\n",
                f"BATCH_SIZE = {batch_size}                                               # Batch size tổng (DDP chia đều per-GPU)\n",
                f"LABELED_BS = {labeled_bs}                                               # Số mẫu có nhãn trong batch\n",
                f"LABELNUM = {int(labelnum)}                                               # Số lượng mẫu có nhãn (154 = 100%, 8 = 5% của 154 mẫu)\n",
                f"MAX_SAMPLES = {int(max_samples)}                                           # Giới hạn pool huấn luyện (2200 = full cả unlabeled, 154 = chỉ dùng tập 154 ca)\n",
                f"LABELED_SAMPLING = \"{labeled_sampling}\"                                 # \"first\" (mặc định) hoặc \"stratified\" (chia đều đủ 5 class)\n",
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
                "    # \"--resume\",  # Tắt auto-resume: mỗi lần submit là một Run độc lập mới trên W&B\n",
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
                "env[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"  # Chống phân mảnh bộ nhớ VRAM cho tensor 3D lớn\n",
                "env[\"NCCL_DEBUG\"] = \"WARN\"\n",
                "env[\"NCCL_IB_DISABLE\"] = \"1\"\n",
                "env[\"NCCL_P2P_DISABLE\"] = \"1\"\n",
                "env[\"TORCH_NCCL_ENABLE_MONITORING\"] = \"0\"  # Ngăn chặn watchdog false-positive abort khi garbage collector/GIL bận\n",
                "env[\"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\"] = \"120\"\n",
                "\n",
                "if use_ddp:\n",
                "    print(f\"[*] Phát hiện {num_gpus} GPU -> Khởi chạy chế độ DDP (torchrun) port={master_port}...\", flush=True)\n",
                "    train_cmd = [\n",
                "        \"torchrun\",\n",
                "        f\"--nproc_per_node={num_gpus}\",\n",
                "        f\"--master_port={master_port}\",\n",
                "        \"train.py\",\n",
                "        \"--ddp\"\n",
                "    ] + base_args\n",
                "else:\n",
                "    print(\"[*] Chế độ: Đơn GPU hoặc DataParallel (DP) qua python...\", flush=True)\n",
                "    train_cmd = [\n",
                "        \"python\", \"-u\", \"train.py\",\n",
                "        \"--gpu\", \"auto\"\n",
                "    ] + base_args\n",
                "\n",
                "print(\"=\" * 75, flush=True)\n",
                "print(f\"[*] Khởi chạy lệnh:\\n{' '.join(train_cmd)}\", flush=True)\n",
                "print(f\"[*] Theo dõi biểu đồ W&B Cloud tại: https://wandb.ai/{WANDB_PROJECT}\", flush=True)\n",
                "print(\"=\" * 75, flush=True)\n",
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
                "    print(line, end=\"\", flush=True)\n",
                "\n",
                "process.wait()\n",
                "\n",
                "print(\"\\n\" + \"=\" * 75, flush=True)\n",
                "if process.returncode != 0:\n",
                "    raise RuntimeError(f\"[LỖI] Huấn luyện kết thúc với mã lỗi {process.returncode}!\")\n",
                "else:\n",
                "    print(f\"[SUCCESS] Huấn luyện hoàn tất thành công cho {EXP_NAME}!\", flush=True)\n",
                "print(\"=\" * 75, flush=True)",
            ]
        ),
        markdown_cell([
            "### 🔬 Bước 7: Đánh giá Độc lập với Unified Benchmark (`evaluate.py`)\n",
            "Nạp checkpoint tốt nhất (`best_model.pth` chứa trọng số của EMA Teacher) để tính toán điểm mDice, per-class Dice và đo lường FLOPs, Latency.",
        ]),
        code_cell([
            "import os\n",
            "import subprocess\n",
            "\n",
            "snapshot_dir = os.path.join(EXP_DIR, EXP_NAME, f\"fold_{FOLD_ID}\")\n",
            "best_pth = os.path.join(snapshot_dir, \"best_model.pth\")\n",
            "if not os.path.exists(best_pth):\n",
            "    best_pth = os.path.join(snapshot_dir, \"checkpoint_latest.pth\")\n",
            "\n",
            "if os.path.exists(best_pth):\n",
            "    print(f\"[✓] Tìm thấy best model checkpoint (EMA Teacher): {best_pth}\", flush=True)\n",
            "    eval_cmd = [\n",
            "        \"python\", \"evaluate.py\",\n",
            "        \"--checkpoint\", best_pth,\n",
            "        \"--root_path\", str(DATA_ROOT),\n",
            "        \"--task\", TASK,\n",
            "        \"--model\", MODEL_NAME,\n",
            "        \"--fold\", str(FOLD_ID),\n",
            f"        \"--patch_size\", \"{patch_size_eval[0]}\", \"{patch_size_eval[1]}\", \"{patch_size_eval[2]}\",\n",
            f"        \"--stride_xy\", \"{val_stride_xy}\",\n",
            f"        \"--stride_z\", \"{val_stride_z}\",\n",
            "        \"--use_tta\",\n",
            "        \"--eval_space\", \"native\",\n",
            "        \"--profile\"\n",
            "    ]\n",
            "    print(f\"[*] Chạy đánh giá:\\n{' '.join(eval_cmd)}\\n\", flush=True)\n",
            "    try:\n",
            "        subprocess.run(eval_cmd, check=True)\n",
            "    except Exception as e:\n",
            "        print(f\"[!] Cảnh báo: Đánh giá bằng evaluate.py gặp lỗi nhưng không ảnh hưởng tới kết quả huấn luyện: {e}\", flush=True)\n",
            "else:\n",
            "    print(f\"[!] Chưa tìm thấy checkpoint tại {best_pth}. Hãy kiểm tra lại thư mục snapshot.\", flush=True)",
        ]),
        markdown_cell([
            "### 📦 Bước 8: Đóng gói Kết quả Checkpoint & Logs\n",
            "Nén toàn bộ kết quả mô hình thành file `.tar.gz` để tải về trực tiếp từ mục **Output** của Kaggle.",
        ]),
        package_results_cell(repo["working_root"], sync_wandb=True, flush=True),
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
