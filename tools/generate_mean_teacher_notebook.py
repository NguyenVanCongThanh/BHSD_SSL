import json

def build_mean_teacher_notebook():
    cells = [
        # Cell 0: Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧠 MedSSL-BHSD: Mean Teacher Semi-Supervised Segmentation trên Dual GPU T4 (Kaggle)\n",
                "\n",
                "Phân đoạn 5 phân loại xuất huyết não chi tiết (EDH, IPH, IVH, SAH, SDH) + Background trên 2x GPU T4 Kaggle theo phương pháp **Mean Teacher (Tarvainen & Valpola, NeurIPS 2017)** đã được tối ưu hóa cho bài toán mất cân bằng dữ liệu 3D BHSD.\n",
                "\n",
                "- **Kiến trúc SOTA nnU-Net 3D (7 stages)**: Sử dụng mô hình nnU-Net 3D 7 stages chuẩn mực (`nnunet3d`) với InstanceNorm3d và LeakyReLU, kernel anisotropic (1, 3, 3) ở 2 tầng đầu bảo toàn chi tiết Z, downsampling tích chập có bước nhảy (strided convolution), độ sâu phân giải tối ưu cho patch (32, 192, 192) -> bottleneck 320 kênh (2, 3, 3).\n",
                "- **Data Augmentation 3D Mở Rộng**: Tự động áp dụng RandomScaling3D, GammaTransform và GaussianBlur3D chuẩn nnUNet v2 nhằm chống overfitting trên tập nhãn hạn chế.\n",
                "- **Automatic Mixed Precision (AMP FP16) & GradScaler**: Tự động tăng tốc huấn luyện lên ~2x qua Tensor Cores của 2x GPU T4, tiết kiệm ~40% VRAM và chống tràn số (underflow/overflow).\n",
                "- **Polynomial Learning Rate Decay**: Learning rate suy giảm đa thức theo từng bước ($lr = base\\_lr \\times (1 - iter/max\\_iter)^{0.9}$), cải thiện độ hội tụ và độ mượt mà của mDice.\n",
                "- **EMA Teacher & Balanced Consistency**: Trọng số Teacher cập nhật liên tục qua Exponential Moving Average ổn định (EMA decay 0.999); Consistency Loss cân bằng theo từng lớp (class-balanced channel MSE), downweight background (x0.05) và loại bỏ threshold cứng để chống triệt để sụp đổ mô hình (collapse).\n",
                "- **Tối ưu 2 GPU T4 (DDP qua `torchrun`)**: Tự động nhận diện 2x GPU T4, chia đều VRAM (~5.5GB/GPU), tốc độ cao hơn và tận dụng tối đa băng thông NVLink/PCIe.\n",
                "- **Sliding-Window Validation định kỳ trên Teacher**: Mỗi 200 iters tự động đánh giá mô hình Teacher bằng kỹ thuật cửa sổ trượt 3D (AMP-accelerated), trực quan hóa mẫu dự đoán tương tác trên W&B và lưu `best_model.pth`.\n",
                "- **Sử dụng trọn vẹn dữ liệu Fold 0**: Sử dụng đầy đủ 154 mẫu có nhãn của Fold 0 (`labelnum: 154`) cùng 1,978 mẫu không nhãn cho consistency regularization.\n",
                "- **Checkpoint kép & W&B Artifacts**: Lưu `checkpoint_latest.pth` mỗi 50 iters để chống timeout 12h, đồng thời tích hợp đồng bộ hóa W&B Cloud.\n",
                "- **Benchmark độc lập (`evaluate.py`)**: Tự động nạp `best_model.pth` để đo FLOPs, Latency, Params và per-class Dice chi tiết.\n",
                "- **Tự động đóng gói kết quả**: Nén toàn bộ log và checkpoint thành `.tar.gz` sẵn sàng tải về từ tab Output."
            ]
        },
        # Cell 1: Step 1 - Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📦 Bước 1: Cài đặt các thư viện cần thiết"
            ]
        },
        # Cell 2: Step 1 - Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cài đặt thư viện với thanh tiến trình rõ ràng (không dùng -q để theo dõi realtime, tránh cảm giác bị đơ)\n",
                "!pip install --progress-bar on medpy tensorboardX h5py SimpleITK nibabel wandb pyyaml"
            ]
        },
        # Cell 3: Step 2 - Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🚀 Bước 2: Clone & Cập nhật mã nguồn MedSSL-BHSD từ GitHub"
            ]
        },
        # Cell 4: Step 2 - Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import subprocess\n",
                "\n",
                "REPO_NAME = \"NguyenVanCongThanh/BHSD_SSL\"\n",
                "GITHUB_TOKEN = \"\"  # Điền token ghp_xxx nếu repo là Private, nếu Public thì để trống\n",
                "WORK_DIR = \"/kaggle/working/BHSD_SSL\"\n",
                "\n",
                "# Chống treo: Vô hiệu hóa prompt chờ nhập pass ngầm của Git trong non-interactive shell\n",
                "env = os.environ.copy()\n",
                "env[\"GIT_TERMINAL_PROMPT\"] = \"0\"\n",
                "\n",
                "if GITHUB_TOKEN.strip():\n",
                "    GITHUB_REPO_URL = f\"https://{GITHUB_TOKEN.strip()}@github.com/{REPO_NAME}.git\"\n",
                "else:\n",
                "    GITHUB_REPO_URL = f\"https://github.com/{REPO_NAME}.git\"\n",
                "\n",
                "if not os.path.exists(WORK_DIR):\n",
                "    print(f\"[*] Đang clone repository {REPO_NAME}...\", flush=True)\n",
                "    res = subprocess.run([\"git\", \"clone\", GITHUB_REPO_URL, WORK_DIR], env=env, timeout=120)\n",
                "    if res.returncode != 0:\n",
                "        raise RuntimeError(f\"[X] Không thể clone {REPO_NAME}. Hãy kiểm tra mạng hoặc URL repo!\")\n",
                "else:\n",
                "    print(f\"[*] Đồng bộ mã nguồn mới nhất từ GitHub tại {WORK_DIR}...\", flush=True)\n",
                "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"fetch\", \"origin\"], env=env, check=True, timeout=60)\n",
                "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"checkout\", \".\"], check=True)\n",
                "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"clean\", \"-fd\"], check=True)\n",
                "    res = subprocess.run([\"git\", \"-C\", WORK_DIR, \"rev-parse\", \"--abbrev-ref\", \"HEAD\"], capture_output=True, text=True)\n",
                "    branch = res.stdout.strip() if res.returncode == 0 and res.stdout.strip() else \"main\"\n",
                "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"reset\", \"--hard\", f\"origin/{branch}\"], check=True)\n",
                "\n",
                "%cd {WORK_DIR}\n",
                "!git log -1 --oneline"
            ]
        },
        # Cell 5: Step 3 - Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 Bước 3: Định vị Dataset BHSD từ Kaggle Input"
            ]
        },
        # Cell 6: Step 3 - Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import glob\n",
                "\n",
                "INPUT_DIR = \"/kaggle/input\"\n",
                "WORKING_DIR = \"/kaggle/working\"\n",
                "\n",
                "def find_bhsd_root(base_dir):\n",
                "    if not os.path.exists(base_dir):\n",
                "        return None\n",
                "    for root, dirs, files in os.walk(base_dir):\n",
                "        if \"data\" in dirs:\n",
                "            data_p = os.path.join(root, \"data\")\n",
                "            if os.path.exists(data_p) and any(f.endswith('.h5') for f in os.listdir(data_p)):\n",
                "                return root\n",
                "        if any(f.endswith('_train.txt') for f in files) and \"data\" in dirs:\n",
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
                "    \n",
                "    if not tar_path:\n",
                "        raise FileNotFoundError(f\"[X] Không tìm thấy dataset BHSD trong {INPUT_DIR}! Hãy bấm '+ Add Input'.\")\n",
                "    \n",
                "    print(f\"[*] Đang giải nén {os.path.basename(tar_path)} vào {WORKING_DIR} (khoảng 1-2 phút)...\", flush=True)\n",
                "    if tar_path.endswith(\".tar.gz\") or tar_path.endswith(\".tgz\"):\n",
                "        !tar -xzf \"{tar_path}\" -C {WORKING_DIR}\n",
                "    else:\n",
                "        !tar -xf \"{tar_path}\" -C {WORKING_DIR}\n",
                "    DATA_ROOT = find_bhsd_root(WORKING_DIR)\n",
                "    assert DATA_ROOT, f\"[X] Giải nén xong nhưng không tìm thấy data/ và splits/ trong {WORKING_DIR}!\"\n",
                "\n",
                "print(f\"[SUCCESS] Đường dẫn DATA_ROOT: {DATA_ROOT}\", flush=True)\n",
                "\n",
                "# Kiểm tra tính toàn vẹn dữ liệu\n",
                "splits_dir = os.path.join(DATA_ROOT, \"splits\") if os.path.exists(os.path.join(DATA_ROOT, \"splits\")) else DATA_ROOT\n",
                "data_dir = os.path.join(DATA_ROOT, \"data\")\n",
                "h5_files = glob.glob(os.path.join(data_dir, \"*.h5\"))\n",
                "print(f\"  [+] Tổng số file .h5: {len(h5_files)}\", flush=True)\n",
                "for f in range(5):\n",
                "    f_train = os.path.join(splits_dir, f\"fold{f}_train.txt\")\n",
                "    f_val = os.path.join(splits_dir, f\"fold{f}_val.txt\")\n",
                "    if os.path.exists(f_train) and os.path.exists(f_val):\n",
                "        n_tr = len([l for l in open(f_train) if l.strip()])\n",
                "        n_va = len([l for l in open(f_val) if l.strip()])\n",
                "        print(f\"  [+] Fold {f}: {n_tr} train cases | {n_va} val cases\", flush=True)\n",
                "print(\"[OK] Dữ liệu đã sẵn sàng!\", flush=True)"
            ]
        },
        # Cell 7: Step 4 - Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### ⚡ Bước 4: Kiểm tra Cấu hình Phần cứng Dual GPU (2x T4)"
            ]
        },
        # Cell 8: Step 4 - Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "\n",
                "print(\"=\" * 60, flush=True)\n",
                "assert torch.cuda.is_available(), \"[CẢNH BÁO] Chưa bật GPU! Hãy chọn GPU T4 x 2 ở mục Settings.\"\n",
                "num_gpus = torch.cuda.device_count()\n",
                "print(f\"[GPU] Phát hiện {num_gpus} GPU khả dụng:\", flush=True)\n",
                "for i in range(num_gpus):\n",
                "    name = torch.cuda.get_device_name(i)\n",
                "    vram = round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)\n",
                "    print(f\"  [+] GPU {i}: {name} | VRAM: {vram} GB\", flush=True)\n",
                "\n",
                "if num_gpus >= 2:\n",
                "    print(\"[*] Đã sẵn sàng chế độ ĐA GPU (2x T4 - DDP via torchrun) để tăng tốc và cân bằng VRAM!\", flush=True)\n",
                "else:\n",
                "    print(\"[*] Chế độ đơn GPU (1x T4 hoặc P100) sẵn sàng.\", flush=True)\n",
                "print(\"=\" * 60, flush=True)"
            ]
        },
        # Cell 9: Step 5 - Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📊 Bước 5: Đăng nhập & Cấu hình Weights & Biases (W&B)\n",
                "Theo dõi loss components (supervised, consistency, consistency_weight) và biểu đồ mDice trên W&B Cloud."
            ]
        },
        # Cell 10: Step 5 - Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
    "import os\n",
    "import wandb\n",
    "\n",
    "WANDB_PROJECT = 'BHSD_MeanTeacher'\n",
    "key = os.environ.get(\"WANDB_API_KEY\", \"\")\n",
    "if not key:\n",
    "    try:\n",
    "        from kaggle_secrets import UserSecretsClient\n",
    "        key = UserSecretsClient().get_secret(\"WANDB_API_KEY\")\n",
    "    except Exception:\n",
    "        key = \"\"\n",
    "if key:\n",
    "    os.environ[\"WANDB_API_KEY\"] = key\n",
    "    wandb.login(key=key)\n",
    "else:\n",
    "    os.environ[\"WANDB_MODE\"] = \"offline\"\n",
    "    print(\"No W&B secret configured: logging offline; sync output later.\")\n",
    "os.environ[\"WANDB_PROJECT\"] = WANDB_PROJECT\n"
]
        },
        # Cell 11: Step 6 - Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🏋️‍♂️ Bước 6: Huấn luyện Mean Teacher End-to-End (2x T4 DDP)\n",
                "- **Model**: `nnunet3d` (7 resolution stages với InstanceNorm3d & LeakyReLU, kernel anisotropic (1, 3, 3), downsampling có học, chuẩn nnU-Net v2 Option 1 & Spacing 0.5).\n",
                "- **Patch size**: [32, 192, 192] (bao quát trọn vẹn toàn bộ não trên mặt phẳng axial và đầy đủ lát cắt Z).\n",
                "- **Data Augmentation**: Tự động kích hoạt Scaling3D, GammaTransform, GaussianBlur3D chuẩn nnUNet v2.\n",
                "- **Tối ưu hóa**: Mixed Precision (FP16 qua AMP & GradScaler) + Polynomial LR Decay per iter.\n",
                "- **Batch size**: 4 (mỗi GPU T4 xử lý 2 mẫu: 1 labeled, 1 unlabeled, VRAM an toàn ~7.5GB / 15GB).\n",
                "- **Consistency**: Hybrid (50% Soft Dice foreground + 50% Class-balanced MSE), cosine EMA schedule (0.99 -> 0.999), downweight background."
            ]
        },
        # Cell 12: Step 6 - Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import subprocess\n",
                "import datetime\n",
                "import torch\n",
                "\n",
                "# --- THÔNG SỐ THÍ NGHIỆM ---\n",
                "FOLD_ID = 0                                                  # Chọn fold (0..4)\n",
                "TASK = \"multiclass\"                                          # \"multiclass\" (6 lớp) hoặc \"binary\" (2 lớp)\n",
                "MODEL_NAME = \"nnunet3d\"                                      # SOTA nnU-Net 3D\n",
                "METHOD_NAME = \"mean_teacher\"\n",
                "RUN_TIMESTAMP = datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                "EXP_NAME = f\"BHSD_{TASK}_MeanTeacher_fold{FOLD_ID}_{RUN_TIMESTAMP}\"  # Run mới độc lập mỗi lần chạy\n",
                "EXP_DIR = \"/kaggle/working/experiments\"\n",
                "CONFIG_FILE = \"configs/methods/mean_teacher.yaml\"\n",
                "MAX_ITER = 30000                                             # Tổng 30k iters (~5h trên 2x T4, an toàn trong 12h Kaggle)\n",
                "VAL_INTERVAL = 250                                           # Đánh giá validation trượt 3D mỗi 250 bước\n",
                "SAVE_INTERVAL = 50                                           # Lưu checkpoint_latest.pth mỗi 50 bước\n",
                "BATCH_SIZE = 4                                               # Batch size tổng (2 GPU -> 2 sample/GPU: 1 lab + 1 unlab)\n",
                "LABELED_BS = 2                                               # 2 mẫu có nhãn trong batch\n",
                "LABELNUM = 154                                               # Sử dụng full toàn bộ 154 mẫu có nhãn của fold 0\n",
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
                "    \"--batch_size\", str(BATCH_SIZE),\n",
                "    \"--labeled_bs\", str(LABELED_BS),\n",
                "    \"--max_iteration\", str(MAX_ITER),\n",
                "    \"--val_interval\", str(VAL_INTERVAL),\n",
                "    \"--save_interval\", str(SAVE_INTERVAL),\n",
                "    \"--use_wandb\",\n",
                "    \"--save_wandb_model\",\n",
                "    \"--wandb_project\", str(WANDB_PROJECT)\n",
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
                "env[\"NCCL_DEBUG\"] = \"WARN\"\n",
                "env[\"NCCL_IB_DISABLE\"] = \"1\"\n",
                "env[\"NCCL_P2P_DISABLE\"] = \"1\"\n",
                "env[\"TORCH_NCCL_ENABLE_MONITORING\"] = \"0\"  # Ngăn chặn watchdog false-positive abort khi garbage collector/GIL bận\n",
                "env[\"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\"] = \"1800\"\n",
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
                "print(\"=\" * 75, flush=True)"
            ]
        },
        # Cell 13: Step 7 - Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔬 Bước 7: Đánh giá Độc lập với Unified Benchmark (`evaluate.py`)\n",
                "Nạp checkpoint tốt nhất (`best_model.pth` chứa trọng số của EMA Teacher) để tính toán điểm mDice, per-class Dice và đo lường FLOPs, Latency."
            ]
        },
        # Cell 14: Step 7 - Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
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
                "        \"--patch_size\", \"32\", \"192\", \"192\",\n",
                "        \"--stride_xy\", \"32\",\n",
                "        \"--stride_z\", \"8\",\n",
                "        \"--use_tta\",\n",
                "        \"--profile\"\n",
                "    ]\n",
                "    print(f\"[*] Chạy đánh giá:\\n{' '.join(eval_cmd)}\\n\", flush=True)\n",
                "    subprocess.run(eval_cmd, check=True)\n",
                "else:\n",
                "    print(f\"[!] Chưa tìm thấy checkpoint tại {best_pth}. Hãy kiểm tra lại thư mục snapshot.\", flush=True)"
            ]
        },
        # Cell 15: Step 8 - Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📦 Bước 8: Đóng gói Kết quả Checkpoint & Logs\n",
                "Nén toàn bộ kết quả mô hình thành file `.tar.gz` để tải về trực tiếp từ mục **Output** của Kaggle."
            ]
        },
        # Cell 16: Step 8 - Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "\n",
                "OUTPUT_ARCHIVE = f\"/kaggle/working/{EXP_NAME}_results.tar.gz\"\n",
                "\n",
                "if os.path.exists(EXP_DIR):\n",
                "    print(f\"[*] Đang nén thư mục kết quả {EXP_DIR}...\", flush=True)\n",
                "    !tar -czf {OUTPUT_ARCHIVE} -C {EXP_DIR} .\n",
                "    size_mb = round(os.path.getsize(OUTPUT_ARCHIVE) / (1024**2), 2)\n",
                "    print(f\"\\n[✓] Đã tạo thành công file lưu trữ: {OUTPUT_ARCHIVE} ({size_mb} MB)\", flush=True)\n",
                "    print(\"[*] File đã sẵn sàng tải về tại mục Output bên phải màn hình.\", flush=True)\n",
                "    print(f\"[*] Xem lại toàn bộ biểu đồ và so sánh: https://wandb.ai/{WANDB_PROJECT}\", flush=True)\n",
                "else:\n",
                "    print(f\"[!] Thư mục {EXP_DIR} không tồn tại.\", flush=True)"
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

if __name__ == "__main__":
    nb = build_mean_teacher_notebook()
    targets = [
        "kaggle_mean_teacher/kaggle_bhsd_mean_teacher_dual_gpu.ipynb",
        "scripts/kaggle_bhsd_mean_teacher_dual_gpu.ipynb"
    ]
    for target in targets:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Generated {target} successfully.")
