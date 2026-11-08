"""
Shared Kaggle Notebook Cell Builders & Preset Loaders.
Eliminates duplicate boilerplate across Mean Teacher, Supervised, and PICK
notebook generators while preserving exact resume markers and nbformat output.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRESETS_PATH = PROJECT_ROOT / "configs" / "kaggle" / "notebook_presets.json"


def load_notebook_presets(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load externalized Kaggle notebook presets and shared repository defaults."""
    target = Path(config_path) if config_path else DEFAULT_PRESETS_PATH
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_method_setting(
    method_key: str,
    setting: str,
    presets: Optional[Dict[str, Any]] = None,
    default_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Look up preset settings for a method with alias and default fallback resolution."""
    data = presets if presets is not None else load_notebook_presets()
    section = data[method_key]
    aliases = section.get("aliases", {})
    canonical = aliases.get(setting, setting)
    settings_map = section.get("settings", {})
    if canonical in settings_map:
        return settings_map[canonical]
    fallback = default_key or next(iter(settings_map))
    return settings_map[fallback]


def markdown_cell(source: Sequence[str]) -> Dict[str, Any]:
    """Create an nbformat v4 markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": list(source),
    }


def code_cell(source: Sequence[str]) -> Dict[str, Any]:
    """Create an nbformat v4 code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(source),
    }


def build_notebook_envelope(
    cells: List[Dict[str, Any]],
    presets: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wrap notebook cells with standardized Kaggle GPU kernelspec metadata."""
    data = presets if presets is not None else load_notebook_presets()
    return {
        "cells": cells,
        "metadata": copy.deepcopy(data["metadata"]),
        "nbformat": 4,
        "nbformat_minor": 2,
    }


def pip_install_cell(
    packages: Sequence[str],
    quiet: bool = False,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the dependency installation code cell."""
    pkg_str = " ".join(packages)
    if quiet:
        lines = [f"%pip install -q {pkg_str}"]
    else:
        lines = []
        if comment:
            lines.append(f"{comment}\n")
        lines.append(f"!pip install --progress-bar on {pkg_str}")
    return code_cell(lines)


def clone_repo_cell(
    repo_name: str,
    work_dir: str,
    *,
    flush: bool = True,
    with_magic_cd: bool = True,
    timeout: Optional[int] = 120,
    non_interactive: bool = True,
) -> Dict[str, Any]:
    """Build the GitHub clone / reset code cell shared across Kaggle notebooks."""
    flush_kw = ", flush=True" if flush else ""
    env_arg = ", env=env" if non_interactive else ""
    fetch_timeout = ", timeout=60" if timeout else ""
    clone_timeout = f", timeout={timeout}" if timeout else ""

    lines = [
        "import os\n",
    ]
    if with_magic_cd and non_interactive and flush:
        lines.append("import sys\n")
    lines.extend([
        "import subprocess\n",
        "\n",
        f"REPO_NAME = \"{repo_name}\"\n",
        "GITHUB_TOKEN = \"\"  # Điền token ghp_xxx nếu repo là Private, nếu Public thì để trống\n",
        f"WORK_DIR = \"{work_dir}\"\n",
        "\n",
    ])
    if non_interactive:
        lines.extend([
            "# Chống treo: Vô hiệu hóa prompt chờ nhập pass ngầm của Git trong non-interactive shell\n",
            "env = os.environ.copy()\n",
            "env[\"GIT_TERMINAL_PROMPT\"] = \"0\"\n",
            "\n",
        ])
    lines.extend([
        "if GITHUB_TOKEN.strip():\n",
        "    GITHUB_REPO_URL = f\"https://{GITHUB_TOKEN.strip()}@github.com/{REPO_NAME}.git\"\n",
        "else:\n",
        "    GITHUB_REPO_URL = f\"https://github.com/{REPO_NAME}.git\"\n",
        "\n",
        "if not os.path.exists(WORK_DIR):\n",
        f"    print(f\"[*] Đang clone repository {{REPO_NAME}}...\"{flush_kw})\n",
    ])
    if timeout:
        lines.extend([
            f"    res = subprocess.run([\"git\", \"clone\", GITHUB_REPO_URL, WORK_DIR]{env_arg}{clone_timeout})\n",
            "    if res.returncode != 0:\n",
            "        raise RuntimeError(f\"[X] Không thể clone {REPO_NAME}. Hãy kiểm tra mạng hoặc URL repo!\")\n",
        ])
    else:
        lines.append(f"    subprocess.run([\"git\", \"clone\", GITHUB_REPO_URL, WORK_DIR], check=True)\n")

    lines.extend([
        "else:\n",
        f"    print(f\"[*] Đồng bộ mã nguồn mới nhất từ GitHub tại {{WORK_DIR}}...\"{flush_kw})\n",
        f"    subprocess.run([\"git\", \"-C\", WORK_DIR, \"fetch\", \"origin\"]{env_arg}, check=True{fetch_timeout})\n",
        "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"checkout\", \".\"], check=True)\n",
        "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"clean\", \"-fd\"], check=True)\n",
        "    res = subprocess.run([\"git\", \"-C\", WORK_DIR, \"rev-parse\", \"--abbrev-ref\", \"HEAD\"], capture_output=True, text=True)\n",
        "    branch = res.stdout.strip() if res.returncode == 0 and res.stdout.strip() else \"main\"\n",
        "    subprocess.run([\"git\", \"-C\", WORK_DIR, \"reset\", \"--hard\", f\"origin/{branch}\"], check=True)\n",
        "\n",
    ])
    if with_magic_cd:
        lines.extend([
            "%cd {WORK_DIR}\n",
            "!git log -1 --oneline",
        ])
    else:
        lines.extend([
            "os.chdir(WORK_DIR)\n",
            "print(f\"[OK] Đã vào thư mục làm việc: {os.getcwd()}\", flush=True)",
        ])
    return code_cell(lines)


def wandb_setup_cell(wandb_project: str, verify_viewer: bool = True) -> Dict[str, Any]:
    """Build the Weights & Biases login & offline fallback code cell."""
    if not verify_viewer:
        return code_cell([
            "import wandb\n",
            "WANDB_API_KEY = \"\"\n",
            f"WANDB_PROJECT = \"{wandb_project}\"\n",
            "if WANDB_API_KEY.strip():\n",
            "    os.environ[\"WANDB_API_KEY\"] = WANDB_API_KEY.strip()\n",
            "    os.environ[\"WANDB_MODE\"] = \"online\"\n",
            "    wandb.login(key=WANDB_API_KEY.strip(), relogin=True)\n",
            "else:\n",
            "    os.environ[\"WANDB_MODE\"] = \"offline\"\n",
            "os.environ[\"WANDB_PROJECT\"] = WANDB_PROJECT",
        ])

    return code_cell([
        "import os\n",
        "import wandb\n",
        "\n",
        f"WANDB_PROJECT = '{wandb_project}'\n",
        "# 1. Key được truyền từ controller hoặc cấu hình thủ công:\n",
        "WANDB_API_KEY = \"\"\n",
        "\n",
        "key = str(WANDB_API_KEY).strip() if WANDB_API_KEY else \"\"\n",
        "if not key:\n",
        "    key = os.environ.get(\"WANDB_API_KEY\", \"\").strip()\n",
        "if not key:\n",
        "    try:\n",
        "        from kaggle_secrets import UserSecretsClient\n",
        "        user_secrets = UserSecretsClient()\n",
        "        for secret_name in [\"WANDB_API_KEY\", \"wandb_key\", \"wandb_api_key\", \"WANDB_KEY\", \"wandb\"]:\n",
        "            try:\n",
        "                val = user_secrets.get_secret(secret_name)\n",
        "                if val and str(val).strip():\n",
        "                    key = str(val).strip()\n",
        "                    print(f\"[*] Đã nạp W&B key từ Kaggle Secrets ({secret_name})\", flush=True)\n",
        "                    break\n",
        "            except Exception:\n",
        "                pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "if key:\n",
        "    os.environ[\"WANDB_API_KEY\"] = key\n",
        "    os.environ[\"WANDB_MODE\"] = \"online\"\n",
        "    try:\n",
        "        wandb.login(key=key, relogin=True)\n",
        "        _api = wandb.Api(api_key=key)\n",
        "        _viewer = _api.viewer\n",
        "        print(f\"[✓] Đã kết nối W&B Cloud thành công (User: {_viewer.username})! Mode: ONLINE | Dự án: {WANDB_PROJECT}\", flush=True)\n",
        "    except Exception as e:\n",
        "        print(f\"[!] Xác thực W&B gặp lỗi: {e}. Tự động chuyển sang OFFLINE mode (lưu kết quả cục bộ).\", flush=True)\n",
        "        os.environ[\"WANDB_MODE\"] = \"offline\"\n",
        "else:\n",
        "    os.environ[\"WANDB_MODE\"] = \"offline\"\n",
        "    print(\"[*] Không tìm thấy W&B key: Chạy chế độ OFFLINE (dữ liệu lưu cục bộ).\", flush=True)\n",
        "os.environ[\"WANDB_PROJECT\"] = WANDB_PROJECT\n",
    ])


def locate_dataset_cell(
    input_root: str,
    working_root: str,
    *,
    mode: str = "mt",
) -> Dict[str, Any]:
    """Build the Kaggle BHSD dataset discovery & integrity verification cell."""
    if mode == "supervised":
        return code_cell([
            "import glob\n",
            "\n",
            f"INPUT_DIR = \"{input_root}\"\n",
            f"WORKING_DIR = \"{working_root}\"\n",
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
            "print(f\"[SUCCESS] DATA_ROOT: {DATA_ROOT}\", flush=True)",
        ])

    flush_kw = ", flush=True" if mode == "mt" else ""
    import_sys = ["import sys\n"] if mode == "mt" else []
    if mode == "mt":
        finder_lines = [
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
        ]
    else:
        finder_lines = [
            "def find_bhsd_root(base_dir):\n",
            "    if not os.path.exists(base_dir):\n",
            "        return None\n",
            "    for root, dirs, files in os.walk(base_dir):\n",
            "        if \"data\" in dirs and (\"splits\" in dirs or \"fold0_train.txt\" in files):\n",
            "            return root\n",
            "    return None\n",
        ]

    return code_cell(
        ["import os\n"]
        + import_sys
        + [
            "import glob\n",
            "\n",
            f"INPUT_DIR = \"{input_root}\"\n",
            f"WORKING_DIR = \"{working_root}\"\n",
            "\n",
        ]
        + finder_lines
        + [
            "\n",
            f"print(\"[*] Đang định vị dataset BHSD...\"{flush_kw})\n",
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
            f"    print(f\"[*] Đang giải nén {{os.path.basename(tar_path)}} vào {{WORKING_DIR}} (khoảng 1-2 phút)...\"{flush_kw})\n",
            "    if tar_path.endswith(\".tar.gz\") or tar_path.endswith(\".tgz\"):\n",
            "        !tar -xzf \"{tar_path}\" -C {WORKING_DIR}\n",
            "    else:\n",
            "        !tar -xf \"{tar_path}\" -C {WORKING_DIR}\n",
            "    DATA_ROOT = find_bhsd_root(WORKING_DIR)\n",
            "    assert DATA_ROOT, f\"[X] Giải nén xong nhưng không tìm thấy data/ và splits/ trong {WORKING_DIR}!\"\n",
            "\n",
            f"print(f\"[SUCCESS] Đường dẫn DATA_ROOT: {{DATA_ROOT}}\"{flush_kw})\n",
            "\n",
            "# Kiểm tra tính toàn vẹn dữ liệu\n",
            "splits_dir = os.path.join(DATA_ROOT, \"splits\") if os.path.exists(os.path.join(DATA_ROOT, \"splits\")) else DATA_ROOT\n",
            "data_dir = os.path.join(DATA_ROOT, \"data\")\n",
            "h5_files = glob.glob(os.path.join(data_dir, \"*.h5\"))\n",
            f"print(f\"  [+] Tổng số file .h5: {{len(h5_files)}}\"{flush_kw})\n",
            "for f in range(5):\n",
            "    f_train = os.path.join(splits_dir, f\"fold{f}_train.txt\")\n",
            "    f_val = os.path.join(splits_dir, f\"fold{f}_val.txt\")\n",
            "    if os.path.exists(f_train) and os.path.exists(f_val):\n",
            "        n_tr = len([l for l in open(f_train) if l.strip()])\n",
            "        n_va = len([l for l in open(f_val) if l.strip()])\n",
            f"        print(f\"  [+] Fold {{f}}: {{n_tr}} train cases | {{n_va}} val cases\"{flush_kw})\n",
            f"print(\"[OK] Dữ liệu đã sẵn sàng!\"{flush_kw})",
        ]
    )


def check_gpu_cell(flush: bool = True, detailed: bool = True) -> Dict[str, Any]:
    """Build the CUDA / Dual-GPU readiness check cell."""
    if not detailed:
        return code_cell([
            "import torch\n",
            "assert torch.cuda.is_available(), \"[CẢNH BÁO] Chưa bật GPU! Hãy bật GPU T4 x 2 trong Settings.\"\n",
            "num_gpus = torch.cuda.device_count()\n",
            "print(f\"[GPU] Phát hiện {num_gpus} GPU khả dụng:\", flush=True)\n",
            "for i in range(num_gpus):\n",
            "    print(f\"  [+] GPU {i}: {torch.cuda.get_device_name(i)}\", flush=True)",
        ])

    flush_kw = ", flush=True" if flush else ""
    return code_cell([
        "import torch\n",
        "\n",
        f"print(\"=\" * 60{flush_kw})\n",
        "assert torch.cuda.is_available(), \"[CẢNH BÁO] Chưa bật GPU! Hãy chọn GPU T4 x 2 ở mục Settings.\"\n",
        "num_gpus = torch.cuda.device_count()\n",
        f"print(f\"[GPU] Phát hiện {{num_gpus}} GPU khả dụng:\"{flush_kw})\n",
        "for i in range(num_gpus):\n",
        "    name = torch.cuda.get_device_name(i)\n",
        "    vram = round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)\n",
        f"    print(f\"  [+] GPU {{i}}: {{name}} | VRAM: {{vram}} GB\"{flush_kw})\n",
        "\n",
        "if num_gpus >= 2:\n",
        f"    print(\"[*] Đã sẵn sàng chế độ ĐA GPU (2x T4 - DDP via torchrun) để tăng tốc và cân bằng VRAM!\"{flush_kw})\n",
        "else:\n",
        f"    print(\"[*] Chế độ đơn GPU (1x T4 hoặc P100) sẵn sàng.\"{flush_kw})\n",
        f"print(\"=\" * 60{flush_kw})",
    ])


def package_results_cell(working_root: str, sync_wandb: bool = True, flush: bool = True) -> Dict[str, Any]:
    """Build the archive packaging (and optional W&B offline sync) code cell."""
    if not sync_wandb:
        return code_cell([
            f"OUTPUT_ARCHIVE = f\"{working_root}/{{EXP_NAME}}_results.tar.gz\"\n",
            "if os.path.exists(EXP_DIR):\n",
            "    !tar -czf {OUTPUT_ARCHIVE} -C {EXP_DIR} .\n",
            "    print(f\"[✓] Đã tạo file kết quả {OUTPUT_ARCHIVE}\")",
        ])

    flush_kw = ", flush=True" if flush else ""
    return code_cell([
        "import os\n",
        "\n",
        "# Tự động đồng bộ W&B nếu có offline runs và đã cấu hình API key\n",
        "if os.environ.get('WANDB_API_KEY') and os.path.exists(f'{WORK_DIR}/wandb'):\n",
        "    print('[*] Đang kiểm tra và đồng bộ dữ liệu W&B lên Cloud...', flush=True)\n",
        "    !wandb sync {WORK_DIR}/wandb/offline-run-* 2>/dev/null || true\n",
        "\n",
        f"OUTPUT_ARCHIVE = f\"{working_root}/{{EXP_NAME}}_results.tar.gz\"\n",
        "\n",
        "if os.path.exists(EXP_DIR):\n",
        f"    print(f\"[*] Đang nén thư mục kết quả {{EXP_DIR}}...\"{flush_kw})\n",
        "    !tar -czf {OUTPUT_ARCHIVE} -C {EXP_DIR} .\n",
        "    size_mb = round(os.path.getsize(OUTPUT_ARCHIVE) / (1024**2), 2)\n",
        f"    print(f\"\\n[✓] Đã tạo thành công file lưu trữ: {{OUTPUT_ARCHIVE}} ({{size_mb}} MB)\"{flush_kw})\n",
        f"    print(\"[*] File đã sẵn sàng tải về tại mục Output bên phải màn hình.\"{flush_kw})\n",
        f"    print(f\"[*] Xem lại toàn bộ biểu đồ và so sánh: https://wandb.ai/{{WANDB_PROJECT}}\"{flush_kw})\n",
        "else:\n",
        f"    print(f\"[!] Thư mục {{EXP_DIR}} không tồn tại.\"{flush_kw})",
    ])
