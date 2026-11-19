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
    package_results_cell,
    pip_install_cell,
    resolve_method_setting,
    wandb_setup_cell,
)


def build_supervised_notebook(
    task: str = "multiclass",
    labelnum: int = 154,
    max_samples: int = 154,
    labeled_sampling: str = "first",
    setting: str = "opt1_nnunet",
):
    presets = load_notebook_presets()
    repo = presets["repository"]
    sup_defaults = presets["supervised"]["defaults"]
    spec = resolve_method_setting("supervised", setting, presets=presets, default_key="opt1_nnunet")

    model_name = spec["model_name"]
    batch_size = int(os.environ.get("BHSD_BATCH_SIZE") or spec["batch_size"])
    labeled_bs = int(os.environ.get("BHSD_LABELED_BS") or spec["labeled_bs"])
    max_iter = int(os.environ.get("BHSD_MAX_ITER") or spec["max_iter"])
    lr_max_iter = os.environ.get("BHSD_LR_MAX_ITER")
    base_lr_override = os.environ.get("BHSD_BASE_LR")
    use_sync_bn = os.environ.get("BHSD_USE_SYNC_BN", "").lower() in ("1", "true", "yes")
    single_gpu = os.environ.get("BHSD_SINGLE_GPU", "").lower() in ("1", "true", "yes")
    patch_size_override = os.environ.get("BHSD_PATCH_SIZE", "").strip()
    has_residual_override = os.environ.get("BHSD_HAS_RESIDUAL", "").strip().lower()
    has_dropout_override = os.environ.get("BHSD_HAS_DROPOUT", "").strip().lower()
    normalization_override = os.environ.get("BHSD_NORMALIZATION", "").strip()
    foreground_prob_override = os.environ.get("BHSD_FOREGROUND_PROB", "").strip()
    loss_type_override = os.environ.get("BHSD_LOSS_TYPE", "").strip().lower()
    anisotropic_kernels_override = os.environ.get("BHSD_ANISOTROPIC_KERNELS", "").strip().lower()
    n_filters_override = os.environ.get("BHSD_N_FILTERS", "").strip()
    batch_dice_override = os.environ.get("BHSD_BATCH_DICE", "").strip().lower()
    aug_preset_override = os.environ.get("BHSD_AUG_PRESET", "").strip().lower()
    extended_aug_override = os.environ.get("BHSD_EXTENDED_AUG", "").strip().lower()
    optimizer_override = os.environ.get("BHSD_OPTIMIZER", "").strip().lower()
    lr_scheduler_override = os.environ.get("BHSD_LR_SCHEDULER", "").strip().lower()
    warmup_iters_override = os.environ.get("BHSD_WARMUP_ITERS", "").strip()
    extra_overrides_raw = os.environ.get("BHSD_EXTRA_OVERRIDES", "").strip()
    val_stride_xy = spec["val_stride_xy"]
    val_stride_z = spec["val_stride_z"]
    patch_size_eval = list(spec["patch_size_eval"])
    config_file = spec["config_file"]
    arch_desc = spec["arch_desc"]
    extra_cli_lines = list(spec.get("extra_cli_lines", []))

    def _append_cli_pair(flag: str, value: str) -> None:
        extra_cli_lines.append(f"    {json.dumps(flag)}, {json.dumps(str(value))},\n")

    if patch_size_override:
        cleaned = patch_size_override.strip("[]() ")
        parts = [int(p.strip()) for p in cleaned.split(",") if p.strip()]
        if len(parts) != 3 or any(p <= 0 for p in parts):
            raise ValueError(f"BHSD_PATCH_SIZE must have 3 positive integers, got {patch_size_override!r}")
        pz, py, px = parts
        patch_size_eval = [str(pz), str(py), str(px)]
        val_stride_z = max(1, pz // 2)
        val_stride_xy = max(1, min(py, px) // 2)
        _append_cli_pair("--data.patch_size", f"[{pz},{py},{px}]")
    if has_residual_override in ("true", "false"):
        _append_cli_pair("--model.has_residual", has_residual_override)
    if has_dropout_override in ("true", "false"):
        _append_cli_pair("--model.has_dropout", has_dropout_override)
    if normalization_override:
        _append_cli_pair("--model.normalization", normalization_override)
    if foreground_prob_override:
        fp = float(foreground_prob_override)
        if not (0.0 <= fp <= 1.0):
            raise ValueError(f"BHSD_FOREGROUND_PROB must be in [0.0, 1.0], got {foreground_prob_override!r}")
        _append_cli_pair("--data.foreground_prob", fp)
    if loss_type_override:
        if loss_type_override not in ("ce_dice", "focal_tversky"):
            raise ValueError(f"BHSD_LOSS_TYPE must be 'ce_dice' or 'focal_tversky', got {loss_type_override!r}")
        _append_cli_pair("--method.loss_type", loss_type_override)
    if anisotropic_kernels_override in ("true", "false"):
        _append_cli_pair("--model.anisotropic_kernels", anisotropic_kernels_override)
    if n_filters_override:
        nf = int(n_filters_override)
        if nf <= 0:
            raise ValueError(f"BHSD_N_FILTERS must be positive, got {n_filters_override!r}")
        _append_cli_pair("--model.n_filters", nf)
    if batch_dice_override in ("true", "false"):
        _append_cli_pair("--method.batch_dice", batch_dice_override)
    if aug_preset_override:
        if aug_preset_override not in ("nnunet", "legacy"):
            raise ValueError(f"BHSD_AUG_PRESET must be 'nnunet' or 'legacy', got {aug_preset_override!r}")
        _append_cli_pair("--data.aug_preset", aug_preset_override)
    if extended_aug_override in ("true", "false"):
        _append_cli_pair("--data.extended_aug", extended_aug_override)
    if optimizer_override:
        if optimizer_override not in ("sgd", "adam", "adamw"):
            raise ValueError(f"BHSD_OPTIMIZER must be 'sgd', 'adam', or 'adamw', got {optimizer_override!r}")
        _append_cli_pair("--method.optimizer", optimizer_override)
    if lr_scheduler_override:
        if lr_scheduler_override not in ("poly", "cosine", "constant"):
            raise ValueError(f"BHSD_LR_SCHEDULER must be 'poly', 'cosine', or 'constant', got {lr_scheduler_override!r}")
        _append_cli_pair("--method.lr_scheduler", lr_scheduler_override)
    if warmup_iters_override:
        wi = int(warmup_iters_override)
        if wi < 0:
            raise ValueError(f"BHSD_WARMUP_ITERS must be >= 0, got {warmup_iters_override!r}")
        _append_cli_pair("--method.warmup_iters", wi)
    if lr_max_iter:
        _append_cli_pair("--lr_max_iteration", int(lr_max_iter))
    if base_lr_override:
        _append_cli_pair("--base_lr", float(base_lr_override))
    if use_sync_bn and not single_gpu:
        _append_cli_pair("--use_sync_bn", "true")
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
                f"LABELNUM = {int(labelnum)}\n",
                f"MAX_SAMPLES = {int(max_samples)}\n",
                f"LABELED_SAMPLING = \"{labeled_sampling}\"\n",
                f"PATCH_SIZE_EVAL = {patch_size_eval}\n",
                "\n",
                "RUN_TIMESTAMP = datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                f"EXP_NAME = f\"BHSD_{{TASK}}_Supervised_{{MODEL_NAME}}_fold{{FOLD_ID}}_{{RUN_TIMESTAMP}}\"\n",
                f"EXP_DIR = \"{exp_dir}\"\n",
                "\n",
                "num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0\n",
                f"FORCE_SINGLE_GPU = {single_gpu}\n",
                "use_ddp = (num_gpus >= 2) and (not FORCE_SINGLE_GPU)\n",
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
                "    gpu_arg = \"0\" if FORCE_SINGLE_GPU else \"auto\"\n",
                "    print(f\"[*] Chạy chế độ đơn GPU (--gpu {gpu_arg})...\", flush=True)\n",
                "    train_cmd = [\"python\", \"-u\", \"train.py\", \"--gpu\", gpu_arg] + base_args\n",
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


SETTING_NOTEBOOK_TARGETS = {
    "opt1_nnunet": "scripts/kaggle_bhsd_supervised_dual_gpu.ipynb",
    "vnet": "scripts/kaggle_bhsd_supervised_vnet_dual_gpu.ipynb",
    "vnet_v2": "scripts/kaggle_bhsd_supervised_vnet_v2_dual_gpu.ipynb",
    "vnet7_plain": "scripts/kaggle_bhsd_supervised_vnet7_plain_dual_gpu.ipynb",
    "vnet5_plain": "scripts/kaggle_bhsd_supervised_vnet5_plain_dual_gpu.ipynb",
    "vnet5_res": "scripts/kaggle_bhsd_supervised_vnet5_res_dual_gpu.ipynb",
}


def main():
    import argparse

    presets = load_notebook_presets()
    defaults = presets["supervised"]["defaults"]
    aliases = presets["supervised"].get("aliases", {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["multiclass", "binary"], default=os.environ.get("BHSD_TASK", "binary"))
    parser.add_argument(
        "--setting",
        default=os.environ.get("BHSD_SETTING", "all"),
        help="'opt1_nnunet', 'vnet', 'vnet_v2', 'vnet7_plain', 'vnet5_plain', 'vnet5_res', 'vnet_legacy', or 'all'",
    )
    parser.add_argument("--labelnum", type=int, default=int(os.environ.get("BHSD_LABELNUM") or defaults["labelnum"]))
    parser.add_argument("--max_samples", type=int, default=int(os.environ.get("BHSD_MAX_SAMPLES") or defaults["max_samples"]))
    parser.add_argument(
        "--labeled_sampling",
        choices=["first", "stratified"],
        default=os.environ.get("BHSD_LABELED_SAMPLING") or defaults.get("labeled_sampling", "first"),
    )
    parser.add_argument("--output", default=None)
    args, _ = parser.parse_known_args()

    os.makedirs("scripts", exist_ok=True)

    if args.setting in ("all", "default"):
        for setting_key, target_path in SETTING_NOTEBOOK_TARGETS.items():
            nb = build_supervised_notebook(
                task=args.task,
                labelnum=args.labelnum,
                max_samples=args.max_samples,
                labeled_sampling=args.labeled_sampling,
                setting=setting_key,
            )
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(
                f"Updated {target_path} (task={args.task}, setting={setting_key}, "
                f"labelnum={args.labelnum}, max_samples={args.max_samples}, "
                f"labeled_sampling={args.labeled_sampling}) successfully."
            )
    else:
        canonical = aliases.get(args.setting, args.setting)
        nb = build_supervised_notebook(
            task=args.task,
            labelnum=args.labelnum,
            max_samples=args.max_samples,
            labeled_sampling=args.labeled_sampling,
            setting=canonical,
        )
        if args.output:
            target = args.output
        elif canonical in SETTING_NOTEBOOK_TARGETS:
            target = SETTING_NOTEBOOK_TARGETS[canonical]
        elif canonical == "vnet_legacy":
            target = "scripts/kaggle_bhsd_supervised_vnet_legacy_dual_gpu.ipynb"
        else:
            target = "scripts/kaggle_bhsd_supervised_dual_gpu.ipynb"

        with open(target, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(
            f"Updated {target} (task={args.task}, setting={args.setting}, "
            f"labelnum={args.labelnum}, max_samples={args.max_samples}, "
            f"labeled_sampling={args.labeled_sampling}) successfully."
        )


if __name__ == "__main__":
    main()
