"""
Unified Kaggle Training Notebook Renderer for MedSSL-BHSD.
Replaces four divergent method-specific generator scripts with a single,
typed, preset-driven renderer supporting uniform CLI override forwarding
across supervised, mean_teacher, lefed, and pick.
"""

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

# Maps NotebookBuilder notebook_type keys -> (canonical method_key, default setting)
NOTEBOOK_TYPE_TO_METHOD: Dict[str, Tuple[str, str]] = {
    "mean_teacher": ("mean_teacher", "default"),
    "pick": ("pick", "default"),
    "lefed": ("lefed", "vnet5"),
    "supervised": ("supervised", "opt1_nnunet"),
    "supervised_vnet": ("supervised", "vnet"),
    "supervised_vnet_v2": ("supervised", "vnet_v2"),
    "supervised_vnet7_plain": ("supervised", "vnet7_plain"),
    "supervised_vnet5_plain": ("supervised", "vnet5_plain"),
    "supervised_vnet5_res": ("supervised", "vnet5_res"),
    "supervised_unet3d": ("supervised", "unet3d"),
    "supervised_nnunet_v2_auto": ("supervised", "nnunet_v2_auto"),
}

_METHOD_METADATA: Dict[str, Dict[str, Any]] = {
    "supervised": {
        "title": "# MedSSL-BHSD: Supervised Baseline on Kaggle GPU\n",
        "subtitle": (
            "Supervised 3D brain hemorrhage segmentation baseline on BHSD for fair comparison "
            "with nnU-Net and Semi-Supervised methods (Mean Teacher, PICK, CPS).\n"
        ),
        "locate_mode": "supervised",
        "with_magic_cd": False,
        "clone_timeout": None,
        "clone_non_interactive": False,
        "gpu_detailed": False,
        "verify_viewer": False,
        "step_headings": False,
        "timestamp_fmt": "%Y%m%d_%H%M",
        "ckpt_candidates": ("best_model.pth", "self_train/best_model.pth", "checkpoint_latest.pth"),
        "default_setting": "opt1_nnunet",
    },
    "mean_teacher": {
        "title": "# MedSSL-BHSD: Mean Teacher Semi-Supervised Segmentation on Dual GPU T4 (Kaggle)\n",
        "subtitle": (
            "3D brain hemorrhage segmentation on 2x T4 GPUs using **Mean Teacher "
            "(Tarvainen & Valpola, NeurIPS 2017)** optimized for BHSD.\n"
        ),
        "locate_mode": "mt",
        "with_magic_cd": True,
        "clone_timeout": 120,
        "clone_non_interactive": True,
        "gpu_detailed": True,
        "verify_viewer": True,
        "step_headings": True,
        "train_heading": "### Step 6: Train Mean Teacher End-to-End (DDP)",
        "timestamp_fmt": "%Y%m%d_%H%M%S",
        "ckpt_candidates": ("best_model.pth", "checkpoint_latest.pth"),
        "default_setting": "default",
    },
    "lefed": {
        "title": "# MedSSL-BHSD: LeFeD Semi-Supervised Segmentation on Dual GPU T4 (Kaggle)\n",
        "subtitle": (
            "3D brain hemorrhage segmentation on 2x T4 GPUs using **LeFeD "
            "(Learning Feedback Dual-Branch)** adapted for BHSD.\n"
        ),
        "locate_mode": "mt",
        "with_magic_cd": True,
        "clone_timeout": 120,
        "clone_non_interactive": True,
        "gpu_detailed": True,
        "verify_viewer": True,
        "step_headings": True,
        "train_heading": "### Step 6: Train LeFeD End-to-End (DDP)",
        "timestamp_fmt": "%Y%m%d_%H%M%S",
        "ckpt_candidates": ("best_model.pth", "checkpoint_latest.pth"),
        "default_setting": "vnet5",
    },
    "pick": {
        "title": "# MedSSL-BHSD: PICK Semi-Supervised Segmentation on Dual GPU T4 (Kaggle)\n",
        "subtitle": (
            "3D brain hemorrhage segmentation on 2x T4 GPUs using **PICK** "
            "within the unified MedSSL-BHSD framework.\n"
        ),
        "locate_mode": "pick",
        "with_magic_cd": True,
        "clone_timeout": None,
        "clone_non_interactive": False,
        "gpu_detailed": True,
        "verify_viewer": True,
        "step_headings": True,
        "train_heading": "### Step 6: Train PICK Multi-Class / Binary (DDP)",
        "timestamp_fmt": "%Y%m%d_%H%M%S",
        "ckpt_candidates": (
            "self_train/best_model.pth",
            "self_train/VNet_best_model.pth",
            "pre_train/best_model.pth",
            "pre_train/VNet_best_model.pth",
        ),
        "default_setting": "default",
    },
}


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _env_opt_str(name: str) -> Optional[str]:
    val = os.environ.get(name, "").strip()
    return val if val else None


def _env_opt_int(name: str) -> Optional[int]:
    val = _env_opt_str(name)
    return int(val) if val is not None else None


def _env_opt_float(name: str) -> Optional[float]:
    val = _env_opt_str(name)
    return float(val) if val is not None else None


def _parse_extra_overrides(raw: Any) -> List[Tuple[str, str]]:
    if not raw:
        return []
    items = json.loads(raw) if isinstance(raw, str) else list(raw)
    if not isinstance(items, list):
        raise ValueError("extra_overrides must be a list of [key, value] pairs or KEY=VALUE strings")
    result: List[Tuple[str, str]] = []
    for entry in items:
        if isinstance(entry, str):
            if "=" not in entry:
                raise ValueError(f"--override must be in KEY=VALUE format, got {entry!r}")
            k, v = entry.split("=", 1)
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            k, v = str(entry[0]), str(entry[1])
        else:
            raise ValueError(f"Invalid override entry: {entry!r}")
        k = k.strip().lstrip("-")
        v = v.strip()
        if not k or not re.fullmatch(r"[A-Za-z0-9_.-]+", k):
            raise ValueError(f"Invalid override key: {k!r}")
        result.append((k, v))
    return result


@dataclass
class NotebookRenderOptions:
    """Typed notebook generation options shared uniformly across all training methods."""

    task: Optional[str] = None
    setting: Optional[str] = None
    fold_id: Optional[int] = None
    labelnum: Optional[int] = None
    max_samples: Optional[int] = None
    labeled_sampling: Optional[str] = None
    max_iteration: Optional[int] = None
    lr_max_iteration: Optional[int] = None
    base_lr: Optional[float] = None
    batch_size: Optional[int] = None
    labeled_bs: Optional[int] = None
    patch_size: Optional[str] = None
    has_residual: Optional[str] = None
    has_dropout: Optional[str] = None
    normalization: Optional[str] = None
    foreground_prob: Optional[float] = None
    loss_type: Optional[str] = None
    anisotropic_kernels: Optional[str] = None
    n_filters: Optional[int] = None
    batch_dice: Optional[str] = None
    aug_preset: Optional[str] = None
    extended_aug: Optional[str] = None
    optimizer: Optional[str] = None
    lr_scheduler: Optional[str] = None
    warmup_iters: Optional[int] = None
    use_sync_bn: bool = False
    single_gpu: bool = False
    run_name: Optional[str] = None
    extra_overrides: List[Tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_env(
        cls,
        *,
        task: Optional[str] = None,
        setting: Optional[str] = None,
        labelnum: Optional[int] = None,
        max_samples: Optional[int] = None,
        labeled_sampling: Optional[str] = None,
    ) -> "NotebookRenderOptions":
        """Build options from explicit args overlaid with BHSD_* environment variables."""
        extra_raw = _env_opt_str("BHSD_EXTRA_OVERRIDES")
        return cls(
            task=task if task is not None else _env_opt_str("BHSD_TASK"),
            setting=setting if setting is not None else _env_opt_str("BHSD_SETTING"),
            labelnum=labelnum if labelnum is not None else _env_opt_int("BHSD_LABELNUM"),
            max_samples=max_samples if max_samples is not None else _env_opt_int("BHSD_MAX_SAMPLES"),
            labeled_sampling=(
                labeled_sampling if labeled_sampling is not None else _env_opt_str("BHSD_LABELED_SAMPLING")
            ),
            max_iteration=_env_opt_int("BHSD_MAX_ITER"),
            lr_max_iteration=_env_opt_int("BHSD_LR_MAX_ITER"),
            base_lr=_env_opt_float("BHSD_BASE_LR"),
            batch_size=_env_opt_int("BHSD_BATCH_SIZE"),
            labeled_bs=_env_opt_int("BHSD_LABELED_BS"),
            patch_size=_env_opt_str("BHSD_PATCH_SIZE"),
            has_residual=_env_opt_str("BHSD_HAS_RESIDUAL"),
            has_dropout=_env_opt_str("BHSD_HAS_DROPOUT"),
            normalization=_env_opt_str("BHSD_NORMALIZATION"),
            foreground_prob=_env_opt_float("BHSD_FOREGROUND_PROB"),
            loss_type=_env_opt_str("BHSD_LOSS_TYPE"),
            anisotropic_kernels=_env_opt_str("BHSD_ANISOTROPIC_KERNELS"),
            n_filters=_env_opt_int("BHSD_N_FILTERS"),
            batch_dice=_env_opt_str("BHSD_BATCH_DICE"),
            aug_preset=_env_opt_str("BHSD_AUG_PRESET"),
            extended_aug=_env_opt_str("BHSD_EXTENDED_AUG"),
            optimizer=_env_opt_str("BHSD_OPTIMIZER"),
            lr_scheduler=_env_opt_str("BHSD_LR_SCHEDULER"),
            warmup_iters=_env_opt_int("BHSD_WARMUP_ITERS"),
            use_sync_bn=_env_bool("BHSD_USE_SYNC_BN"),
            single_gpu=_env_bool("BHSD_SINGLE_GPU"),
            run_name=_env_opt_str("BHSD_RUN_NAME"),
            extra_overrides=_parse_extra_overrides(extra_raw),
        )

    @classmethod
    def from_cli_args(cls, args: Any, default_setting: Optional[str] = None) -> "NotebookRenderOptions":
        """Build typed options directly from kaggle_ctl.py argparse Namespace."""
        setting = getattr(args, "setting", None)
        if setting == "default" and default_setting and default_setting != "default":
            setting = default_setting
        return cls(
            task=getattr(args, "task", None),
            setting=setting,
            labelnum=getattr(args, "labelnum", None),
            max_samples=getattr(args, "max_samples", None),
            labeled_sampling=getattr(args, "labeled_sampling", None),
            max_iteration=getattr(args, "max_iteration", None),
            lr_max_iteration=getattr(args, "lr_max_iteration", None),
            base_lr=getattr(args, "base_lr", None),
            batch_size=getattr(args, "batch_size", None),
            labeled_bs=getattr(args, "labeled_bs", None),
            patch_size=getattr(args, "patch_size", None),
            has_residual=getattr(args, "has_residual", None),
            has_dropout=getattr(args, "has_dropout", None),
            normalization=getattr(args, "normalization", None),
            foreground_prob=getattr(args, "foreground_prob", None),
            loss_type=getattr(args, "loss_type", None),
            anisotropic_kernels=getattr(args, "anisotropic_kernels", None),
            n_filters=getattr(args, "n_filters", None),
            batch_dice=getattr(args, "batch_dice", None),
            aug_preset=getattr(args, "aug_preset", None),
            extended_aug=getattr(args, "extended_aug", None),
            optimizer=getattr(args, "optimizer", None),
            lr_scheduler=getattr(args, "lr_scheduler", None),
            warmup_iters=getattr(args, "warmup_iters", None),
            use_sync_bn=bool(getattr(args, "use_sync_bn", False)),
            single_gpu=bool(getattr(args, "single_gpu", False)),
            run_name=getattr(args, "run_name", None),
            extra_overrides=_parse_extra_overrides(getattr(args, "override", None)),
        )


def _validate_and_collect_overrides(
    options: NotebookRenderOptions,
    extra_cli_lines: List[str],
    patch_size_eval: Optional[List[str]],
) -> Tuple[Optional[List[str]], Optional[Tuple[int, int, int]]]:
    """Validate typed overrides and append corresponding train.py CLI flags uniformly."""

    def _append_cli_pair(flag: str, value: Any) -> None:
        extra_cli_lines.append(f"    {json.dumps(flag)}, {json.dumps(str(value))},\n")

    parsed_patch: Optional[Tuple[int, int, int]] = None
    if options.patch_size:
        cleaned = str(options.patch_size).strip("[]() ")
        parts = [int(p.strip()) for p in cleaned.split(",") if p.strip()]
        if len(parts) != 3 or any(p <= 0 for p in parts):
            raise ValueError(f"patch_size must have 3 positive integers, got {options.patch_size!r}")
        pz, py, px = parts
        parsed_patch = (pz, py, px)
        patch_size_eval = [str(pz), str(py), str(px)]
        _append_cli_pair("--data.patch_size", f"[{pz},{py},{px}]")

    if options.has_residual is not None:
        val = str(options.has_residual).strip().lower()
        if val in ("true", "false"):
            _append_cli_pair("--model.has_residual", val)
    if options.has_dropout is not None:
        val = str(options.has_dropout).strip().lower()
        if val in ("true", "false"):
            _append_cli_pair("--model.has_dropout", val)
    if options.normalization:
        _append_cli_pair("--model.normalization", str(options.normalization).strip())
    if options.foreground_prob is not None:
        fp = float(options.foreground_prob)
        if not (0.0 <= fp <= 1.0):
            raise ValueError(f"foreground_prob must be in [0.0, 1.0], got {options.foreground_prob!r}")
        _append_cli_pair("--data.foreground_prob", fp)
    if options.loss_type:
        lt = str(options.loss_type).strip().lower()
        if lt not in ("ce_dice", "focal_tversky"):
            raise ValueError(f"loss_type must be 'ce_dice' or 'focal_tversky', got {options.loss_type!r}")
        _append_cli_pair("--method.loss_type", lt)
    if options.anisotropic_kernels is not None:
        val = str(options.anisotropic_kernels).strip().lower()
        if val in ("true", "false"):
            _append_cli_pair("--model.anisotropic_kernels", val)
    if options.n_filters is not None:
        nf = int(options.n_filters)
        if nf <= 0:
            raise ValueError(f"n_filters must be positive, got {options.n_filters!r}")
        _append_cli_pair("--model.n_filters", nf)
    if options.batch_dice is not None:
        val = str(options.batch_dice).strip().lower()
        if val in ("true", "false"):
            _append_cli_pair("--method.batch_dice", val)
    if options.aug_preset:
        ap = str(options.aug_preset).strip().lower()
        if ap not in ("nnunet", "legacy"):
            raise ValueError(f"aug_preset must be 'nnunet' or 'legacy', got {options.aug_preset!r}")
        _append_cli_pair("--data.aug_preset", ap)
    if options.extended_aug is not None:
        val = str(options.extended_aug).strip().lower()
        if val in ("true", "false"):
            _append_cli_pair("--data.extended_aug", val)
    if options.optimizer:
        opt = str(options.optimizer).strip().lower()
        if opt not in ("sgd", "adam", "adamw"):
            raise ValueError(f"optimizer must be 'sgd', 'adam', or 'adamw', got {options.optimizer!r}")
        _append_cli_pair("--method.optimizer", opt)
    if options.lr_scheduler:
        sched = str(options.lr_scheduler).strip().lower()
        if sched not in ("poly", "cosine", "constant"):
            raise ValueError(f"lr_scheduler must be 'poly', 'cosine', or 'constant', got {options.lr_scheduler!r}")
        _append_cli_pair("--method.lr_scheduler", sched)
    if options.warmup_iters is not None:
        wi = int(options.warmup_iters)
        if wi < 0:
            raise ValueError(f"warmup_iters must be >= 0, got {options.warmup_iters!r}")
        _append_cli_pair("--method.warmup_iters", wi)
    if options.lr_max_iteration is not None:
        _append_cli_pair("--lr_max_iteration", int(options.lr_max_iteration))
    if options.base_lr is not None:
        _append_cli_pair("--base_lr", float(options.base_lr))
    if options.use_sync_bn and not options.single_gpu:
        _append_cli_pair("--use_sync_bn", "true")
    for k, v in options.extra_overrides:
        _append_cli_pair(f"--{k}", v)

    return patch_size_eval, parsed_patch


def _build_exp_name_line(
    method_key: str,
    canonical_setting: str,
    task: str,
    fold_id: int,
    labelnum: Optional[int],
    max_iter: Optional[int],
    batch_size: int,
    options: NotebookRenderOptions,
    parsed_patch: Optional[Tuple[int, int, int]],
) -> str:
    if options.run_name:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", options.run_name):
            raise ValueError(f"run_name contains invalid characters: {options.run_name!r}")
        return f"EXP_NAME = {json.dumps(options.run_name)}\n"

    if method_key == "supervised":
        task_tag = "bin" if task == "binary" else "mc"
        gpu_tag = "1gpu" if options.single_gpu else "ddp"
        name_parts = [canonical_setting, task_tag, f"f{fold_id}", gpu_tag]
        if options.has_residual and options.has_residual.lower() in ("true", "false"):
            name_parts.append("res" if options.has_residual.lower() == "true" else "nores")
        if options.has_dropout and options.has_dropout.lower() in ("true", "false"):
            name_parts.append("drop" if options.has_dropout.lower() == "true" else "nodrop")
        if options.normalization:
            norm_map = {"batchnorm": "bn", "instancenorm": "in", "groupnorm": "gn", "none": "nonorm"}
            name_parts.append(norm_map.get(options.normalization.lower(), options.normalization.lower()))
        if options.use_sync_bn and not options.single_gpu:
            name_parts.append("syncbn")
        if options.foreground_prob is not None:
            name_parts.append(f"fg{float(options.foreground_prob):g}")
        if options.loss_type:
            lt = options.loss_type.lower()
            name_parts.append("ftloss" if lt == "focal_tversky" else lt)
        if options.anisotropic_kernels and options.anisotropic_kernels.lower() in ("true", "false"):
            name_parts.append("aniso" if options.anisotropic_kernels.lower() == "true" else "iso")
        if options.n_filters is not None:
            name_parts.append(f"nf{int(options.n_filters)}")
        if options.batch_dice and options.batch_dice.lower() in ("true", "false"):
            name_parts.append("bdice" if options.batch_dice.lower() == "true" else "nobdice")
        if options.aug_preset:
            name_parts.append(f"aug{options.aug_preset.lower()}")
        if options.extended_aug and options.extended_aug.lower() in ("true", "false"):
            name_parts.append("extaug" if options.extended_aug.lower() == "true" else "noextaug")
        if options.optimizer:
            name_parts.append(options.optimizer.lower())
        if options.lr_scheduler:
            name_parts.append(options.lr_scheduler.lower())
        if options.warmup_iters is not None:
            name_parts.append(f"wu{int(options.warmup_iters)}")
        if parsed_patch is not None:
            pz, py, px = parsed_patch
            name_parts.append(f"p{pz}x{py}x{px}")
        if options.base_lr is not None:
            name_parts.append(f"lr{float(options.base_lr):g}")
        if options.batch_size is not None or options.labeled_bs is not None:
            name_parts.append(f"bs{batch_size}")
        if labelnum is not None and int(labelnum) < 154:
            name_parts.append(f"lbl{int(labelnum)}")
        if options.lr_max_iteration is not None:
            lrm = int(options.lr_max_iteration)
            name_parts.append(f"lrmax{lrm // 1000}k" if lrm % 1000 == 0 else f"lrmax{lrm}")
        if options.max_iteration is not None or options.lr_max_iteration is not None:
            name_parts.append(f"i{max_iter}")
        return f"EXP_NAME = f{json.dumps('_'.join(name_parts) + '_{RUN_TIMESTAMP}')}\n"

    if method_key == "lefed":
        task_tag = "bin" if task == "binary" else "mc"
        gpu_tag = "1gpu" if options.single_gpu else "ddp"
        name_parts = ["lefed", canonical_setting, task_tag, f"f{fold_id}", gpu_tag, f"lbl{int(labelnum)}", f"i{max_iter}"]
        return f"EXP_NAME = f{json.dumps('_'.join(name_parts) + '_{RUN_TIMESTAMP}')}\n"

    if method_key == "mean_teacher":
        return 'EXP_NAME = f"BHSD_{TASK}_MeanTeacher_{SETTING}_fold{FOLD_ID}_{RUN_TIMESTAMP}"\n'

    # pick
    return 'EXP_NAME = f"BHSD_{TASK}_PICK_dual_gpu_fold{FOLD_ID}"\n'


def render_training_notebook(
    method_key: str,
    options: Optional[NotebookRenderOptions] = None,
    presets: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Render a complete Kaggle training notebook (.ipynb dict) for any supported method
    ('supervised', 'mean_teacher', 'lefed', 'pick') with uniform override forwarding.
    """
    method_key = str(method_key).strip().lower()
    if method_key not in _METHOD_METADATA:
        raise ValueError(f"Unsupported notebook method '{method_key}'. Supported: {sorted(_METHOD_METADATA)}")

    presets = presets if presets is not None else load_notebook_presets()
    meta = _METHOD_METADATA[method_key]
    repo = presets["repository"]
    method_section = presets[method_key]
    defaults = method_section["defaults"]
    opts = options if options is not None else NotebookRenderOptions()

    task = opts.task or defaults.get("task", "multiclass")
    fold_id = opts.fold_id if opts.fold_id is not None else int(defaults.get("fold_id", 0))
    val_interval = int(defaults["val_interval"])
    save_interval = int(defaults["save_interval"])
    wandb_project = defaults["wandb_project"]
    exp_dir = repo["exp_dir"]

    if "settings" in method_section:
        setting_name = opts.setting or meta["default_setting"]
        spec = resolve_method_setting(method_key, setting_name, presets=presets, default_key=meta["default_setting"])
        aliases = method_section.get("aliases", {})
        canonical_setting = aliases.get(setting_name, setting_name)
    else:
        setting_name = "default"
        canonical_setting = "default"
        spec = defaults

    model_name = spec["model_name"]
    batch_size = int(opts.batch_size if opts.batch_size is not None else spec["batch_size"])
    labeled_bs = int(opts.labeled_bs if opts.labeled_bs is not None else spec["labeled_bs"])
    max_iter = opts.max_iteration if opts.max_iteration is not None else spec.get("max_iter")
    labelnum = opts.labelnum if opts.labelnum is not None else defaults.get("labelnum")
    max_samples = opts.max_samples if opts.max_samples is not None else defaults.get("max_samples")
    labeled_sampling = opts.labeled_sampling or defaults.get("labeled_sampling")
    config_file = spec["config_file"]
    arch_desc = spec.get("arch_desc", "")
    patch_size_eval = list(spec["patch_size_eval"]) if spec.get("patch_size_eval") is not None else None
    extra_cli_lines = list(spec.get("extra_cli_lines", []))

    patch_size_eval, parsed_patch = _validate_and_collect_overrides(opts, extra_cli_lines, patch_size_eval)
    exp_name_line = _build_exp_name_line(
        method_key=method_key,
        canonical_setting=canonical_setting,
        task=task,
        fold_id=fold_id,
        labelnum=labelnum,
        max_iter=max_iter,
        batch_size=batch_size,
        options=opts,
        parsed_patch=parsed_patch,
    )

    intro_lines = [meta["title"], "\n", meta["subtitle"], "\n"]
    if arch_desc:
        intro_lines.append(arch_desc)
    intro_lines.extend([
        "- **3D Data Augmentation**: nnU-Net style RandomScaling3D, GammaTransform, and GaussianBlur3D.\n",
        "- **Mixed Precision (AMP FP16)**: Automatic GradScaler optimization for T4 Tensor Cores.\n",
        "- **Multi-GPU DDP (`torchrun`)**: Distributed training with periodic 3D sliding-window validation.\n",
        "- **Cloud Sync**: Metrics & visuals on W&B Cloud; checkpoints stored on Hugging Face Hub.\n",
        "- **Model Profiling (`evaluate.py --profile_only`)**: Measures Params, GFLOPs, Latency, and Peak VRAM.",
    ])

    cells: List[Dict[str, Any]] = [markdown_cell(intro_lines)]
    if meta["step_headings"]:
        cells.append(markdown_cell(["### Step 1: Install Required Packages"]))
    cells.append(pip_install_cell(presets["packages"], quiet=True))
    if meta["step_headings"]:
        cells.append(markdown_cell(["### Step 2: Clone / Sync MedSSL-BHSD Repository"]))
    cells.append(
        clone_repo_cell(
            repo["repo_name"],
            repo["work_dir"],
            flush=True,
            with_magic_cd=meta["with_magic_cd"],
            timeout=meta["clone_timeout"],
            non_interactive=meta["clone_non_interactive"],
        )
    )
    if meta["step_headings"]:
        cells.append(markdown_cell(["### Step 3: Locate BHSD Dataset"]))
    cells.append(locate_dataset_cell(repo["input_root"], repo["working_root"], mode=meta["locate_mode"]))
    if meta["step_headings"]:
        cells.append(markdown_cell(["### Step 4: Verify GPU Hardware"]))
    cells.append(check_gpu_cell(flush=True, detailed=meta["gpu_detailed"]))
    if meta["step_headings"]:
        cells.append(markdown_cell(["### Step 5: Configure Hugging Face Storage & Weights & Biases (W&B)"]))
    cells.append(wandb_setup_cell(wandb_project, verify_viewer=meta["verify_viewer"]))
    if meta["step_headings"]:
        cells.append(markdown_cell([meta["train_heading"]]))

    # Build unified training cell with consistent anti-deadlock marker and single/DDP handling
    train_lines: List[str] = [
        "import os\n",
        "import sys\n",
        "import subprocess\n",
        "import datetime\n",
        "import socket\n",
        "import torch\n",
        "\n",
        f"FOLD_ID = {fold_id}\n",
        f'TASK = "{task}"\n',
        f'MODEL_NAME = "{model_name}"\n',
        f'METHOD_NAME = "{method_key}"\n',
        f'SETTING = "{setting_name}"\n',
        f'CONFIG_FILE = "{config_file}"\n',
        f"VAL_INTERVAL = {val_interval}\n",
        f"SAVE_INTERVAL = {save_interval}\n",
        f"BATCH_SIZE = {batch_size}\n",
        f"LABELED_BS = {labeled_bs}\n",
    ]
    if max_iter is not None:
        train_lines.append(f"MAX_ITER = {int(max_iter)}\n")
    if labelnum is not None:
        train_lines.append(f"LABELNUM = {int(labelnum)}\n")
    if max_samples is not None:
        train_lines.append(f"MAX_SAMPLES = {int(max_samples)}\n")
    if labeled_sampling is not None:
        train_lines.append(f'LABELED_SAMPLING = "{labeled_sampling}"\n')
    if patch_size_eval is not None:
        train_lines.append(f"PATCH_SIZE_EVAL = {patch_size_eval}\n")

    train_lines.extend([
        "\n",
        f'RUN_TIMESTAMP = datetime.datetime.now().strftime("{meta["timestamp_fmt"]}")\n',
        exp_name_line,
        f'EXP_DIR = "{exp_dir}"\n',
        "\n",
        "num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0\n",
        f"FORCE_SINGLE_GPU = {bool(opts.single_gpu)}\n",
        "use_ddp = (num_gpus >= 2) and (not FORCE_SINGLE_GPU)\n",
        "\n",
        "base_args = [\n",
        '    "--config", CONFIG_FILE,\n',
        '    "--fold", str(FOLD_ID),\n',
        '    "--task", TASK,\n',
        '    "--method", METHOD_NAME,\n',
        '    "--exp", EXP_NAME,\n',
        '    "--exp_dir", EXP_DIR,\n',
        '    "--root_path", str(DATA_ROOT),\n',
        '    "--batch_size", str(BATCH_SIZE),\n',
        '    "--labeled_bs", str(LABELED_BS),\n',
        '    "--val_interval", str(VAL_INTERVAL),\n',
        '    "--save_interval", str(SAVE_INTERVAL),\n',
        '    "--use_wandb",\n',
        '    "--save_wandb_model",\n',
        '    "--wandb_project", str(WANDB_PROJECT),\n',
        '    "--wandb_run_name", EXP_NAME,\n',
    ])
    if method_key != "pick":
        train_lines.append('    "--amp",\n')
    if method_key == "pick":
        train_lines.append('    "--model", MODEL_NAME,\n')
    if max_iter is not None:
        train_lines.append('    "--max_iteration", str(MAX_ITER),\n')
    if labelnum is not None:
        train_lines.append('    "--labelnum", str(LABELNUM),\n')
    if max_samples is not None:
        train_lines.append('    "--max_samples", str(MAX_SAMPLES),\n')
    if labeled_sampling is not None:
        train_lines.append('    "--labeled_sampling", str(LABELED_SAMPLING),\n')
    train_lines.extend(extra_cli_lines)
    train_lines.extend([
        "]\n",
        "\n",
        "# Cấu hình Anti-Deadlock cho NCCL trên 2x T4 PCIe\n",
        "def find_free_port():\n",
        "    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:\n",
        "        s.bind(('', 0))\n",
        "        return s.getsockname()[1]\n",
        "master_port = str(find_free_port())\n",
        "\n",
        "env = os.environ.copy()\n",
        'env["PYTHONUNBUFFERED"] = "1"\n',
        'env["OMP_NUM_THREADS"] = "1"\n',
        'env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\n',
        'env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"\n',
        'env["NCCL_DEBUG"] = "WARN"\n',
        'env["NCCL_IB_DISABLE"] = "1"\n',
        'env["NCCL_P2P_DISABLE"] = "1"\n',
        'env["TORCH_NCCL_ENABLE_MONITORING"] = "0"\n',
        'env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "120"\n',
        "\n",
        "if use_ddp:\n",
        '    print(f"[INFO] Launching DDP via torchrun on {num_gpus} GPUs (port={master_port})...", flush=True)\n',
        '    ddp_flag = [] if "--ddp" in base_args else ["--ddp"]\n',
        "    train_cmd = [\n",
        '        "torchrun",\n',
        '        f"--nproc_per_node={num_gpus}",\n',
        '        f"--master_port={master_port}",\n',
        '        "train.py",\n',
        "    ] + ddp_flag + base_args\n",
        "else:\n",
        '    gpu_arg = "0" if FORCE_SINGLE_GPU else "auto"\n',
        '    print(f"[INFO] Launching single-GPU training (--gpu {gpu_arg})...", flush=True)\n',
        '    train_cmd = ["python", "-u", "train.py", "--gpu", gpu_arg] + base_args\n',
        "\n",
        'print("=" * 75, flush=True)\n',
        'print(f"[INFO] Command:\\n{\' \'.join(train_cmd)}", flush=True)\n',
        'print("=" * 75, flush=True)\n',
        "\n",
        "process = subprocess.Popen(\n",
        "    train_cmd,\n",
        "    stdout=subprocess.PIPE,\n",
        "    stderr=subprocess.STDOUT,\n",
        "    text=True,\n",
        "    bufsize=1,\n",
        "    env=env,\n",
        ")\n",
        "for line in iter(process.stdout.readline, ''):\n",
        "    print(line, end='', flush=True)\n",
        "process.stdout.close()\n",
        "ret = process.wait()\n",
        "if ret != 0:\n",
        '    raise RuntimeError(f"[ERR] Training failed with exit code {ret}")\n',
        'print(f"[OK] Training completed successfully for {EXP_NAME}.", flush=True)\n',
    ])
    cells.append(code_cell(train_lines))

    if meta["step_headings"]:
        cells.append(
            markdown_cell(["### Step 7: Model Complexity & Efficiency Benchmark (`evaluate.py --profile_only`)"])
        )
    cells.append(
        profile_model_cell(
            patch_size_eval=patch_size_eval,
            checkpoint_candidates=meta["ckpt_candidates"],
        )
    )
    return build_notebook_envelope(cells, presets=presets)
