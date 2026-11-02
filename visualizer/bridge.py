"""
Codebase & Telemetry Bridge for the Research Method Visualization System.
Extracts real resolved YAML configurations via core.config_parser.load_config,
exact source code snippets with line numbers, and empirical run telemetry
from docs/analysis/.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config_parser import load_config


def extract_code_snippet(rel_path: str, start_line: int, end_line: int, title: str = "") -> Dict[str, Any]:
    """Read exact lines from a repository source file with 1-based line numbers."""
    abs_path = REPO_ROOT / rel_path
    if not abs_path.is_file():
        return {
            "path": rel_path,
            "abs_path": str(abs_path),
            "start_line": start_line,
            "end_line": end_line,
            "title": title or f"{rel_path}:{start_line}-{end_line}",
            "code": f"# File not found: {rel_path}",
        }
    lines = abs_path.read_text(encoding="utf-8").splitlines()
    s_idx = max(0, start_line - 1)
    e_idx = min(len(lines), end_line)
    numbered = [f"{i + 1:4d} | {lines[i]}" for i in range(s_idx, e_idx)]
    return {
        "path": rel_path,
        "abs_path": str(abs_path),
        "start_line": s_idx + 1,
        "end_line": e_idx,
        "title": title or f"{rel_path}:{s_idx + 1}-{e_idx}",
        "code": "\n".join(numbered),
    }


def load_raw_yaml(rel_path: str) -> Dict[str, Any]:
    abs_path = REPO_ROOT / rel_path
    if not abs_path.is_file():
        return {}
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config_presets() -> Dict[str, Any]:
    """Resolve actual repository configurations using core.config_parser.load_config."""
    mt_method_cfg = load_config(config_path=str(REPO_ROOT / "configs/methods/mean_teacher.yaml")).to_dict()
    mt_fold0_cfg = load_config(config_path=str(REPO_ROOT / "configs/experiments/bhsd_mt_nnunet_fold0.yaml")).to_dict()
    cps_cfg = load_config(config_path=str(REPO_ROOT / "configs/methods/cps.yaml")).to_dict()
    bcp_cfg = load_config(config_path=str(REPO_ROOT / "configs/methods/bcp.yaml")).to_dict()
    pick_cfg = load_config(config_path=str(REPO_ROOT / "configs/methods/pick.yaml")).to_dict()
    lefed_cfg = load_config(config_path=str(REPO_ROOT / "configs/methods/lefed.yaml")).to_dict()
    sup_cfg = load_config(config_path=str(REPO_ROOT / "configs/methods/supervised.yaml")).to_dict()

    raw_base = load_raw_yaml("configs/base.yaml")
    raw_dataset = load_raw_yaml("configs/datasets/bhsd_multiclass.yaml")
    raw_mt = load_raw_yaml("configs/methods/mean_teacher.yaml")
    raw_model_nnunet = load_raw_yaml("configs/models/nnunet3d.yaml")
    raw_exp_mt_fold0 = load_raw_yaml("configs/experiments/bhsd_mt_nnunet_fold0.yaml")

    return {
        "resolved": {
            "mean_teacher_default": mt_method_cfg,
            "mean_teacher_fold0_preset": mt_fold0_cfg,
            "cps_default": cps_cfg,
            "bcp_default": bcp_cfg,
            "pick_default": pick_cfg,
            "lefed_default": lefed_cfg,
            "supervised_default": sup_cfg,
        },
        "raw_tiers": {
            "tier1_base": {"path": "configs/base.yaml", "data": raw_base},
            "tier2_dataset": {"path": "configs/datasets/bhsd_multiclass.yaml", "data": raw_dataset},
            "tier3_method": {"path": "configs/methods/mean_teacher.yaml", "data": raw_mt},
            "tier4_model": {"path": "configs/models/nnunet3d.yaml", "data": raw_model_nnunet},
            "tier5_experiment": {"path": "configs/experiments/bhsd_mt_nnunet_fold0.yaml", "data": raw_exp_mt_fold0},
        },
    }


def load_empirical_runs() -> Dict[str, Any]:
    """Load real empirical run summaries and validation trajectories from docs/analysis/."""
    runs = {}
    analysis_dir = REPO_ROOT / "docs" / "analysis"

    for run_id, label, desc in [
        (
            "o6s4pex3",
            "Run o6s4pex3 (2026-10-30 | Patch 28×256×256 | Batch 8 [4L+4U] | 12k budget)",
            "Chạy với cấu hình tối ưu configs/methods/mean_teacher.yaml: nnUNet3D 7 stages, patch [28,256,256], global batch 8 (2L+2U mỗi GPU T4), consistency=0.05, rampup=4000 iters, rampdown=7800, cosine EMA 0.99->0.999. Dừng ở iter 8,600/12,000 (~11.89h).",
        ),
        (
            "9tvq8qld",
            "Run 9tvq8qld (2026-10-28 | Patch 32×192×192 | Batch 4 [2L+2U] | 30k budget)",
            "Chạy với patch [32,192,192], global batch 4 (1L+1U mỗi GPU T4), consistency=0.05, rampup=8000 iters, rampdown=19500, max_iteration=30,000. Dừng ở iter 16,250/30,000 (~11.93h).",
        ),
    ]:
        sum_path = analysis_dir / f"run_{run_id}_summary.json"
        csv_path = analysis_dir / f"run_{run_id}_validation.csv"
        summary = json.loads(sum_path.read_text(encoding="utf-8")) if sum_path.is_file() else {}
        val_rows: List[Dict[str, Any]] = []
        if csv_path.is_file():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val_rows.append({
                        "iteration": int(float(row["iteration"])),
                        "legacy_mdice": round(float(row["legacy_mdice"]), 5),
                        "positive_macro": round(float(row["positive_macro"]), 5),
                        "Epidural": round(float(row["Epidural"]), 5),
                        "Intraparenchymal": round(float(row["Intraparenchymal"]), 5),
                        "Intraventricular": round(float(row["Intraventricular"]), 5),
                        "Subarachnoid": round(float(row["Subarachnoid"]), 5),
                        "Subdural": round(float(row["Subdural"]), 5),
                    })
        runs[run_id] = {
            "id": run_id,
            "label": label,
            "description": desc,
            "summary": summary,
            "validation_trajectory": val_rows,
        }

    # Add detailed per-class table at final validation from verified markdown reports
    runs["o6s4pex3"]["class_breakdown_final"] = {
        "iteration": 8400,
        "total_val_cases": 38,
        "classes": [
            {"name": "Epidural (EDH)", "pos_cases": 6, "neg_cases": 32, "pos_dice": 0.0000, "case_mean_dice": 0.8421, "precision": 0.0000, "recall": 0.0000, "fp_neg_cases": 0, "fp_ml_mean": 0.000},
            {"name": "Intraparenchymal (IPH)", "pos_cases": 23, "neg_cases": 15, "pos_dice": 0.5395, "case_mean_dice": 0.3791, "precision": 0.5611, "recall": 0.7912, "fp_neg_cases": 13, "fp_ml_mean": 2.778},
            {"name": "Intraventricular (IVH)", "pos_cases": 22, "neg_cases": 16, "pos_dice": 0.4872, "case_mean_dice": 0.3347, "precision": 0.6374, "recall": 0.5566, "fp_neg_cases": 14, "fp_ml_mean": 0.293},
            {"name": "Subarachnoid (SAH)", "pos_cases": 20, "neg_cases": 18, "pos_dice": 0.2559, "case_mean_dice": 0.1347, "precision": 0.3247, "recall": 0.2518, "fp_neg_cases": 18, "fp_ml_mean": 1.905},
            {"name": "Subdural (SDH)", "pos_cases": 13, "neg_cases": 25, "pos_dice": 0.3453, "case_mean_dice": 0.1181, "precision": 0.4074, "recall": 0.4286, "fp_neg_cases": 25, "fp_ml_mean": 4.072},
        ],
        "loss_windows": [
            {"window": "1 – 2,000", "sup_mean": 0.53437, "cons_raw_mean": 0.08668, "weighted_ratio_pct": 0.0476},
            {"window": "2,001 – 4,000", "sup_mean": 0.48121, "cons_raw_mean": 0.04684, "weighted_ratio_pct": 0.3496},
            {"window": "4,001 – 6,000", "sup_mean": 0.46325, "cons_raw_mean": 0.04595, "weighted_ratio_pct": 0.4951},
            {"window": "6,001 – 8,000", "sup_mean": 0.45071, "cons_raw_mean": 0.04303, "weighted_ratio_pct": 0.4795},
            {"window": "8,001 – 8,600", "sup_mean": 0.44601, "cons_raw_mean": 0.04809, "weighted_ratio_pct": 0.5263},
        ],
    }

    runs["9tvq8qld"]["class_breakdown_final"] = {
        "iteration": 16000,
        "total_val_cases": 38,
        "classes": [
            {"name": "Epidural (EDH)", "pos_cases": 6, "neg_cases": 32, "pos_dice": 0.0000, "case_mean_dice": 0.8421, "precision": 0.0000, "recall": 0.0000, "fp_neg_cases": 0, "fp_ml_mean": 0.000},
            {"name": "Intraparenchymal (IPH)", "pos_cases": 23, "neg_cases": 15, "pos_dice": 0.5432, "case_mean_dice": 0.3814, "precision": 0.5787, "recall": 0.7636, "fp_neg_cases": 13, "fp_ml_mean": 2.651},
            {"name": "Intraventricular (IVH)", "pos_cases": 22, "neg_cases": 16, "pos_dice": 0.5136, "case_mean_dice": 0.3237, "precision": 0.6548, "recall": 0.5841, "fp_neg_cases": 15, "fp_ml_mean": 0.322},
            {"name": "Subarachnoid (SAH)", "pos_cases": 20, "neg_cases": 18, "pos_dice": 0.2810, "case_mean_dice": 0.1479, "precision": 0.3775, "recall": 0.2921, "fp_neg_cases": 18, "fp_ml_mean": 2.055},
            {"name": "Subdural (SDH)", "pos_cases": 13, "neg_cases": 25, "pos_dice": 0.3393, "case_mean_dice": 0.1161, "precision": 0.3469, "recall": 0.4431, "fp_neg_cases": 25, "fp_ml_mean": 5.645},
        ],
        "loss_windows": [
            {"window": "1 – 2,000", "sup_mean": 0.54236, "cons_raw_mean": 0.08958, "weighted_ratio_pct": 0.0164},
            {"window": "2,001 – 4,000", "sup_mean": 0.50687, "cons_raw_mean": 0.06126, "weighted_ratio_pct": 0.0947},
            {"window": "6,001 – 8,000", "sup_mean": 0.47044, "cons_raw_mean": 0.05949, "weighted_ratio_pct": 0.5682},
            {"window": "8,001 – 10,000", "sup_mean": 0.46470, "cons_raw_mean": 0.06409, "weighted_ratio_pct": 0.6901},
            {"window": "12,001 – 14,000", "sup_mean": 0.45319, "cons_raw_mean": 0.06184, "weighted_ratio_pct": 0.6829},
            {"window": "14,001 – 16,250", "sup_mean": 0.44775, "cons_raw_mean": 0.06465, "weighted_ratio_pct": 0.7225},
        ],
    }

    return runs


def collect_code_snippets() -> Dict[str, Dict[str, Any]]:
    """Extract key source code snippets across the repository with exact line ranges."""
    return {
        "mt_init": extract_code_snippet("methods/mean_teacher.py", 37, 94, "MeanTeacherMethod.__init__ (Teacher Clone, Detach & Loss Init)"),
        "mt_consistency_schedule": extract_code_snippet("methods/mean_teacher.py", 95, 112, "MeanTeacherMethod.get_current_consistency_weight (Ramp-up & Cosine Ramp-down)"),
        "mt_train_step": extract_code_snippet("methods/mean_teacher.py", 113, 191, "MeanTeacherMethod.train_step (Teacher No-Grad, Perturbation, Multi-Scale Sup & Hybrid Cons)"),
        "mt_ema_hook": extract_code_snippet("methods/mean_teacher.py", 192, 197, "MeanTeacherMethod.after_optimizer_step (Triggering EMA Update)"),
        "mt_validate": extract_code_snippet("methods/mean_teacher.py", 231, 233, "MeanTeacherMethod.validate (Evaluating EMA Teacher)"),
        "ema_update_fn": extract_code_snippet("methods/ssl_utils.py", 93, 121, "update_ema_variables (Cosine Alpha Schedule & Buffer Copy)"),
        "consistency_loss": extract_code_snippet("losses/consistency.py", 14, 136, "ConsistencyLoss & compute_soft_dice_consistency (Hybrid Soft Dice + Class-Balanced MSE)"),
        "mask_dice_loss": extract_code_snippet("losses/dice.py", 60, 101, "MaskDiceLoss (Foreground-only Reduction when include_background=False)"),
        "deep_supervision": extract_code_snippet("losses/deep_supervision.py", 13, 35, "compute_ds_weights & resolve_ds_weights (Exponential Decay Normalization)"),
        "nnunet_arch": extract_code_snippet("models/backbones/nnunet3d.py", 40, 271, "nnUNet3D (7-Stage Anisotropic Architecture & Deep Supervision Heads)"),
        "gradient_handler": extract_code_snippet("training/gradient_handler.py", 39, 111, "GradientHandler.apply_gradients (AMP Unscale, DDP Finite Consensus, Clip 12.0 & after_step_fn)"),
        "batch_layout": extract_code_snippet("core/batching.py", 20, 43, "resolve_batch_layout (Global-to-Local DDP Partitioning)"),
        "two_stream_sampler": extract_code_snippet("data/samplers.py", 34, 75, "TwoStreamBatchSampler (Primary Labeled + Secondary Eternal Unlabeled)"),
        "anti_leakage_crop": extract_code_snippet("data/dataset_bhsd.py", 218, 236, "BHSDDataset.__getitem__ (Zeroing Unlabeled Mask Before Foreground-Oversampled RandomCrop)"),
        "random_crop_fg": extract_code_snippet("data/transforms3d.py", 30, 88, "RandomCrop (33% Foreground Oversampling with Jitter)"),
        "lr_scheduler": extract_code_snippet("core/optimizers.py", 64, 96, "calculate_lr (Polynomial & Cosine Annealing LR with Warmup)"),
        "cps_train_step": extract_code_snippet("methods/cps.py", 90, 140, "CPSMethod.train_step (Dual-Network Cross Pseudo Supervision)"),
        "bcp_train_step": extract_code_snippet("methods/bcp.py", 33, 79, "BCPMethod.train_step (Bidirectional 3D Copy-Paste)"),
        "pick_train_step": extract_code_snippet("methods/pick.py", 53, 166, "PICKMethod.pretrain_step & selftrain_step (2-Phase CutMix + Lesion MIM + Aux)"),
        "lefed_train_step": extract_code_snippet("methods/lefed.py", 50, 136, "LeFeDMethod._feedback_steps & optimize_batch (3-Step Feature Feedback Loop)"),
        "supervised_train_step": extract_code_snippet("methods/supervised.py", 26, 41, "SupervisedMethod.train_step (Labeled-Only Lower Bound)"),
    }
