"""
Codebase & Telemetry Bridge for the Research Method Visualization System.
Extracts real resolved YAML configurations via core.config_parser.load_config,
AST-indexed source code snippets with line numbers, and empirical run telemetry
from docs/analysis/.
"""

import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config_parser import load_config


RESOLVED_PRESET_PATHS: Dict[str, str] = {
    "mean_teacher_default": "configs/methods/mean_teacher.yaml",
    "mean_teacher_fold0_preset": "configs/experiments/bhsd_mt_nnunet_fold0.yaml",
    "cps_default": "configs/methods/cps.yaml",
    "bcp_default": "configs/methods/bcp.yaml",
    "pick_default": "configs/methods/pick.yaml",
    "lefed_default": "configs/methods/lefed.yaml",
    "supervised_default": "configs/methods/supervised.yaml",
}

RAW_TIER_PATHS: Dict[str, str] = {
    "tier1_base": "configs/base.yaml",
    "tier2_dataset": "configs/datasets/bhsd_multiclass.yaml",
    "tier3_method": "configs/methods/mean_teacher.yaml",
    "tier4_model": "configs/models/nnunet3d.yaml",
    "tier5_experiment": "configs/experiments/bhsd_mt_nnunet_fold0.yaml",
}

EMPIRICAL_CATALOG_PATH = REPO_ROOT / "docs" / "analysis" / "empirical_runs.json"

CODE_SNIPPET_SPECS: Dict[str, Tuple[str, Tuple[str, ...], str]] = {
    "mt_init": (
        "methods/mean_teacher.py",
        ("MeanTeacherMethod.__init__",),
        "MeanTeacherMethod.__init__ (Teacher Clone, Detach & Loss Init)",
    ),
    "mt_consistency_schedule": (
        "methods/mean_teacher.py",
        ("MeanTeacherMethod.get_current_consistency_weight",),
        "MeanTeacherMethod.get_current_consistency_weight (Ramp-up & Cosine Ramp-down)",
    ),
    "mt_train_step": (
        "methods/mean_teacher.py",
        ("MeanTeacherMethod.train_step",),
        "MeanTeacherMethod.train_step (Teacher No-Grad, Perturbation, Multi-Scale Sup & Hybrid Cons)",
    ),
    "mt_ema_hook": (
        "methods/mean_teacher.py",
        ("MeanTeacherMethod.after_optimizer_step",),
        "MeanTeacherMethod.after_optimizer_step (Triggering EMA Update)",
    ),
    "mt_validate": (
        "methods/mean_teacher.py",
        ("MeanTeacherMethod.validate",),
        "MeanTeacherMethod.validate (Evaluating EMA Teacher)",
    ),
    "ema_update_fn": (
        "methods/ssl_utils.py",
        ("update_ema_variables",),
        "update_ema_variables (Cosine Alpha Schedule & Buffer Copy)",
    ),
    "consistency_loss": (
        "losses/consistency.py",
        ("compute_soft_dice_consistency", "ConsistencyLoss"),
        "ConsistencyLoss & compute_soft_dice_consistency (Hybrid Soft Dice + Class-Balanced MSE)",
    ),
    "mask_dice_loss": (
        "losses/dice.py",
        ("MaskDiceLoss",),
        "MaskDiceLoss (Foreground-only Reduction when include_background=False)",
    ),
    "deep_supervision": (
        "losses/deep_supervision.py",
        ("compute_ds_weights", "resolve_ds_weights"),
        "compute_ds_weights & resolve_ds_weights (Exponential Decay Normalization)",
    ),
    "nnunet_arch": (
        "models/backbones/nnunet3d.py",
        ("nnUNet3D",),
        "nnUNet3D (7-Stage Anisotropic Architecture & Deep Supervision Heads)",
    ),
    "gradient_handler": (
        "training/gradient_handler.py",
        ("GradientHandler.apply_gradients",),
        "GradientHandler.apply_gradients (AMP Unscale, DDP Finite Consensus, Clip & after_step_fn)",
    ),
    "batch_layout": (
        "core/batching.py",
        ("resolve_batch_layout",),
        "resolve_batch_layout (Global-to-Local DDP Partitioning)",
    ),
    "two_stream_sampler": (
        "data/samplers.py",
        ("TwoStreamBatchSampler",),
        "TwoStreamBatchSampler (Primary Labeled + Secondary Eternal Unlabeled)",
    ),
    "anti_leakage_crop": (
        "data/dataset_bhsd.py",
        ("BHSDDataset.__getitem__",),
        "BHSDDataset.__getitem__ (Zeroing Unlabeled Mask Before Foreground-Oversampled RandomCrop)",
    ),
    "random_crop_fg": (
        "data/transforms3d.py",
        ("RandomCrop",),
        "RandomCrop (Foreground Oversampling with Jitter)",
    ),
    "lr_scheduler": (
        "core/optimizers.py",
        ("calculate_lr",),
        "calculate_lr (Polynomial & Cosine Annealing LR with Warmup)",
    ),
    "cps_train_step": (
        "methods/cps.py",
        ("CPSMethod.train_step",),
        "CPSMethod.train_step (Dual-Network Cross Pseudo Supervision)",
    ),
    "bcp_train_step": (
        "methods/bcp.py",
        ("BCPMethod.train_step",),
        "BCPMethod.train_step (Bidirectional 3D Copy-Paste)",
    ),
    "pick_train_step": (
        "methods/pick.py",
        ("PICKMethod.pretrain_step", "PICKMethod.selftrain_step"),
        "PICKMethod.pretrain_step & selftrain_step (2-Phase CutMix + Lesion MIM + Aux)",
    ),
    "lefed_train_step": (
        "methods/lefed.py",
        ("LeFeDMethod._feedback_steps", "LeFeDMethod.optimize_batch"),
        "LeFeDMethod._feedback_steps & optimize_batch (3-Step Feature Feedback Loop)",
    ),
    "supervised_train_step": (
        "methods/supervised.py",
        ("SupervisedMethod.train_step",),
        "SupervisedMethod.train_step (Labeled-Only Lower Bound)",
    ),
}


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


def _find_ast_node(tree: ast.AST, qual_name: str) -> Optional[ast.AST]:
    """Locate a module-level or nested ClassDef/FunctionDef node by dotted name."""
    cur_body = getattr(tree, "body", [])
    node: Optional[ast.AST] = None
    for part in qual_name.split("."):
        node = next(
            (item for item in cur_body if getattr(item, "name", None) == part),
            None,
        )
        if node is None:
            return None
        cur_body = getattr(node, "body", [])
    return node


def extract_symbol_snippet(
    rel_path: str,
    symbols: Sequence[str],
    title: str = "",
) -> Dict[str, Any]:
    """Resolve line ranges dynamically via Python AST for named classes/methods/functions."""
    abs_path = REPO_ROOT / rel_path
    if not abs_path.is_file():
        return extract_code_snippet(rel_path, 1, 1, title=title)

    source = abs_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(abs_path))
    spans: List[Tuple[int, int]] = []
    for sym in symbols:
        node = _find_ast_node(tree, sym)
        if node is None or not hasattr(node, "lineno"):
            continue
        decorators = getattr(node, "decorator_list", []) or []
        start = min([node.lineno] + [d.lineno for d in decorators])
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        spans.append((start, end))

    if not spans:
        return {
            "path": rel_path,
            "abs_path": str(abs_path),
            "start_line": 1,
            "end_line": 1,
            "title": title or f"{rel_path}::{', '.join(symbols)}",
            "code": f"# Symbols not found in {rel_path}: {', '.join(symbols)}",
        }

    spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    lines = source.splitlines()
    blocks: List[str] = []
    for block_idx, (start, end) in enumerate(merged):
        if block_idx > 0:
            blocks.append("   ... | # ...")
        s_idx = max(0, start - 1)
        e_idx = min(len(lines), end)
        blocks.extend(f"{i + 1:4d} | {lines[i]}" for i in range(s_idx, e_idx))

    return {
        "path": rel_path,
        "abs_path": str(abs_path),
        "start_line": merged[0][0],
        "end_line": merged[-1][1],
        "title": title or f"{rel_path}::{', '.join(symbols)}",
        "code": "\n".join(blocks),
    }


def load_raw_yaml(rel_path: str) -> Dict[str, Any]:
    abs_path = REPO_ROOT / rel_path
    if not abs_path.is_file():
        return {}
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config_presets() -> Dict[str, Any]:
    """Resolve actual repository configurations using core.config_parser.load_config."""
    resolved = {
        key: load_config(config_path=str(REPO_ROOT / rel_path)).to_dict()
        for key, rel_path in RESOLVED_PRESET_PATHS.items()
    }
    raw_tiers = {
        tier_key: {"path": rel_path, "data": load_raw_yaml(rel_path)}
        for tier_key, rel_path in RAW_TIER_PATHS.items()
    }
    return {
        "resolved": resolved,
        "raw_tiers": raw_tiers,
    }


def _parse_validation_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Parse validation metrics CSV without hardcoding specific class column names."""
    if not csv_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for col, raw_val in row.items():
                if raw_val is None or raw_val == "":
                    continue
                if col == "iteration":
                    parsed[col] = int(float(raw_val))
                else:
                    parsed[col] = round(float(raw_val), 5)
            rows.append(parsed)
    return rows


def load_empirical_runs(catalog_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load empirical run metadata, summaries, and trajectories from docs/analysis/."""
    analysis_dir = REPO_ROOT / "docs" / "analysis"
    meta_path = Path(catalog_path) if catalog_path is not None else EMPIRICAL_CATALOG_PATH
    if not meta_path.is_file():
        return {}

    catalog = json.loads(meta_path.read_text(encoding="utf-8"))
    entries = catalog.get("runs", catalog) if isinstance(catalog, dict) else catalog

    runs: Dict[str, Any] = {}
    for entry in entries:
        run_id = entry["id"]
        summary_name = entry.get("summary_file", f"run_{run_id}_summary.json")
        csv_name = entry.get("validation_csv", f"run_{run_id}_validation.csv")
        sum_path = analysis_dir / summary_name
        csv_path = analysis_dir / csv_name

        summary = json.loads(sum_path.read_text(encoding="utf-8")) if sum_path.is_file() else {}
        val_rows = _parse_validation_csv(csv_path)

        run_payload: Dict[str, Any] = {
            "id": run_id,
            "label": entry.get("label", run_id),
            "description": entry.get("description", ""),
            "summary": summary,
            "validation_trajectory": val_rows,
        }
        if "class_breakdown_final" in entry:
            run_payload["class_breakdown_final"] = entry["class_breakdown_final"]
        runs[run_id] = run_payload

    return runs


def collect_code_snippets() -> Dict[str, Dict[str, Any]]:
    """Extract key source code snippets across the repository via AST symbol resolution."""
    return {
        key: extract_symbol_snippet(rel_path, symbols, title=title)
        for key, (rel_path, symbols, title) in CODE_SNIPPET_SPECS.items()
    }
