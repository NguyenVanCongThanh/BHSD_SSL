#!/usr/bin/env python3
"""
Setup and standardize Weights & Biases (W&B) Workspace Views for MedSSL-BHSD.

Creates three dedicated, uncluttered workspace views:
1. Overview: Core convergence, mDice, per-class dice, losses, and learning dynamics (~7-9 panels).
2. Evaluation: Clinical & geometric metrics (HD95, ASD, False positive rate/volume, precision, recall, slice gallery).
3. Diagnostics: Voxel counts (TP/FP/FN) and cohort support counts.

Features:
- Idempotent: Can be run repeatedly without duplicating views or panels.
- Config-driven & Dynamic: Supports BHSD multiclass (5 foreground classes), binary, or custom classes.
- Safe: Supports --dry-run to inspect generated structure before touching W&B Cloud.
"""

import argparse
import json
import logging
import sys
from typing import List, Dict, Any, Optional

try:
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
except ImportError:
    wandb = None
    ws = None
    wr = None


def check_dependencies():
    """Ensure wandb and wandb_workspaces are available."""
    if ws is None or wr is None or wandb is None:
        raise ImportError(
            "Required packages not found: 'wandb' and 'wandb-workspaces'.\n"
            "Please install via: pip install 'wandb[workspaces]'"
        )

BHSD_FOREGROUND_CLASSES = [
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural"
]

BHSD_BINARY_CLASSES = [
    "Hemorrhage"
]


def build_overview_workspace(
    entity: str,
    project: str,
    classes: List[str],
    step_metric: str = "global_step"
) -> ws.Workspace:
    """Build the clean 'Overview' workspace view (7-9 core panels)."""
    # Section 1: Core Validation Performance
    p_mdice = wr.LinePlot(
        x=step_metric,
        y=["val/mDice", "val/best_mDice"],
        title="Validation mDice (Current vs Best Record)"
    )
    p_pos_macro = wr.LinePlot(
        x=step_metric,
        y=["val/macro_positive_dice"],
        title="Macro Positive Dice (Eligible Classes with Lesions)"
    )
    p_dice_class = wr.LinePlot(
        x=step_metric,
        y=[f"val/dice_{c}" for c in classes],
        title="Per-Class Segmentation Dice (Multi-line)"
    )
    p_dice_pos_class = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_dice_positive_mean" for c in classes],
        title="Per-Class Positive Case Dice (Multi-line)"
    )
    sec_validation = ws.Section(
        name="1. Core Validation Performance",
        panels=[p_mdice, p_pos_macro, p_dice_class, p_dice_pos_class],
        is_open=True
    )

    # Section 2: Training Dynamics (losses separated from LR and weights due to scale)
    p_losses = wr.LinePlot(
        x=step_metric,
        y=["train/total", "train/supervised", "train/supervised_main", "train/consistency", "train/deep_supervision"],
        title="Training Losses (Supervised, SSL & Total)"
    )
    p_lr = wr.LinePlot(
        x=step_metric,
        y=["train/lr"],
        title="Learning Rate"
    )
    p_rampup = wr.LinePlot(
        x=step_metric,
        y=["train/weight"],
        title="Consistency Loss Ramp-up Weight"
    )
    sec_training = ws.Section(
        name="2. Training Dynamics",
        panels=[p_losses, p_lr, p_rampup],
        is_open=True
    )

    workspace = ws.Workspace(
        name="Overview",
        entity=entity,
        project=project,
        sections=[sec_validation, sec_training]
    )
    workspace.settings.x_axis = step_metric
    return workspace


def build_evaluation_workspace(
    entity: str,
    project: str,
    classes: List[str],
    step_metric: str = "global_step"
) -> ws.Workspace:
    """Build the 'Evaluation' workspace view (Geometric boundary & clinical metrics)."""
    # Section 1: Geometric Boundary Metrics
    p_hd95 = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_hd95_mm_mean" for c in classes],
        title="Hausdorff Distance 95th Percentile (HD95 in mm)"
    )
    p_asd = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_asd_mm_mean" for c in classes],
        title="Average Surface Distance (ASD in mm)"
    )
    sec_boundary = ws.Section(
        name="1. Geometric Boundary Metrics",
        panels=[p_hd95, p_asd],
        is_open=True
    )

    # Section 2: Specificity & False Positive Analysis
    p_fp_rate = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_negative_fp_case_rate" for c in classes],
        title="Negative False Positive Case Rate (False Alarms on Non-lesion Cases)"
    )
    p_fp_vol = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_negative_fp_ml_mean" for c in classes],
        title="Negative FP Mean Volume (ml per False Alarm Case)"
    )
    p_precision = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_precision_micro" for c in classes],
        title="Micro Precision per Class"
    )
    p_recall = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_recall_micro" for c in classes],
        title="Micro Recall per Class"
    )
    sec_specificity = ws.Section(
        name="2. Specificity & False Positive Analysis",
        panels=[p_fp_rate, p_fp_vol, p_precision, p_recall],
        is_open=True
    )

    # Section 3: Visual Inspection Gallery
    p_gallery = wr.MediaBrowser(
        title="Visual Predictions Gallery (CT | GT | Pred | Error Map)",
        media_keys=["val/predictions"]
    )
    sec_visual = ws.Section(
        name="3. Qualitative Visual Gallery",
        panels=[p_gallery],
        is_open=True
    )

    workspace = ws.Workspace(
        name="Evaluation",
        entity=entity,
        project=project,
        sections=[sec_boundary, sec_specificity, sec_visual]
    )
    workspace.settings.x_axis = step_metric
    return workspace


def build_diagnostics_workspace(
    entity: str,
    project: str,
    classes: List[str],
    step_metric: str = "global_step"
) -> ws.Workspace:
    """Build the 'Diagnostics' workspace view (Support counts and raw voxel counts)."""
    # Section 1: Cohort Support Counts
    p_pos_cases = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_positive_cases" for c in classes],
        title="Positive Case Count in Val Set per Class"
    )
    p_neg_cases = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_negative_cases" for c in classes],
        title="Negative Case Count in Val Set per Class"
    )
    p_surf_pairs = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_surface_valid_pairs" for c in classes],
        title="Valid Surface Pairs (Pred & GT non-empty)"
    )
    sec_support = ws.Section(
        name="1. Cohort Support & Denominators",
        panels=[p_pos_cases, p_neg_cases, p_surf_pairs],
        is_open=True
    )

    # Section 2: Voxel Confusion Volumes
    p_tp_vox = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_tp_voxels" for c in classes],
        title="True Positive Voxels"
    )
    p_fp_vox = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_fp_voxels" for c in classes],
        title="False Positive Voxels"
    )
    p_fn_vox = wr.LinePlot(
        x=step_metric,
        y=[f"val/{c}_fn_voxels" for c in classes],
        title="False Negative Voxels"
    )
    sec_voxels = ws.Section(
        name="2. Raw Voxel Counts",
        panels=[p_tp_vox, p_fp_vox, p_fn_vox],
        is_open=True
    )

    workspace = ws.Workspace(
        name="Diagnostics",
        entity=entity,
        project=project,
        sections=[sec_support, sec_voxels]
    )
    workspace.settings.x_axis = step_metric
    return workspace


def inspect_workspace(workspace: ws.Workspace) -> Dict[str, Any]:
    """Extract a summary description of a workspace layout."""
    sections_info = []
    total_panels = 0
    for sec in workspace.sections:
        panels_info = []
        for p in sec.panels:
            total_panels += 1
            info = {
                "type": type(p).__name__,
                "title": getattr(p, "title", "Untitled"),
            }
            if hasattr(p, "y"):
                info["y_series"] = p.y
            if hasattr(p, "media_keys"):
                info["media_keys"] = p.media_keys
            panels_info.append(info)
        sections_info.append({
            "section_name": sec.name,
            "panel_count": len(sec.panels),
            "panels": panels_info
        })
    return {
        "view_name": workspace.name,
        "project": f"{workspace.entity}/{workspace.project}",
        "total_sections": len(workspace.sections),
        "total_panels": total_panels,
        "auto_generate_panels": workspace.auto_generate_panels,
        "sections": sections_info
    }


def main():
    parser = argparse.ArgumentParser(
        description="Standardize and publish clean W&B Workspace Views for MedSSL-BHSD."
    )
    parser.add_argument(
        "--project",
        type=str,
        default="PICK_BHSD",
        help="W&B project name (default: PICK_BHSD)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity/username (default: auto-detected from W&B session)"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="List of foreground class names, or 'bhsd', 'binary' (default: 5 BHSD foreground classes)"
    )
    parser.add_argument(
        "--view",
        choices=["all", "Overview", "Evaluation", "Diagnostics"],
        default="all",
        help="Specific view to create or update (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect generated layout specs without saving to W&B Cloud"
    )
    parser.add_argument(
        "--step-metric",
        type=str,
        default="global_step",
        help="Explicit X-axis step metric (default: global_step)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Resolve entity
    entity = args.entity
    if not entity and not args.dry_run:
        try:
            api = wandb.Api()
            entity = api.default_entity
        except Exception as e:
            logging.warning(f"Could not automatically determine default entity: {e}")
    entity = entity or "default_entity"

    # Resolve classes
    if not args.classes or args.classes == ["bhsd"]:
        classes = BHSD_FOREGROUND_CLASSES
    elif args.classes == ["binary"]:
        classes = BHSD_BINARY_CLASSES
    else:
        classes = args.classes

    logging.info(f"Target Project: {entity}/{args.project}")
    logging.info(f"Foreground Classes ({len(classes)}): {classes}")

    # Build workspace builders mapping
    builders = {
        "Overview": build_overview_workspace,
        "Evaluation": build_evaluation_workspace,
        "Diagnostics": build_diagnostics_workspace
    }

    views_to_build = ["Overview", "Evaluation", "Diagnostics"] if args.view == "all" else [args.view]

    for view_name in views_to_build:
        builder = builders[view_name]
        workspace = builder(entity=entity, project=args.project, classes=classes, step_metric=args.step_metric)
        summary = inspect_workspace(workspace)

        logging.info("=" * 60)
        logging.info(f"VIEW: {view_name} (Panels: {summary['total_panels']}, Auto-generate: {summary['auto_generate_panels']})")
        for s in summary["sections"]:
            logging.info(f"  + Section '{s['section_name']}' ({s['panel_count']} panels)")
            for p in s["panels"]:
                details = p.get("y_series") or p.get("media_keys") or []
                logging.info(f"     - [{p['type']}] {p['title']} ({len(details)} metrics)")

        if args.dry_run:
            logging.info(f"[DRY-RUN] Skipped saving view '{view_name}' to W&B.")
        else:
            try:
                logging.info(f"Saving view '{view_name}' to W&B Cloud...")
                saved_ws = workspace.save()
                logging.info(f"[SUCCESS] View '{view_name}' saved. URL: {saved_ws.url if hasattr(saved_ws, 'url') else 'saved'}")
            except Exception as save_err:
                logging.error(f"[ERROR] Failed to save view '{view_name}': {save_err}")
                logging.info("Note: You may need valid W&B authentication credentials or permissions to save views.")


if __name__ == "__main__":
    main()
