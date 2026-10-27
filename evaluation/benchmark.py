"""
Benchmark Evaluation Module for 5-Fold Cross Validation.
Aggregates Out-of-Fold (OOF) predictions, computes per-class metrics with Mean +/- Std,
and formats comparative benchmark reports. Compatible with multiple datasets and architectures.
"""

from typing import Dict, List, Optional, Tuple
import os
import json
import logging
import numpy as np
import torch
from tqdm import tqdm

from core.registry import DATASETS
from evaluation.metrics import (
    CLASS_NAMES_MULTICLASS,
    CLASS_NAMES_BINARY,
    cal_multiclass_dice,
    cal_binary_dice,
    format_dice_summary,
    case_segmentation_stats, aggregate_segmentation_stats, METRIC_POLICY,
)
from evaluation.sliding_window import sliding_window_inference_3d
from utils.checkpoint_resolver import resolve_checkpoint
from utils.evaluation_checkpoint import build_evaluation_model


def benchmark_5folds(
    model_builder,
    exp_dir: str,
    root_path: str,
    task: str = "multiclass",
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    stride_z: int = 8,
    stride_xy: int = 32,
    use_amp: bool = True,
    patch_size: Tuple[int, int, int] = (32, 160, 160),
    dataset_cls=None,
    dataset_name: str = "bhsd",
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    use_gaussian: bool = True,
    use_tta: bool = False,
    checkpoint_preference="best",
    allow_latest=False,
    include_pretrain=False,
    on_missing="error",
    compute_surface=False,
) -> Dict:
    """
    Run Out-of-Fold (OOF) 5-fold benchmark evaluation.
    
    Args:
        model_builder: Callable returning a torch.nn.Module given model_name and args.
        exp_dir: Base directory of the experiment containing fold_0, fold_1, etc.
        root_path: Path to dataset root directory.
        task: 'multiclass' or 'binary'.
        model_name: Model architecture name.
        device: Torch device.
        stride_z, stride_xy: Sliding window inference strides.
        use_amp: Whether to use FP16 AMP.
        patch_size: 3D patch dimensions.
        dataset_cls: Optional Dataset class.
        dataset_name: Registered dataset name.
        class_names: Optional custom class names list.
        num_classes: Optional number of classes.
        use_gaussian: Whether to apply Gaussian importance weighting mask during sliding window.
        use_tta: Whether to apply Test-Time Augmentation (TTA) flips across spatial axes.
        checkpoint_preference: Require best (default) or latest.
        allow_latest: Explicitly permit latest when best is missing in the selected phase.
        include_pretrain: Permit pretraining only if no later phase has checkpoints.
        on_missing: Raise on missing folds, or skip with an incomplete summary.
        
    Returns:
        Dict of results containing fold-wise metrics, overall OOF mean, std, and per-class breakdown.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_binary = (task.lower() == "binary")
    if num_classes is None:
        num_classes = 2 if is_binary else 6

    if class_names is None:
        class_names = CLASS_NAMES_BINARY if is_binary else CLASS_NAMES_MULTICLASS

    fg_class_names = class_names[1:] if len(class_names) > 1 else class_names

    fold_results = {}
    all_fold_dices = []
    per_class_all_folds = []
    oof_stats = []

    print("==========================================================================")
    print(f"=== BENCHMARK 5-FOLD CROSS VALIDATION: {os.path.basename(exp_dir)} ===")
    print(f"=== Dataset: {dataset_name.upper()} | Task: {task.upper()} ({num_classes} classes) | Model: {model_name} ===")
    print("==========================================================================")

    if on_missing not in ('error', 'skip'):
        raise ValueError('on_missing must be error or skip')
    resolved = {}
    skipped = {}
    # Preflight all folds before expensive inference.
    for fold in range(5):
        try:
            resolved[fold] = resolve_checkpoint(
                exp_dir, fold, model_name, checkpoint_preference,
                include_pretrain, allow_latest)
        except FileNotFoundError as exc:
            if on_missing == 'error':
                raise
            skipped[str(fold)] = str(exc)
            logging.warning('%s', exc)

    datasets = {}
    seen_cases = set()
    for fold in resolved:
        if dataset_cls is not None:
            val_dataset = dataset_cls(
                base_dir=root_path,
                split='val',
                binary=is_binary,
                patch_size=patch_size,
                fold=fold
            )
        else:
            d_name = dataset_name
            val_dataset = DATASETS.build(
                d_name,
                base_dir=root_path,
                split='val',
                binary=is_binary,
                patch_size=patch_size,
                fold=fold,
                num_classes=num_classes,
                class_names=class_names
            )

        cases = set(getattr(val_dataset, 'image_list', []))
        if seen_cases & cases:
            raise ValueError(f'OOF validation cases repeat across folds: {sorted(seen_cases & cases)[:5]}')
        seen_cases.update(cases)
        datasets[fold] = val_dataset

    for fold, (ckpt_path, kind) in resolved.items():
        print(f"[Fold {fold}] checkpoint={ckpt_path} kind={kind}")
        model, metadata, actual_model = build_evaluation_model(
            ckpt_path, model_builder, model_name, num_classes,
            fold=fold, task=task, dataset_name=dataset_name)
        if kind == 'best' and metadata.get('checkpoint_kind') not in (None, 'best', 'unverified'):
            raise ValueError(f'{ckpt_path}: file named best has provenance {metadata["checkpoint_kind"]}')
        if kind == 'best' and metadata.get('checkpoint_kind', 'unverified') == 'unverified':
            logging.warning('%s: legacy best; validation provenance unverified', ckpt_path)
        model.to(device)
        model.eval()

        val_dataset = datasets[fold]

        if len(val_dataset) == 0:
            raise ValueError(f"Fold {fold}: empty validation dataset")
        case_dices = []
        fold_stats = []
        for idx in tqdm(range(len(val_dataset)), desc=f"Fold {fold} Eval"):
            sample_item = val_dataset[idx]
            sample = sample_item[0] if isinstance(sample_item, tuple) else sample_item
            image, label = sample['image'], sample['label']

            pred = sliding_window_inference_3d(
                model=model,
                image=image,
                num_classes=num_classes,
                patch_size=patch_size,
                stride_z=stride_z,
                stride_xy=stride_xy,
                use_amp=use_amp,
                device=device,
                use_gaussian=use_gaussian,
                use_tta=use_tta
            )

            fold_stats.append(case_segmentation_stats(pred, label, num_classes,
                              spacing=sample.get('metadata', {}).get('spacing'), compute_surface=compute_surface))
            case_dices.append([row['dice'] for row in fold_stats[-1]['classes']])

        fold_mean_per_class = np.mean(np.array(case_dices), axis=0).tolist()
        fold_mdice = float(np.mean(fold_mean_per_class))

        oof_stats.extend(fold_stats)
        fold_results[f"fold_{fold}"] = {
            "detailed_metrics": aggregate_segmentation_stats(fold_stats),
            "mDice": fold_mdice,
            "per_class": fold_mean_per_class,
            "num_cases": len(val_dataset),
            "checkpoint": ckpt_path,
            "checkpoint_kind": kind,
            "checkpoint_provenance": metadata.get("checkpoint_kind", "unverified"),
            "weights_source": "ema_net" if "ema_net" in metadata else "net",
            "model": actual_model
        }
        all_fold_dices.append(fold_mdice)
        per_class_all_folds.append(fold_mean_per_class)

        summary_str = format_dice_summary(fold_mean_per_class, fg_class_names)
        print(f"[Fold {fold} Done] mDice: {fold_mdice:.4f} | {summary_str}")

    if not all_fold_dices:
        print("[ERROR] Không tìm thấy kết quả của fold nào!")
        raise RuntimeError("No folds evaluated; no benchmark result produced")

    overall_mean = float(np.mean(all_fold_dices))
    overall_std = float(np.std(all_fold_dices))
    per_class_matrix = np.array(per_class_all_folds)
    per_class_means = np.mean(per_class_matrix, axis=0).tolist()
    per_class_stds = np.std(per_class_matrix, axis=0).tolist()

    benchmark_summary = {
        "dataset": dataset_name,
        "task": task,
        "model": model_name,
        "fold_mean_dice": overall_mean,
        "fold_std_dice": overall_std,
        "aggregation": "unweighted_mean_over_folds",
        "complete": len(fold_results) == 5,
        "evaluated_folds": list(resolved),
        "skipped_folds": skipped,
        "checkpoint_preference": checkpoint_preference,
        "allow_latest": allow_latest,
        "metric_policy": METRIC_POLICY,
        "oof_case_metrics": aggregate_segmentation_stats(oof_stats),
        "inference": {"patch_size": list(patch_size), "stride_z": stride_z,
                      "stride_xy": stride_xy, "gaussian": use_gaussian, "tta": use_tta},
        "per_class_means": per_class_means,
        "per_class_stds": per_class_stds,
        "fold_results": fold_results
    }

    # Print Report Table
    print("\n" + "=" * 70)
    print("=== TỔNG HỢP BENCHMARK 5-FOLD CROSS VALIDATION (OOF) ===")
    print("=" * 70)
    print(f"Dataset: {dataset_name.upper()} | Model: {model_name} | Nhiệm vụ: {task.upper()}")
    print(f"Mean fold Dice: {overall_mean:.4f} ± {overall_std:.4f}")
    print("-" * 70)
    for c_idx, name in enumerate(fg_class_names):
        if c_idx < len(per_class_means):
            c_m = per_class_means[c_idx]
            c_s = per_class_stds[c_idx]
            print(f"  - {name:<22}: {c_m:.4f} ± {c_s:.4f}")
    print("=" * 70)

    # Save summary json
    summary_file = os.path.join(exp_dir, "benchmark_summary.json")
    with open(summary_file + '.tmp', 'w') as f:
        json.dump(benchmark_summary, f, indent=2)
    os.replace(summary_file + '.tmp', summary_file)
    print(f"-> Đã lưu chi tiết benchmark tại: {summary_file}")

    return benchmark_summary
