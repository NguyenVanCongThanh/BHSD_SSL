"""
Benchmark Evaluation Module for 5-Fold Cross Validation on BHSD.
Aggregates Out-of-Fold (OOF) predictions, computes per-class metrics with Mean +/- Std,
and formats comparative benchmark reports.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from evaluation.metrics import (
    CLASS_NAMES_MULTICLASS,
    CLASS_NAMES_BINARY,
    cal_multiclass_dice,
    cal_binary_dice,
    format_dice_summary,
)
from evaluation.sliding_window import sliding_window_inference_3d


def benchmark_5folds(
    model_builder,
    exp_dir: str,
    root_path: str,
    task: str = "multiclass",
    model_name: str = "vnet",
    device: Optional[torch.device] = None,
    stride_z: int = 8,
    stride_xy: int = 32,
    use_amp: bool = True,
    patch_size: Tuple[int, int, int] = (32, 160, 160),
    dataset_cls=None
) -> Dict:
    """
    Run Out-of-Fold (OOF) 5-fold benchmark evaluation.
    
    Args:
        model_builder: Callable or registry function returning a torch.nn.Module given model_name and args.
        exp_dir: Base directory of the experiment containing fold_0, fold_1, etc.
        root_path: Path to BHSD dataset root directory.
        task: 'multiclass' or 'binary'.
        model_name: Model architecture name.
        device: Torch device.
        stride_z, stride_xy: Sliding window inference strides.
        use_amp: Whether to use FP16 AMP.
        patch_size: 3D patch dimensions.
        dataset_cls: BHSDDataset class.
        
    Returns:
        Dict of results containing fold-wise metrics, overall OOF mean, std, and per-class breakdown.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_binary = (task.lower() == "binary")
    num_classes = 2 if is_binary else 6
    class_names = CLASS_NAMES_BINARY if is_binary else CLASS_NAMES_MULTICLASS

    if dataset_cls is None:
        from data.dataset_bhsd import BHSDDataset
        dataset_cls = BHSDDataset

    fold_results = {}
    all_fold_dices = []
    per_class_all_folds = []

    print("==========================================================================")
    print(f"=== BENCHMARK 5-FOLD CROSS VALIDATION: {os.path.basename(exp_dir)} ===")
    print(f"=== Task: {task.upper()} ({num_classes} classes) | Model: {model_name} ===")
    print("==========================================================================")

    for fold in range(5):
        # Locate checkpoint
        candidate_paths = [
            os.path.join(exp_dir, f"fold_{fold}", "self_train", f"{model_name}_best_model.pth"),
            os.path.join(exp_dir, f"fold_{fold}", f"{model_name}_best_model.pth"),
            os.path.join(exp_dir, f"fold_{fold}", "self_train", "checkpoint_latest.pth"),
            os.path.join(exp_dir, f"fold_{fold}", "checkpoint_latest.pth"),
            os.path.join(exp_dir, f"fold{fold}", "self_train", f"{model_name}_best_model.pth"),
        ]

        ckpt_path = None
        for cand in candidate_paths:
            if os.path.exists(cand):
                ckpt_path = cand
                break

        if ckpt_path is None:
            logging.warning(f"[Fold {fold}] Checkpoint not found in candidates, skipping fold {fold}!")
            continue

        print(f"\n--- [Fold {fold}/4] Đang đánh giá với weights: {ckpt_path} ---")

        # Build model
        model = model_builder(model_name, in_channels=1, num_classes=num_classes)
        state = torch.load(str(ckpt_path), map_location=device)
        # Prioritize teacher weights (ema_net) if evaluating Mean Teacher checkpoints
        net_dict = state.get('ema_net', None)
        if net_dict is None:
            net_dict = state.get('net', state)
        clean_dict = {k[7:] if k.startswith('module.') else k: v for k, v in net_dict.items()}
        model.load_state_dict(clean_dict, strict=False)
        model.to(device)
        model.eval()

        val_dataset = dataset_cls(
            base_dir=root_path,
            split='val',
            binary=is_binary,
            patch_size=patch_size,
            fold=fold
        )

        case_dices = []
        for idx in tqdm(range(len(val_dataset)), desc=f"Fold {fold} Eval"):
            sample, _ = val_dataset[idx]
            image, label = sample['image'], sample['label']

            pred = sliding_window_inference_3d(
                model=model,
                image=image,
                num_classes=num_classes,
                patch_size=patch_size,
                stride_z=stride_z,
                stride_xy=stride_xy,
                use_amp=use_amp,
                device=device
            )

            if is_binary:
                d = cal_binary_dice(pred, label)
                case_dices.append([d])
            else:
                cd = cal_multiclass_dice(pred, label, num_classes=num_classes)
                case_dices.append(cd)

        fold_mean_per_class = np.mean(np.array(case_dices), axis=0).tolist()
        fold_mdice = float(np.mean(fold_mean_per_class))

        fold_results[f"fold_{fold}"] = {
            "mDice": fold_mdice,
            "per_class": fold_mean_per_class,
            "num_cases": len(val_dataset),
            "checkpoint": ckpt_path
        }
        all_fold_dices.append(fold_mdice)
        per_class_all_folds.append(fold_mean_per_class)

        summary_str = format_dice_summary(fold_mean_per_class, class_names[1:])
        print(f"[Fold {fold} Done] mDice: {fold_mdice:.4f} | {summary_str}")

    if not all_fold_dices:
        print("[ERROR] Không tìm thấy kết quả của fold nào!")
        return {}

    overall_mean = float(np.mean(all_fold_dices))
    overall_std = float(np.std(all_fold_dices))
    per_class_matrix = np.array(per_class_all_folds)
    per_class_means = np.mean(per_class_matrix, axis=0).tolist()
    per_class_stds = np.std(per_class_matrix, axis=0).tolist()

    benchmark_summary = {
        "task": task,
        "model": model_name,
        "overall_oof_mean_dice": overall_mean,
        "overall_oof_std_dice": overall_std,
        "per_class_means": per_class_means,
        "per_class_stds": per_class_stds,
        "fold_results": fold_results
    }

    # Print Report Table
    print("\n" + "=" * 70)
    print("=== TỔNG HỢP BENCHMARK 5-FOLD CROSS VALIDATION (OOF) ===")
    print("=" * 70)
    print(f"Phương pháp / Model: {model_name} | Nhiệm vụ: {task.upper()}")
    print(f"Mean OOF Dice: {overall_mean:.4f} ± {overall_std:.4f}")
    print("-" * 70)
    for c_idx, name in enumerate(class_names[1:]):
        c_m = per_class_means[c_idx]
        c_s = per_class_stds[c_idx]
        print(f"  - {name:<22}: {c_m:.4f} ± {c_s:.4f}")
    print("=" * 70)

    # Save summary json
    summary_file = os.path.join(exp_dir, "benchmark_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"-> Đã lưu chi tiết benchmark tại: {summary_file}")

    return benchmark_summary
