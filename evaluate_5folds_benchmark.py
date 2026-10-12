#!/usr/bin/env python3
"""
Evaluate and Benchmark 5-Fold Cross Validation for BHSD Semi-Supervised Segmentation.
Calculates Out-of-Fold (OOF) Mean Dice +/- Std across all 5 folds.
Outputs detailed class-wise metrics and comparative benchmark tables.
Now powered by unified evaluation and data modules.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data.dataset_bhsd import BHSDDataset
from networks.net_factory import net_factory
from evaluation.metrics import (
    CLASS_NAMES_MULTICLASS,
    CLASS_NAMES_BINARY,
    cal_multiclass_dice,
    cal_binary_dice,
    format_dice_summary,
)
from evaluation.sliding_window import sliding_window_inference_3d

CLASS_NAMES = CLASS_NAMES_MULTICLASS


def load_model_weights(model: nn.Module, pth_path: str) -> nn.Module:
    state = torch.load(str(pth_path), map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    net_dict = state.get('net', state)
    raw_net = model.module if isinstance(model, nn.DataParallel) else model
    clean_dict = {k[7:] if k.startswith('module.') else k: v for k, v in net_dict.items()}
    raw_net.load_state_dict(clean_dict, strict=False)
    return model


def benchmark_experiment(
    exp_dir: str,
    root_path: str,
    task: str = 'binary',
    model_name: str = 'VNet',
    gpu: str = '0'
):
    if gpu not in ['auto', 'all']:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    is_binary = (task.lower() == 'binary')
    num_classes = 2 if is_binary else 6
    patch_size = (32, 160, 160)
    class_names = CLASS_NAMES_BINARY if is_binary else CLASS_NAMES_MULTICLASS

    print("==========================================================================")
    print(f"=== BENCHMARK 5-FOLD CROSS VALIDATION: {os.path.basename(exp_dir)} ===")
    print(f"=== Task: {task.upper()} ({num_classes} classes) | Model: {model_name} ===")
    print("==========================================================================")

    fold_results = {}
    all_fold_dices = []
    per_class_all_folds = []

    for fold in range(5):
        fold_sub = os.path.join(exp_dir, f"fold_{fold}")
        
        candidates = [
            os.path.join(fold_sub, "self_train", f"{model_name}_best_model.pth"),
            os.path.join(fold_sub, f"{model_name}_best_model.pth"),
            os.path.join(fold_sub, "pre_train", f"{model_name}_best_model.pth"),
            os.path.join(fold_sub, "self_train", "checkpoint_latest.pth"),
        ]
        
        ckpt_path = None
        for cand in candidates:
            if os.path.exists(cand):
                ckpt_path = cand
                break

        if ckpt_path is None:
            print(f"[WARN] No checkpoint found for fold {fold} in {fold_sub}, skipping.")
            continue

        print(f"\n[FOLD {fold}] Loading checkpoint: {ckpt_path}")
        model = net_factory(net_type=model_name, in_chns=1, class_num=num_classes, mode="test")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        model = load_model_weights(model, ckpt_path)
        model.eval()

        val_dataset = BHSDDataset(
            base_dir=root_path,
            split='val',
            binary=is_binary,
            patch_size=patch_size,
            fold=fold
        )

        fold_case_dices = []
        fold_per_class_list = []

        with torch.no_grad():
            for i in tqdm(range(len(val_dataset)), desc=f"Evaluating Fold {fold}"):
                sample, _ = val_dataset[i]
                image, label = sample['image'], sample['label']

                pred = sliding_window_inference_3d(
                    model=model,
                    image=image,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_z=8,
                    stride_xy=32,
                    use_amp=True,
                    device=device
                )

                if is_binary:
                    d = cal_binary_dice(pred, label)
                    fold_case_dices.append(d)
                else:
                    c_dices = cal_multiclass_dice(pred, label, num_classes=num_classes)
                    fold_per_class_list.append(c_dices)
                    fold_case_dices.append(float(np.mean(c_dices)))

        fold_mean_dice = float(np.mean(fold_case_dices))
        fold_results[f"fold_{fold}"] = {
            "mean_dice": fold_mean_dice,
            "num_cases": len(fold_case_dices),
            "checkpoint": ckpt_path
        }
        all_fold_dices.append(fold_mean_dice)

        if not is_binary and fold_per_class_list:
            fold_cls_mean = np.mean(np.array(fold_per_class_list), axis=0).tolist()
            fold_results[f"fold_{fold}"]["per_class_dice"] = {
                class_names[c + 1]: fold_cls_mean[c] for c in range(len(fold_cls_mean))
            }
            per_class_all_folds.append(fold_cls_mean)

        print(f"--> [FOLD {fold} RESULT] Mean Dice: {fold_mean_dice:.4f}")

    if not all_fold_dices:
        print("[ERROR] No fold evaluations completed. Please check experiment directory.")
        return

    overall_mean = float(np.mean(all_fold_dices))
    overall_std = float(np.std(all_fold_dices))

    summary = {
        "experiment": os.path.basename(exp_dir),
        "task": task,
        "model": model_name,
        "completed_folds": len(all_fold_dices),
        "overall_mean_dice": overall_mean,
        "overall_std_dice": overall_std,
        "folds": fold_results
    }

    if not is_binary and per_class_all_folds:
        cls_5fold_mean = np.mean(np.array(per_class_all_folds), axis=0)
        cls_5fold_std = np.std(np.array(per_class_all_folds), axis=0)
        summary["subclass_benchmark"] = {
            class_names[c + 1]: {
                "mean_dice": float(cls_5fold_mean[c]),
                "std_dice": float(cls_5fold_std[c])
            }
            for c in range(len(cls_5fold_mean))
        }

    out_json = os.path.join(exp_dir, "benchmark_5folds_summary.json")
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 70)
    print("=== 5-FOLD CROSS VALIDATION BENCHMARK REPORT ===")
    print("=" * 70)
    print(f"Experiment: {os.path.basename(exp_dir)}")
    print(f"Task:       {task.upper()}")
    print("-" * 70)
    for f in range(5):
        key = f"fold_{f}"
        if key in fold_results:
            d = fold_results[key]['mean_dice']
            print(f"Fold {f}:  Dice = {d:.4f} ({fold_results[key]['num_cases']} cases)")
    print("-" * 70)
    print(f"FINAL 5-FOLD SCORE: {overall_mean:.4f} ± {overall_std:.4f}")
    print("=" * 70)

    if not is_binary and "subclass_benchmark" in summary:
        print("\n--- Subclass Breakdown (5-Fold Mean ± Std) ---")
        for cls_name, vals in summary["subclass_benchmark"].items():
            print(f"  * {cls_name:<20}: {vals['mean_dice']:.4f} ± {vals['std_dice']:.4f}")
        print("=" * 70)

    print(f"\n[INFO] Full benchmark report saved to: {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5-Fold Benchmark for PICK")
    parser.add_argument('--exp_dir', type=str, default='/home/u001015/experiments/BHSD_multiclass_PICK_5folds',
                        help='Path to experiment directory containing fold_0..fold_4')
    parser.add_argument('--root_path', type=str, default='/home/u001015/dataset/BHSD_h5/',
                        help='Path to BHSD H5 dataset directory')
    parser.add_argument('--task', type=str, default='multiclass', choices=['binary', 'multiclass'],
                        help='Task type: binary or multiclass')
    parser.add_argument('--model', type=str, default='VNet', help='Model name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    args = parser.parse_args()

    benchmark_experiment(
        exp_dir=args.exp_dir,
        root_path=args.root_path,
        task=args.task,
        model_name=args.model,
        gpu=args.gpu
    )


if __name__ == "__main__":
    main()
