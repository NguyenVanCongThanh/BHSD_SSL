#!/usr/bin/env python3
"""
Single Unified Evaluation & Benchmark Entrypoint for MedSSL-BHSD Framework.
Supports single checkpoint evaluation, multi-model profiling, and 5-fold cross-validation benchmarking
across multiple medical datasets.

Usage Examples:
    # 1. Benchmark full 5-fold cross-validation experiment:
    python evaluate.py --exp_dir /home/u001015/experiments/BHSD_multiclass_PICK_5folds --dataset bhsd_multiclass

    # 2. Benchmark binary task:
    python evaluate.py --exp_dir /home/u001015/experiments/BHSD_binary_PICK_5folds --dataset bhsd_binary

    # 3. Evaluate single checkpoint:
    python evaluate.py --checkpoint /path/to/model.pth --dataset bhsd_multiclass --fold 0

    # 4. Profile computational complexity:
    python evaluate.py --model unet3d --profile
"""

import os
import sys
import argparse
import logging
import torch

from core.registry import MODELS, DATASETS
from core.config_parser import load_config
from evaluation.benchmark import benchmark_5folds
from evaluation.sliding_window import evaluate_dataset_sliding_window
from evaluation.metrics import format_dice_summary, CLASS_NAMES_MULTICLASS, CLASS_NAMES_BINARY
from evaluation.complexity import profile_model_complexity, format_complexity_summary
import data    # Register datasets
import models  # Register models
from utils.logger import get_logger


def model_builder_helper(name: str, in_channels: int = 1, num_classes: int = 6):
    actual_name = name
    if name == 'pick':
        actual_name = 'pick_vnet'
    return MODELS.build(actual_name, in_channels=in_channels, num_classes=num_classes, has_dropout=False)


def main():
    get_logger()

    parser = argparse.ArgumentParser(description="MedSSL-BHSD Unified Benchmark & Evaluation")
    parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory containing fold_0..fold_4')
    parser.add_argument('--checkpoint', type=str, default=None, help='Direct checkpoint path to evaluate single model')
    parser.add_argument('--dataset', type=str, default='bhsd_multiclass', help='Dataset name or config (e.g. bhsd_multiclass, bhsd_binary, acdc)')
    parser.add_argument('--root_path', type=str, default=None, help='Override path to H5 dataset directory')
    parser.add_argument('--task', type=str, default=None, choices=['multiclass', 'binary'], help='Task type override')
    parser.add_argument('--model', type=str, default='vnet', help='Model architecture name')
    parser.add_argument('--fold', type=int, default=0, help='Fold to evaluate (used if evaluating single checkpoint)')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU ID')
    parser.add_argument('--stride_xy', type=int, default=32, help='XY stride for sliding window')
    parser.add_argument('--stride_z', type=int, default=8, help='Z stride for sliding window')
    parser.add_argument('--profile', action='store_true', help='Measure computational complexity (FLOPs, params, latency, VRAM)')
    parser.add_argument('--patch_size', nargs=3, type=int, default=None, help='Input patch dimensions (Z Y X)')
    args = parser.parse_args()

    if args.gpu not in ('auto', 'all', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load dataset config
    cfg = load_config(dataset_config=args.dataset)
    dataset_name = getattr(cfg.data, 'name', 'bhsd')
    task = args.task or getattr(cfg.data, 'task', getattr(cfg.experiment, 'task', 'multiclass'))
    is_binary = (task == 'binary')
    num_classes = 2 if is_binary else int(getattr(cfg.data, 'num_classes', getattr(cfg.experiment, 'num_classes', 6)))
    root_path = args.root_path or getattr(cfg.data, 'root_path', '/home/u001015/notebook/preprocessed_bhsd_pick/')
    patch_size = tuple(args.patch_size) if args.patch_size else tuple(getattr(cfg.data, 'patch_size', [32, 160, 160]))
    class_names = getattr(cfg.data, 'class_names', None)
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if is_binary else CLASS_NAMES_MULTICLASS

    if args.exp_dir:
        # 5-fold benchmark aggregation
        benchmark_5folds(
            model_builder=model_builder_helper,
            exp_dir=args.exp_dir,
            root_path=root_path,
            task=task,
            model_name=args.model,
            device=device,
            stride_z=args.stride_z,
            stride_xy=args.stride_xy,
            patch_size=patch_size,
            dataset_name=dataset_name,
            class_names=class_names,
            num_classes=num_classes
        )
    elif args.checkpoint:
        # Single checkpoint evaluation
        logging.info(f"Evaluating single checkpoint: {args.checkpoint} on fold {args.fold}...")
        model = model_builder_helper(args.model, in_channels=1, num_classes=num_classes)
        from utils.checkpoint import load_checkpoint
        load_checkpoint(args.checkpoint, model=model, device=device, prefer_ema=True)
        model.to(device)
        model.eval()

        d_name = dataset_name if DATASETS.contains(dataset_name) else "bhsd"
        val_dataset = DATASETS.build(
            d_name,
            base_dir=root_path,
            split='val',
            binary=is_binary,
            patch_size=patch_size,
            fold=args.fold,
            num_classes=num_classes,
            class_names=class_names
        )

        mdice, per_class = evaluate_dataset_sliding_window(
            model=model,
            val_dataset=val_dataset,
            num_classes=num_classes,
            is_binary=is_binary,
            stride_z=args.stride_z,
            stride_xy=args.stride_xy,
            device=device
        )

        fg_names = class_names[1:] if len(class_names) > 1 and not is_binary else class_names
        summary_str = format_dice_summary(per_class, fg_names)
        print("\n" + "=" * 65)
        print(f"Single Checkpoint Evaluation Result:")
        print(f"Dataset:    {dataset_name.upper()} | Model: {args.model} | Fold: {args.fold}")
        print(f"Mean Dice:  {mdice:.4f}")
        print(f"Per-Class:  {summary_str}")
        print("=" * 65)

        if args.profile:
            input_shape = (1, 1, patch_size[0], patch_size[1], patch_size[2])
            metrics = profile_model_complexity(model, input_size=input_shape, device=device)
            print("\n" + format_complexity_summary(metrics))
    elif args.profile:
        # Direct profile mode without checkpoint
        model = model_builder_helper(args.model, in_channels=1, num_classes=num_classes)
        input_shape = (1, 1, patch_size[0], patch_size[1], patch_size[2])
        metrics = profile_model_complexity(model, input_size=input_shape, device=device)
        print("\n" + format_complexity_summary(metrics))
    else:
        print("[ERROR] Please provide either --exp_dir (for 5-fold benchmark) or --checkpoint (for single evaluation).")
        parser.print_help()


if __name__ == "__main__":
    main()
