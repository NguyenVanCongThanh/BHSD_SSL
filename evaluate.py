#!/usr/bin/env python3
"""
Single Unified Evaluation & Benchmark Entrypoint for MedSSL-BHSD Framework.
Usage Examples:
    # Benchmark full 5-fold cross-validation experiment:
    python evaluate.py --exp_dir /home/u001015/experiments/BHSD_multiclass_PICK_5folds --task multiclass

    # Benchmark binary task:
    python evaluate.py --exp_dir /home/u001015/experiments/BHSD_binary_PICK_5folds --task binary

    # Evaluate single checkpoint:
    python evaluate.py --checkpoint /path/to/model.pth --task multiclass --fold 0
"""

import os
import sys
import argparse
import logging
import torch

from core.registry import MODELS
from data.dataset_bhsd import BHSDDataset
from evaluation.benchmark import benchmark_5folds
from evaluation.sliding_window import evaluate_dataset_sliding_window
from evaluation.metrics import format_dice_summary, CLASS_NAMES_MULTICLASS, CLASS_NAMES_BINARY
import models


def model_builder_helper(name: str, in_channels: int = 1, num_classes: int = 6):
    return MODELS.build(name, in_channels=in_channels, num_classes=num_classes, has_dropout=False)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    parser = argparse.ArgumentParser(description="MedSSL-BHSD Unified Benchmark & Evaluation")
    parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory containing fold_0..fold_4')
    parser.add_argument('--checkpoint', type=str, default=None, help='Direct checkpoint path to evaluate single model')
    parser.add_argument('--root_path', type=str, default='/home/u001015/notebook/preprocessed_bhsd_pick/', help='Path to BHSD H5 dataset directory')
    parser.add_argument('--task', type=str, default='multiclass', choices=['multiclass', 'binary'], help='Task type')
    parser.add_argument('--model', type=str, default='vnet', help='Model architecture name')
    parser.add_argument('--fold', type=int, default=0, help='Fold to evaluate (used if evaluating single checkpoint)')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU ID')
    parser.add_argument('--stride_xy', type=int, default=32, help='XY stride for sliding window')
    parser.add_argument('--stride_z', type=int, default=8, help='Z stride for sliding window')
    args = parser.parse_args()

    if args.gpu not in ('auto', 'all', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.exp_dir:
        # 5-fold benchmark aggregation
        benchmark_5folds(
            model_builder=model_builder_helper,
            exp_dir=args.exp_dir,
            root_path=args.root_path,
            task=args.task,
            model_name=args.model,
            device=device,
            stride_z=args.stride_z,
            stride_xy=args.stride_xy,
            dataset_cls=BHSDDataset
        )
    elif args.checkpoint:
        # Single checkpoint evaluation
        is_binary = (args.task == 'binary')
        num_classes = 2 if is_binary else 6
        class_names = CLASS_NAMES_BINARY if is_binary else CLASS_NAMES_MULTICLASS

        logging.info(f"Evaluating single checkpoint: {args.checkpoint} on fold {args.fold}...")
        model = model_builder_helper(args.model, in_channels=1, num_classes=num_classes)
        state = torch.load(args.checkpoint, map_location=device)
        net_dict = state.get('net', state)
        clean_dict = {k[7:] if k.startswith('module.') else k: v for k, v in net_dict.items()}
        model.load_state_dict(clean_dict, strict=False)
        model.to(device)
        model.eval()

        val_dataset = BHSDDataset(
            base_dir=args.root_path,
            split='val',
            binary=is_binary,
            fold=args.fold
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

        summary_str = format_dice_summary(per_class, class_names[1:])
        print("\n" + "=" * 60)
        print(f"Single Checkpoint Evaluation Result:")
        print(f"Mean Dice: {mdice:.4f}")
        print(f"Per-Class: {summary_str}")
        print("=" * 60)
    else:
        print("[ERROR] Please provide either --exp_dir (for 5-fold benchmark) or --checkpoint (for single evaluation).")
        parser.print_help()


if __name__ == "__main__":
    main()
