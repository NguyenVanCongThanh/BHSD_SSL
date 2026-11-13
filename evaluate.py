#!/usr/bin/env python3
"""
Single Unified Evaluation & Benchmark Entrypoint for MedSSL-BHSD Framework.
Supports single checkpoint evaluation, multi-model profiling, and 5-fold cross-validation benchmarking
across multiple medical datasets.

Usage Examples:
    # 1. Benchmark full 5-fold cross-validation experiment:
    python evaluate.py --exp_dir ~/experiments/BHSD_multiclass_PICK_5folds --dataset bhsd_multiclass

    # 2. Benchmark binary task:
    python evaluate.py --exp_dir ~/experiments/BHSD_binary_PICK_5folds --dataset bhsd_binary

    # 3. Evaluate single checkpoint:
    python evaluate.py --checkpoint /path/to/model.pth --dataset bhsd_multiclass --fold 0

    # 4. Profile computational complexity:
    python evaluate.py --model unet3d --profile
"""

import os
import sys
import argparse
import logging
import json
import inspect
import torch

from core.registry import MODELS, DATASETS
from core.config_parser import load_config
from data.constants import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PATCH_SIZE,
    DEFAULT_VAL_STEP_SIZE,
    get_dataset_class_names,
    resolve_sliding_window_strides,
)
from evaluation.benchmark import benchmark_5folds
from evaluation.sliding_window import evaluate_dataset_sliding_window
from evaluation.metrics import format_dice_summary
from evaluation.complexity import profile_model_complexity, format_complexity_summary
import data    # Register datasets
import models  # Register models
from utils.logger import get_logger


def model_builder_helper(name: str, in_channels: int = 1, num_classes: int = 6):
    actual_name = name.lower()
    if actual_name in ('pick', 'pick_model'):
        actual_name = 'pick_vnet'
    elif actual_name == 'lefed':
        actual_name = 'lefed_net'
    model_cls = MODELS.get(actual_name)
    if model_cls is None:
        raise KeyError(f"Model '{actual_name}' is not registered in MODELS")
    sig = inspect.signature(model_cls.__init__)
    kwargs = {'in_channels': in_channels, 'num_classes': num_classes}
    if 'has_dropout' in sig.parameters:
        kwargs['has_dropout'] = False
    if 'img_shape' in sig.parameters:
        kwargs.setdefault('img_shape', (16, 64, 64))
    return MODELS.build(actual_name, **kwargs)


def main():
    get_logger()

    parser = argparse.ArgumentParser(description="MedSSL-BHSD Unified Benchmark & Evaluation")
    parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory containing fold_0..fold_4')
    parser.add_argument('--checkpoint', type=str, default=None, help='Direct checkpoint path to evaluate single model')
    parser.add_argument('--dataset', type=str, default='bhsd_multiclass', help='Dataset name or config (e.g. bhsd_multiclass, bhsd_binary)')
    parser.add_argument('--root_path', type=str, default=None, help='Override path to H5 dataset directory')
    parser.add_argument('--task', type=str, default=None, choices=['multiclass', 'binary'], help='Task type override')
    parser.add_argument('--model', type=str, default=None, help='Model architecture (required for legacy checkpoints without config)')
    parser.add_argument('--fold', type=int, default=0, help='Fold to evaluate (used if evaluating single checkpoint)')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU ID')
    parser.add_argument('--stride_xy', type=int, default=None, help='XY stride for sliding window (default: derived from patch_size and --step_size)')
    parser.add_argument('--stride_z', type=int, default=None, help='Z stride for sliding window (default: derived from patch_size and --step_size)')
    parser.add_argument('--step_size', type=float, default=DEFAULT_VAL_STEP_SIZE, help='Relative step size (1 - overlap) when explicit strides are omitted (default: 0.5 = 50%% overlap)')
    parser.add_argument('--no_gaussian', action='store_true', help='Disable Gaussian weighting in sliding window (default: Gaussian enabled)')
    parser.add_argument('--use_tta', action='store_true', help='Enable 3D Test-Time Augmentation (flips across spatial axes, nnUNet standard)')
    parser.add_argument('--profile', action='store_true', help='Measure computational complexity (FLOPs, params, latency, VRAM)')
    parser.add_argument('--patch_size', nargs=3, type=int, default=None, help='Input patch dimensions (Z Y X)')
    parser.add_argument('--checkpoint_preference', choices=['best', 'latest'], default='best')
    parser.add_argument('--allow_latest', action='store_true', help='Allow best to fall back to latest')
    parser.add_argument('--include_pretrain', action='store_true')
    parser.add_argument('--on_missing', choices=['error', 'skip'], default='error')
    parser.add_argument('--metrics_output', help='Optional JSON path for single-checkpoint metrics and inference policy')
    parser.add_argument('--no_amp', action='store_true', help='Disable evaluation autocast')
    parser.add_argument('--surface_metrics', action='store_true', help='Compute physical HD95/ASD; spacing metadata required')
    parser.add_argument('--eval_space', type=str, choices=['preprocessed', 'native'], default='preprocessed',
                        help='Evaluation space: preprocessed (fast, on .h5 grid) or native (restored to original patient NIfTI space, nnUNet standard)')
    parser.add_argument('--raw_gt_dir', type=str, default=None,
                        help='Optional path to raw NIfTI ground truth directory (e.g. Dataset001_BHSD/labelsTr) for exact native space validation')
    args = parser.parse_args()

    if args.gpu not in ('auto', 'all', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Single checkpoint evaluation: inspect checkpoint metadata first if available
    saved_config = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            ckpt_peek = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            saved_config = ckpt_peek.get('config')
        except Exception:
            saved_config = None

    # Load dataset config
    cfg = load_config(dataset_config=args.dataset,
                      cli_args=['--task', args.task] if args.task else None)
    dataset_name = getattr(cfg.data, 'name', 'bhsd')
    task = args.task or getattr(cfg.data, 'task', getattr(cfg.experiment, 'task', 'multiclass'))
    is_binary = (task == 'binary')
    num_classes = 2 if is_binary else int(getattr(cfg.data, 'num_classes', getattr(cfg.experiment, 'num_classes', 6)))
    raw_root_path = (
        args.root_path
        or getattr(cfg.data, 'root_path', None)
        or os.environ.get('MEDSSL_DATA_ROOT')
        or DEFAULT_DATA_ROOT
    )
    root_path = os.path.expanduser(os.path.expandvars(str(raw_root_path)))

    # Auto-resolve patch_size: CLI explicit > Checkpoint saved config > YAML cfg > fallback
    if args.patch_size:
        patch_size = tuple(args.patch_size)
    elif saved_config and isinstance(saved_config.get('data'), dict) and saved_config['data'].get('patch_size'):
        patch_size = tuple(saved_config['data']['patch_size'])
    else:
        patch_size = tuple(getattr(cfg.data, 'patch_size', DEFAULT_PATCH_SIZE))
    class_names = getattr(cfg.data, 'class_names', None)
    if class_names is None:
        class_names = get_dataset_class_names(dataset_name, task=task, num_classes=num_classes)

    stride_z, stride_xy = resolve_sliding_window_strides(
        patch_size, stride_z=args.stride_z, stride_xy=args.stride_xy, step_size=args.step_size
    )

    if args.exp_dir:
        # 5-fold benchmark aggregation
        benchmark_5folds(
            model_builder=model_builder_helper,
            exp_dir=args.exp_dir,
            root_path=root_path,
            task=task,
            model_name=args.model,
            device=device,
            stride_z=stride_z,
            stride_xy=stride_xy,
            patch_size=patch_size,
            dataset_name=dataset_name,
            class_names=class_names,
            num_classes=num_classes,
            use_gaussian=not args.no_gaussian,
            use_tta=args.use_tta,
            checkpoint_preference=args.checkpoint_preference,
            allow_latest=args.allow_latest,
            include_pretrain=args.include_pretrain,
            use_amp=not args.no_amp,
            compute_surface=args.surface_metrics,
            eval_space=args.eval_space,
            raw_gt_dir=args.raw_gt_dir,
            on_missing=args.on_missing,
            step_size=args.step_size,
        )
    elif args.checkpoint:
        # Single checkpoint evaluation
        logging.info(f"Evaluating single checkpoint: {args.checkpoint} on fold {args.fold}...")
        from utils.evaluation_checkpoint import build_evaluation_model
        model, metadata, actual_model = build_evaluation_model(
            args.checkpoint, model_builder_helper, args.model, num_classes,
            fold=args.fold, task=task, dataset_name=dataset_name)
        model.to(device)
        model.eval()

        d_name = dataset_name
        val_dataset = DATASETS.build(
            d_name,
            base_dir=root_path,
            split='val',
            binary=is_binary,
            patch_size=patch_size,
            fold=args.fold,
            num_classes=num_classes,
            class_names=class_names,
            intensity_units=getattr(cfg.data, 'intensity_units', None),
            normalization=getattr(cfg.data, 'normalization', None)
        )

        details = {}
        mdice, per_class = evaluate_dataset_sliding_window(
            model=model,
            val_dataset=val_dataset,
            num_classes=num_classes,
            is_binary=is_binary,
            patch_size=patch_size,
            stride_z=stride_z,
            stride_xy=stride_xy,
            device=device,
            use_gaussian=not args.no_gaussian,
            use_tta=args.use_tta, use_amp=not args.no_amp, details_sink=details, compute_surface=args.surface_metrics,
            eval_space=args.eval_space, raw_gt_dir=args.raw_gt_dir, step_size=args.step_size
        )

        fg_names = class_names[1:] if len(class_names) > 1 else class_names
        summary_str = format_dice_summary(per_class, fg_names)
        print("\n" + "=" * 65)
        print(f"Single Checkpoint Evaluation Result:")
        print(f"Dataset:    {dataset_name.upper()} | Model: {actual_model} | Fold: {args.fold}")
        print(f"Mean Dice:  {mdice:.4f}")
        if details.get('macro_positive_dice') is not None:
            print(f"Macro Pos:  {details['macro_positive_dice']:.4f}")
        print(f"Per-Class:  {summary_str}")
        print("=" * 65)
        for row in details['per_class']:
            print(f"{class_names[row['class_id']]}: positive Dice={row['dice_positive_mean']}, "
                  f"positive cases={row['positive_cases']}, negative FP rate={row['negative_fp_case_rate']}")
        if args.metrics_output:
            from pathlib import Path
            output = dict(details, checkpoint=args.checkpoint, model=actual_model,
                          weights_source='ema_net' if 'ema_net' in metadata else 'net')
            Path(args.metrics_output).write_text(json.dumps(output, indent=2, allow_nan=False))

        if args.profile:
            input_shape = (1, 1, patch_size[0], patch_size[1], patch_size[2])
            metrics = profile_model_complexity(model, input_size=input_shape, device=device)
            print("\n" + format_complexity_summary(metrics))
    elif args.profile:
        # Direct profile mode without checkpoint
        model = model_builder_helper(args.model or 'vnet', in_channels=1, num_classes=num_classes)
        input_shape = (1, 1, patch_size[0], patch_size[1], patch_size[2])
        metrics = profile_model_complexity(model, input_size=input_shape, device=device)
        print("\n" + format_complexity_summary(metrics))
    else:
        parser.error("Provide --exp_dir, --checkpoint, or --profile")


if __name__ == "__main__":
    main()
