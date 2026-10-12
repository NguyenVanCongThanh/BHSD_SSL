#!/usr/bin/env python3
"""
Standalone CLI Tool to Profile Model Efficiency & Computational Complexity.
Enables fast, zero-data benchmarking of neural network architectures (VNet, UNet3D, UNETR, etc.)
with arbitrary 3D patch dimensions.

Usage Examples:
    # Profile default VNet with standard patch size:
    python tools/profile_model.py --model vnet

    # Profile UNet3D with binary output on custom patch:
    python tools/profile_model.py --model unet3d --num_classes 2 --patch_size 32 160 160

    # Profile on specific GPU:
    python tools/profile_model.py --model vnet --gpu 0
"""

import os
import sys
import argparse
import logging
import torch

# Ensure PICK root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.registry import MODELS
from evaluation.complexity import profile_model_complexity, format_complexity_summary
import models  # Trigger model registry


def main():
    parser = argparse.ArgumentParser(description="MedSSL-BHSD Model Efficiency & Complexity Profiler")
    parser.add_argument('--model', type=str, default='vnet', help='Model architecture name (vnet, unet3d, unetr, unet, etc.)')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of output segmentation classes (e.g. 6 for multiclass, 2 for binary)')
    parser.add_argument('--patch_size', nargs=3, type=int, default=[32, 160, 160], help='Input patch dimensions (Z Y X)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for profiling')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU ID (0, 1, or auto/cpu)')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=50, help='Number of timed runs')
    args = parser.parse_args()

    # Device handling
    if args.gpu not in ('auto', 'all', None, 'cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.gpu == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    print(f"\n[Profiler] Building model '{args.model}' (in_channels={args.in_channels}, num_classes={args.num_classes})...")
    model = MODELS.build(
        args.model,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        has_dropout=False
    ).to(device)

    input_shape = (args.batch_size, args.in_channels, args.patch_size[0], args.patch_size[1], args.patch_size[2])
    print(f"[Profiler] Running benchmark on {device} with input shape {list(input_shape)}...")

    metrics = profile_model_complexity(
        model=model,
        input_size=input_shape,
        device=device,
        num_warmup=args.warmup,
        num_runs=args.runs
    )

    print("\n" + format_complexity_summary(metrics))


if __name__ == '__main__':
    main()
