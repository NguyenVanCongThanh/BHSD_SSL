#!/usr/bin/env python3
"""
Single Unified Training Entrypoint for MedSSL-BHSD Framework.
Usage Examples:
    # Run PICK multi-class on fold 0:
    python train.py --config configs/methods/pick.yaml --fold 0

    # Run PICK binary task:
    python train.py --config configs/methods/pick.yaml --fold 0 --experiment.task binary --experiment.num_classes 2

    # Run Supervised baseline with UNet3D:
    python train.py --config configs/methods/supervised.yaml --model.name unet3d --fold 0

    # Run Mean Teacher:
    python train.py --config configs/methods/mean_teacher.yaml --fold 1 --logging.use_wandb true
"""

import os
import sys
import argparse
import logging
import torch

from core.registry import MODELS, METHODS
from core.config_parser import load_config
import models   # Trigger model registrations
import methods  # Trigger method registrations


def setup_logger(is_main: bool = True):
    log_level = logging.INFO if is_main else logging.WARN
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    parser = argparse.ArgumentParser(description="MedSSL-BHSD Unified Training Entrypoint")
    parser.add_argument('--config', type=str, default='configs/methods/pick.yaml', help='Path to YAML experiment config')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU ID (e.g. 0, 0,1, auto)')
    parser.add_argument('--fold', type=int, default=None, help='Fold override (0..4)')
    parser.add_argument('--task', type=str, default=None, choices=['multiclass', 'binary'], help='Task override')
    parser.add_argument('--resume', action='store_true', default=None, help='Resume training override')
    parser.add_argument('--ddp', action='store_true', help='Enable DistributedDataParallel (DDP)')

    known_args, unknown_args = parser.parse_known_args()

    # DDP detection: via --ddp flag or torchrun environment variables (RANK, WORLD_SIZE)
    from core.distributed import init_distributed_mode, cleanup_distributed, is_main_process
    is_ddp = init_distributed_mode(ddp_requested=known_args.ddp)
    setup_logger(is_main=is_main_process())

    # Apply GPU environment for non-DDP runs (preserving existing DP behavior as default)
    if not is_ddp and known_args.gpu not in ('auto', 'all', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(known_args.gpu)

    # Pass known overrides and unknown CLI args to config parser
    cli_overrides = list(unknown_args)
    if is_ddp:
        cli_overrides.extend(['--experiment.ddp', 'true'])
    if known_args.fold is not None:
        cli_overrides.extend(['--experiment.fold', str(known_args.fold)])
    if known_args.task is not None:
        num_cls = "2" if known_args.task == "binary" else "6"
        cli_overrides.extend(['--experiment.task', known_args.task, '--experiment.num_classes', num_cls])
    if known_args.resume is not None and known_args.resume:
        cli_overrides.extend(['--experiment.resume', 'true'])

    cfg = load_config(config_path=known_args.config, cli_args=cli_overrides)

    task = getattr(cfg.experiment, 'task', 'multiclass')
    fold = getattr(cfg.experiment, 'fold', 0)
    model_name = getattr(cfg.model, 'name', 'vnet')
    method_name = getattr(cfg.method, 'name', 'pick')

    logging.info("=" * 70)
    logging.info("=== MedSSL-BHSD SEMI-SUPERVISED MEDICAL IMAGE SEGMENTATION ===")
    logging.info(f"  Method:      {method_name.upper()}")
    logging.info(f"  Model:       {model_name}")
    logging.info(f"  Task:        {task.upper()} ({cfg.experiment.num_classes} classes)")
    logging.info(f"  Fold:        {fold}")
    logging.info(f"  Config:      {known_args.config}")
    logging.info(f"  Snapshot:    {cfg.experiment.exp_dir}/{cfg.experiment.name}/fold_{fold}")
    logging.info("=" * 70)

    # 1. Build Model
    in_channels = int(getattr(cfg.model, 'in_channels', 1))
    num_classes = int(getattr(cfg.experiment, 'num_classes', 6))
    has_dropout = bool(getattr(cfg.model, 'has_dropout', True))

    model_kwargs = {
        'in_channels': in_channels,
        'num_classes': num_classes,
        'has_dropout': has_dropout
    }
    # Optional parameters for VNet
    if hasattr(cfg.model, 'normalization'):
        model_kwargs['normalization'] = cfg.model.normalization

    model = MODELS.build(model_name, **model_kwargs)
    logging.info(f"[Registry] Built model '{model_name}' successfully.")

    # 2. Build Method / Trainer
    trainer = METHODS.build(method_name, model=model, cfg=cfg)
    logging.info(f"[Registry] Built trainer method '{method_name}' successfully.")

    # 3. Execute Training
    try:
        trainer.train()
    finally:
        if is_ddp:
            cleanup_distributed()


if __name__ == "__main__":
    main()
