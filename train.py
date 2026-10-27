#!/usr/bin/env python3
"""
Single Unified Training Entrypoint for MedSSL-BHSD Framework.
Supports composable Multi-Dataset, Multi-Model, and Multi-Method benchmarking.

Usage Examples:
    # 1. Composable benchmark (Dataset + Method + Model):
    python train.py --dataset bhsd_multiclass --method mean_teacher --model unet3d --fold 0

    # 2. Run PICK multi-class on fold 0 (Legacy / Standard config path):
    python train.py --config configs/methods/pick.yaml --fold 0

    # 3. Run Binary task:
    python train.py --dataset bhsd_binary --method pick --fold 0

    # 4. Benchmark Supervised baseline on custom / external dataset:
    python train.py --dataset configs/datasets/acdc.yaml --method supervised --model vnet --fold 0
"""

import os
import sys
import argparse
import logging
import torch

from core.registry import MODELS, METHODS, DATASETS
from core.config_parser import load_config
import data     # Trigger dataset registrations
import models   # Trigger model registrations
import methods  # Trigger method registrations


from utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description="MedSSL-BHSD Unified Benchmarking & Training Entrypoint")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML experiment config')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name or config (e.g. bhsd_multiclass, bhsd_binary, acdc, la_heart)')
    parser.add_argument('--method', type=str, default=None, help='Method name or config (e.g. pick, mean_teacher, supervised, cps, bcp, lefed)')
    parser.add_argument('--model', type=str, default=None, help='Model architecture (e.g. vnet, unet3d, unetr, nnunet3d)')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU ID (e.g. 0, 0,1, auto)')
    parser.add_argument('--fold', type=int, default=None, help='Fold override (0..4)')
    parser.add_argument('--task', type=str, default=None, choices=['multiclass', 'binary'], help='Task override')
    parser.add_argument('--resume', action='store_true', default=None, help='Resume training override')
    parser.add_argument('--ddp', action='store_true', help='Enable DistributedDataParallel (DDP)')

    known_args, unknown_args = parser.parse_known_args()

    # DDP detection: via --ddp flag or torchrun environment variables (RANK, WORLD_SIZE)
    from core.distributed import init_distributed_mode, cleanup_distributed, is_main_process
    is_ddp = init_distributed_mode(ddp_requested=known_args.ddp)
    get_logger(is_main=is_main_process())

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

    # Determine default config file if neither config nor method is given
    config_file = known_args.config
    if config_file is None and known_args.method is None and known_args.dataset is None:
        config_file = 'configs/methods/pick.yaml'

    cfg = load_config(
        config_path=config_file,
        dataset_config=known_args.dataset,
        model_config=known_args.model,
        method_config=known_args.method,
        cli_args=cli_overrides
    )

    dataset_name = getattr(cfg.data, 'name', 'bhsd')
    task = getattr(cfg.data, 'task', getattr(cfg.experiment, 'task', 'multiclass'))
    num_classes = int(getattr(cfg.data, 'num_classes', getattr(cfg.experiment, 'num_classes', 6)))
    fold = getattr(cfg.experiment, 'fold', 0)
    model_name = getattr(cfg.model, 'name', 'vnet')
    method_name = getattr(cfg.method, 'name', 'pick')

    logging.info("=" * 75)
    logging.info("=== MedSSL-BHSD SEMI-SUPERVISED MEDICAL IMAGE SEGMENTATION ===")
    logging.info(f"  Dataset:     {dataset_name.upper()} ({task.upper()}, {num_classes} classes)")
    logging.info(f"  Method:      {method_name.upper()}")
    logging.info(f"  Model:       {model_name}")
    logging.info(f"  Fold:        {fold}")
    logging.info(f"  Snapshot:    {cfg.experiment.exp_dir}/{cfg.experiment.name}/fold_{fold}")
    logging.info("=" * 75)

    from core.model_builder import model_spec_from_config, build_model_from_spec
    model_spec = model_spec_from_config(cfg.to_dict(), num_classes)
    cfg.model_spec = model_spec
    from core.distributed import get_rank
    model = build_model_from_spec(model_spec, seed=int(getattr(cfg.experiment, 'seed', 1337)),
                                  deterministic=bool(getattr(cfg.experiment, 'deterministic', True)),
                                  rank=get_rank())
    logging.info("[Registry] Built model '%s'", model_spec['name'])

    # 2. Build Method / Trainer
    trainer = METHODS.build(method_name, model=model, cfg=cfg)
    logging.info(f"[Registry] Built trainer method '{method_name}' successfully.")

    # 3. Execute Training
    try:
        trainer.train()
    except BaseException:
        raise
    else:
        if is_ddp:
            cleanup_distributed()


if __name__ == "__main__":
    main()
