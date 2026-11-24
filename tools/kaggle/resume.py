"""
Kaggle Resume Preparation CLI & Notebook Injection Helper for MedSSL-BHSD.

Default behavior (reproducible):
  `inject_resume(..., embed_local_patches=False)` imports `prepare_resume` directly from
  the cloned `GIT_COMMIT` on Kaggle without overwriting any repository files, and records
  `code_provenance="git_commit"` / `local_patches_embedded=False` in `resume_source`.

Opt-in experimental testing (`--embed-local-patches` / `embed_local_patches=True`):
  Explicitly embeds local `core/config_parser.py`, `utils/wandb_manager.py`,
  `utils/hf_storage.py`, and `prepare_resume` source for testing uncommitted edits on
  Kaggle, and records `code_provenance="local_patches_embedded"` /
  `local_patches_embedded=True` in `resume_source`.
"""
import argparse
import inspect
import json
import sys


def prepare_resume(
    run_path,
    checkpoint_kind,
    data_root,
    exp_dir,
    num_gpus,
    resume_mode="same-run",
    max_iteration=None,
    lr_max_iteration=None,
    local_patches_embedded=False,
):
    """Read the source configuration and stage an owned cloud checkpoint before training."""
    import copy
    import json
    import os
    import shutil
    import uuid
    from pathlib import Path
    import wandb
    import yaml

    if resume_mode not in ('same-run', 'new-run'):
        raise ValueError('Invalid resume mode')
    if run_path.count('/') == 0:
        api = wandb.Api()
        entity = (
            os.environ.get('MEDSSL_WANDB_ENTITY')
            or os.environ.get('WANDB_ENTITY')
            or getattr(api, 'default_entity', None)
            or 'default'
        )
        # Probe known projects for run ID
        candidate = None
        for proj in ('BHSD_Supervised', 'BHSD_MeanTeacher', 'BHSD_LeFeD', 'PICK_BHSD'):
            try:
                p = f"{entity}/{proj}/{run_path}"
                candidate = api.run(p)
                run_path = p
                break
            except Exception:
                continue
        if candidate is None:
            default_prefix = os.environ.get(
                'MEDSSL_WANDB_RUN_PREFIX',
                f"{entity}/BHSD_MeanTeacher/",
            )
            if not default_prefix.endswith('/'):
                default_prefix += '/'
            run_path = default_prefix + run_path
    elif run_path.count('/') == 1:
        api = wandb.Api()
        entity = (
            os.environ.get('MEDSSL_WANDB_ENTITY')
            or os.environ.get('WANDB_ENTITY')
            or getattr(api, 'default_entity', None)
            or 'default'
        )
        run_path = f"{entity}/{run_path}"
    if len(run_path.split('/')) != 3:
        raise ValueError('Expected run ID, project/id, or entity/project/id')
    run = wandb.Api().run(run_path)
    if run.state == 'running':
        raise RuntimeError('Source run is still running; stop it before resuming')
    cfg = copy.deepcopy(dict(run.config))
    method_name = cfg.get('method', {}).get('name')
    if method_name not in ('mean_teacher', 'supervised', 'lefed'):
        raise ValueError('Only Mean Teacher, Supervised, and LeFeD resume are supported')
    exp = cfg['experiment']
    layout = cfg.get('resolved', {}).get('batch_layout', {})
    if layout.get('world_size', 1) != max(1, num_gpus):
        raise ValueError('Resume requires the original GPU/world size')
    prefix = f"{exp['name']}_fold{exp['fold']}_{checkpoint_kind}:"
    candidates = [a for a in run.logged_artifacts() if a.name.startswith(prefix)]
    artifact = max(candidates, key=lambda a: int(a.name.rsplit(':v', 1)[1])) if candidates else None

    # If not on W&B, check if HF storage might have it
    run_name = getattr(run, 'name', None)
    if not artifact and not run_name:
        raise FileNotFoundError(f'No {checkpoint_kind} checkpoint uploaded by this run')

    source_name = exp['name']
    if resume_mode == 'new-run':
        exp['name'] = f"{source_name}_resume_{uuid.uuid4().hex[:8]}"
    destination = Path(exp_dir) / exp['name'] / f"fold_{exp['fold']}"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'checkpoint_latest.pth'
    if target.exists():
        raise FileExistsError(f'Refusing to replace an existing checkpoint: {target}')

    staged = False
    # Attempt restoration via Hugging Face Hub first if run has a valid name
    if run_name:
        try:
            from utils.hf_storage import HFStorageManager
            hf_mgr = HFStorageManager()
            if hf_mgr.enabled:
                hf_cand = "checkpoint_latest.pth" if checkpoint_kind == 'latest' else "best_model.pth"
                hf_res = hf_mgr.restore_checkpoint(hf_cand, str(destination), run_name=run_name)
                if hf_res and Path(hf_res).exists():
                    if Path(hf_res).resolve() != target.resolve():
                        shutil.copy2(hf_res, target)
                    staged = True
                    print(f"[RESUME] Checkpoint successfully restored from Hugging Face: {run_name}/{hf_cand}", flush=True)
                # Also restore best_model.pth if resuming latest to preserve best evaluation metric
                if checkpoint_kind == 'latest':
                    try:
                        hf_mgr.restore_checkpoint("best_model.pth", str(destination), run_name=run_name)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[RESUME WARNING] HF restore attempt encountered: {e}", flush=True)

    # Fallback to W&B logged artifacts
    if not staged:
        if not artifact:
            raise FileNotFoundError(f'No {checkpoint_kind} checkpoint uploaded by this run')
        downloaded = Path(artifact.download(root=str(destination / 'resume_download')))
        files = list(downloaded.rglob('*.pth'))
        if len(files) != 1:
            raise RuntimeError('Expected exactly one checkpoint file in artifact')
        shutil.copy2(files[0], target)
    entity, project, run_id = run_path.split('/')
    exp.update(resume=True, resume_policy='latest', exp_dir=str(exp_dir))
    cfg['data']['root_path'] = str(data_root)
    cfg['logging'].update(wandb_resume_new_run=(resume_mode == 'new-run'),
                          wandb_id=None if resume_mode == 'new-run' else run_id, wandb_entity=entity, wandb_project=project,
                          use_wandb=True, wandb_mode='online', wandb_required=True)
    if resume_mode == 'new-run':
        cfg['logging']['wandb_run_name'] = None
    art_name = artifact.name if artifact else f"{checkpoint_kind}_from_hf"
    art_meta = artifact.metadata if artifact else {}
    provenance = 'local_patches_embedded' if local_patches_embedded else 'git_commit'
    cfg['logging']['resume_source'] = {
        'run': run_path,
        'artifact': art_name,
        'metadata': art_meta,
        'mode': resume_mode,
        'local_patches_embedded': bool(local_patches_embedded),
        'code_provenance': provenance,
    }
    allowed_top_level = {'experiment', 'data', 'model', 'method', 'logging'}
    for key in list(cfg.keys()):
        if key not in allowed_top_level:
            cfg.pop(key, None)
    method_cfg = cfg.setdefault('method', {})
    source_max_iter = method_cfg.get('max_iteration')
    source_lr_max_iter = method_cfg.get('lr_max_iteration')
    effective_source_lr_max = source_lr_max_iter if source_lr_max_iter is not None else source_max_iter
    target_max_iter = int(max_iteration) if max_iteration is not None else source_max_iter

    if resume_mode == 'same-run':
        if (lr_max_iteration is not None and effective_source_lr_max is not None
                and int(lr_max_iteration) != int(effective_source_lr_max)):
            raise ValueError(
                f"Cannot change lr_max_iteration from {effective_source_lr_max} to {int(lr_max_iteration)} "
                f"in 'same-run' resume mode; use --resume-mode new-run instead."
            )
        if (source_lr_max_iter is None and source_max_iter is not None
                and target_max_iter is not None and int(target_max_iter) != int(source_max_iter)):
            method_cfg['lr_max_iteration'] = int(source_max_iter)
    elif (lr_max_iteration is not None and effective_source_lr_max is not None
            and int(lr_max_iteration) != int(effective_source_lr_max)):
        print(
            f"[RESUME WARNING] Overriding lr_max_iteration from {effective_source_lr_max} "
            f"to {int(lr_max_iteration)} in 'new-run' mode.",
            flush=True,
        )

    if max_iteration is not None:
        method_cfg['max_iteration'] = int(max_iteration)
    if lr_max_iteration is not None:
        method_cfg['lr_max_iteration'] = int(lr_max_iteration)

    run_summary = getattr(run, 'summary', None) or {}
    source_iter_num = art_meta.get('iter_num')
    if source_iter_num is None and hasattr(run_summary, 'get'):
        source_iter_num = run_summary.get('completed_iters') or run_summary.get('global_step')
    if source_iter_num is not None:
        source_iter_int = int(source_iter_num)
        if target_max_iter is not None and int(target_max_iter) <= source_iter_int:
            raise ValueError(
                f"Resume max_iteration ({target_max_iter}) must be greater than source iter_num ({source_iter_int})."
            )
        scheduler_type = str(method_cfg.get('lr_scheduler', 'poly')).lower()
        effective_target_lr_max = method_cfg.get('lr_max_iteration')
        if effective_target_lr_max is None:
            effective_target_lr_max = target_max_iter
        if (scheduler_type in ('poly', 'cosine') and effective_target_lr_max is not None
                and int(effective_target_lr_max) <= source_iter_int):
            raise ValueError(
                f"Resume lr_max_iteration ({effective_target_lr_max}) must be greater than source iter_num "
                f"({source_iter_int}) for '{scheduler_type}' LR schedule; otherwise LR is 0."
            )
    config_path = destination / 'resume_config.yaml'
    config_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    # Fail before launching any GPU workers if the staged config is incompatible.
    from core.config_parser import load_config
    load_config(config_path=str(config_path))
    (destination / 'resume_source.json').write_text(json.dumps({
        'run': run_path,
        'artifact': art_name,
        'metadata': art_meta,
        'mode': resume_mode,
        'local_patches_embedded': bool(local_patches_embedded),
        'code_provenance': provenance,
    }, indent=2), encoding='utf-8')
    meta_suffix = f"; metadata={art_meta}" if art_meta else ""
    print(f'[RESUME] {run_path}: {art_name}{meta_suffix}', flush=True)
    print(f'[RESUME] W&B mode: {resume_mode}; provenance: {provenance}; source history is preserved.', flush=True)
    args = ['--config', str(config_path), '--resume']
    if num_gpus >= 2:
        args.append('--ddp')
    return args, cfg


def inject_resume(
    notebook,
    run_path,
    checkpoint_kind,
    resume_mode="same-run",
    max_iteration=None,
    lr_max_iteration=None,
    embed_local_patches=False,
):
    """
    Patch only the generated training cell; fresh builds remain byte-for-byte unchanged.

    By default (`embed_local_patches=False`), invokes `tools.kaggle.resume.prepare_resume`
    from the cloned repository commit so runtime code matches `GIT_COMMIT` exactly.
    When `embed_local_patches=True` is explicitly requested, embeds local source files
    for experimental testing and marks provenance as `local_patches_embedded`.
    """
    import os
    marker = '# Cấu hình Anti-Deadlock'
    matches = [c for c in notebook['cells'] if c.get('cell_type') == 'code'
               and 'base_args = [' in ''.join(c['source']) and marker in ''.join(c['source'])]
    if len(matches) != 1:
        raise ValueError('Cannot identify the training cell')

    max_iter_override = max_iteration
    if max_iter_override is None:
        max_iter_env = os.environ.get('BHSD_MAX_ITER')
        max_iter_override = int(max_iter_env) if max_iter_env else None
    lr_max_iter_override = lr_max_iteration
    if lr_max_iter_override is None:
        lr_max_iter_env = os.environ.get('BHSD_LR_MAX_ITER')
        lr_max_iter_override = int(lr_max_iter_env) if lr_max_iter_env else None

    if embed_local_patches:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        code = 'from pathlib import Path\n'
        for relative_path in ('core/config_parser.py', 'utils/wandb_manager.py', 'utils/hf_storage.py'):
            source = (repo_root / relative_path).read_text(encoding='utf-8')
            code += f'Path({relative_path!r}).write_text({source!r}, encoding="utf-8")\n'
        code += inspect.getsource(prepare_resume) + '\n'
    else:
        code = 'from tools.kaggle.resume import prepare_resume\n'

    code += (
        '_resume_gpus = 1 if globals().get("FORCE_SINGLE_GPU", False) else num_gpus\n'
        f'base_args, resume_cfg = prepare_resume({run_path!r}, {checkpoint_kind!r}, '
        f'DATA_ROOT, EXP_DIR, _resume_gpus, resume_mode={resume_mode!r}, '
        f'max_iteration={max_iter_override!r}, lr_max_iteration={lr_max_iter_override!r}, '
        f'local_patches_embedded={bool(embed_local_patches)!r})\n'
        'EXP_NAME = resume_cfg["experiment"]["name"]\n'
        'FOLD_ID = resume_cfg["experiment"]["fold"]\n'
        'TASK = resume_cfg["experiment"]["task"]\n'
        'MODEL_NAME = resume_cfg["model"]["name"]\n'
        'WANDB_PROJECT = resume_cfg["logging"]["wandb_project"]\n\n'
    )
    cell = matches[0]
    cell['source'] = ''.join(cell['source']).replace(marker, code + marker, 1).splitlines(True)


def main(argv=None):
    """CLI entrypoint: `python -m tools.kaggle.resume --run <id> ...`."""
    parser = argparse.ArgumentParser(
        description="Stage a cloud checkpoint and resume_config.yaml from a prior W&B / HF run."
    )
    parser.add_argument("--run", required=True, help="Source run ID, project/id, or entity/project/id")
    parser.add_argument("--checkpoint", choices=["latest", "best"], default="latest", help="Checkpoint kind to restore")
    parser.add_argument("--data-root", required=True, help="Path to preprocessed BHSD dataset root")
    parser.add_argument("--exp-dir", required=True, help="Experiment output root directory")
    parser.add_argument("--num-gpus", type=int, default=2, help="World size / number of GPUs (must match source run)")
    parser.add_argument("--resume-mode", choices=["same-run", "new-run"], default="same-run", help="W&B tracking mode")
    parser.add_argument("--max-iteration", type=int, default=None, help="Optional target max_iteration override")
    parser.add_argument("--lr-max-iteration", type=int, default=None, help="Optional lr_max_iteration override")
    parser.add_argument(
        "--local-patches-embedded",
        action="store_true",
        help="Mark provenance metadata as containing embedded local patches",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write JSON payload with args and metadata")
    args = parser.parse_args(argv)

    train_args, cfg = prepare_resume(
        run_path=args.run,
        checkpoint_kind=args.checkpoint,
        data_root=args.data_root,
        exp_dir=args.exp_dir,
        num_gpus=args.num_gpus,
        resume_mode=args.resume_mode,
        max_iteration=args.max_iteration,
        lr_max_iteration=args.lr_max_iteration,
        local_patches_embedded=args.local_patches_embedded,
    )
    payload = {
        "args": train_args,
        "experiment": cfg["experiment"],
        "model": cfg["model"],
        "logging": cfg["logging"],
    }
    if args.output_json:
        from pathlib import Path
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
