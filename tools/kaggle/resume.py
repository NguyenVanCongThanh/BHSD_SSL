"""Embed opt-in resume preparation in a Kaggle bundle, without changing templates."""
import inspect


def prepare_resume(run_path, checkpoint_kind, data_root, exp_dir, num_gpus, resume_mode="same-run"):
    """Read the source configuration and stage an owned cloud checkpoint before training."""
    import copy
    import json
    import shutil
    import uuid
    from pathlib import Path
    import wandb
    import yaml

    if resume_mode not in ('same-run', 'new-run'):
        raise ValueError('Invalid resume mode')
    if '/' not in run_path:
        run_path = ('thanhnguyen2005243-ho-chi-minh-city-university-of-technology/'
                    'BHSD_MeanTeacher/' + run_path)
    if len(run_path.split('/')) != 3:
        raise ValueError('Expected run ID or entity/project/id')
    run = wandb.Api().run(run_path)
    if run.state == 'running':
        raise RuntimeError('Source run is still running; stop it before resuming')
    cfg = copy.deepcopy(dict(run.config))
    if cfg.get('method', {}).get('name') != 'mean_teacher':
        raise ValueError('Only Mean Teacher resume is supported')
    exp = cfg['experiment']
    layout = cfg.get('resolved', {}).get('batch_layout', {})
    if layout.get('world_size', 1) != max(1, num_gpus):
        raise ValueError('Resume requires the original GPU/world size')
    prefix = f"{exp['name']}_fold{exp['fold']}_{checkpoint_kind}:"
    candidates = [a for a in run.logged_artifacts() if a.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f'No {checkpoint_kind} checkpoint uploaded by this run')
    artifact = max(candidates, key=lambda a: int(a.name.rsplit(':v', 1)[1]))
    source_name = exp['name']
    if resume_mode == 'new-run':
        exp['name'] = f"{source_name}_resume_{uuid.uuid4().hex[:8]}"
    destination = Path(exp_dir) / exp['name'] / f"fold_{exp['fold']}"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'checkpoint_latest.pth'
    if target.exists():
        raise FileExistsError(f'Refusing to replace an existing checkpoint: {target}')
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
    cfg['logging']['resume_source'] = {
        'run': run_path, 'artifact': artifact.name, 'metadata': artifact.metadata,
        'mode': resume_mode,
    }
    cfg.pop('resolved', None)
    config_path = destination / 'resume_config.yaml'
    config_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    (destination / 'resume_source.json').write_text(json.dumps({
        'run': run_path, 'artifact': artifact.name, 'metadata': artifact.metadata,
    }, indent=2), encoding='utf-8')
    print(f'[RESUME] {run_path}: {artifact.name}; metadata={artifact.metadata}', flush=True)
    print(f'[RESUME] W&B mode: {resume_mode}; source history is preserved.', flush=True)
    args = ['--config', str(config_path), '--resume']
    if num_gpus >= 2:
        args.append('--ddp')
    return args, cfg


def inject_resume(notebook, run_path, checkpoint_kind, resume_mode="same-run"):
    """Patch only the generated training cell; fresh builds remain byte-for-byte unchanged."""
    marker = '# Cấu hình Anti-Deadlock'
    matches = [c for c in notebook['cells'] if c.get('cell_type') == 'code'
               and 'base_args = [' in ''.join(c['source']) and marker in ''.join(c['source'])]
    if len(matches) != 1:
        raise ValueError('Cannot identify the Mean Teacher training cell')
    code = ''
    if resume_mode == 'new-run':
        # Bundle the tracking implementation used by this CLI, since Kaggle clones GitHub.
        from pathlib import Path
        manager_source = (Path(__file__).resolve().parents[2] / 'utils/wandb_manager.py').read_text(encoding='utf-8')
        code += ('from pathlib import Path\n'
                 f'Path("utils/wandb_manager.py").write_text({manager_source!r}, encoding="utf-8")\n')
    code += inspect.getsource(prepare_resume) + '\n'
    code += (f'base_args, resume_cfg = prepare_resume({run_path!r}, {checkpoint_kind!r}, '
             f'DATA_ROOT, EXP_DIR, num_gpus, resume_mode={resume_mode!r})\n'
             'EXP_NAME = resume_cfg["experiment"]["name"]\n'
             'FOLD_ID = resume_cfg["experiment"]["fold"]\n'
             'TASK = resume_cfg["experiment"]["task"]\n'
             'MODEL_NAME = resume_cfg["model"]["name"]\n'
             'WANDB_PROJECT = resume_cfg["logging"]["wandb_project"]\n\n')
    cell = matches[0]
    cell['source'] = ''.join(cell['source']).replace(marker, code + marker, 1).splitlines(True)
