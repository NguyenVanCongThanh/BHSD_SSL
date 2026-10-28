"""Timestamped local crawl snapshots, separate from tracked source code."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_root():
    configured = Path(os.environ.get('BHSD_EXPERIMENT_DATA', PROJECT_ROOT / 'experiment_data'))
    return configured.expanduser().resolve()


def new_snapshot(service, identifier, **metadata):
    parts = [service] + identifier.split('/')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', p) or p in ('.', '..') for p in parts):
        raise ValueError('Invalid snapshot service/identifier')
    now = datetime.now(timezone.utc)
    stamp = now.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    folder = data_root().joinpath(*parts, stamp)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'manifest.json', {
        'service': service, 'identifier': identifier,
        'crawled_at_utc': now.isoformat(), **metadata,
    })
    try:
        latest_sym = folder.parent / 'latest'
        if latest_sym.is_symlink() or latest_sym.exists():
            latest_sym.unlink()
        latest_sym.symlink_to(folder.name)
    except Exception:
        pass
    return folder


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def clean_data(obj):
    """Sanitize arbitrary objects for JSON serialization, preserving _step while filtering private keys."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "items"):
        return {str(k): clean_data(v) for k, v in obj.items() if not str(k).startswith("_") or k == "_step"}
    if isinstance(obj, (list, tuple)):
        return [clean_data(x) for x in obj]
    return str(obj)


def save_run_export(run, data):
    folder = new_snapshot('wandb', run.id, entity=run.entity, project=run.project,
                          run_id=run.id, run_created_at=str(run.created_at), url=run.url)
    write_json(folder / 'export.json', data)
    write_json(folder / 'config.json', data.get('config', {}))
    write_json(folder / 'summary.json', data.get('summary', {}))
    return folder


def comparison_file(path=None):
    """Explicit snapshot path wins; otherwise use the newest collection export."""
    if path:
        return Path(path)
    import argparse
    parser = argparse.ArgumentParser(description='Compare saved W&B runs offline')
    parser.add_argument('--input', type=Path, help='Path to a wandb_results.json collection snapshot')
    args, _ = parser.parse_known_args()
    if args.input:
        return args.input
    all_candidates = list(data_root().glob('wandb/_collections/*/wandb_results.json'))
    candidates = [p for p in all_candidates if p.parent.name != 'latest' and not p.parent.is_symlink()]
    if not candidates:
        candidates = all_candidates
    if not candidates:
        raise FileNotFoundError('No comparison snapshot; run tools/fetch_wandb.py or pass --input PATH')
    return max(candidates, key=lambda p: p.parent.name)
