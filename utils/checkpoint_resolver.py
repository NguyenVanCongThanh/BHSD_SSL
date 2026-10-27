"""Deterministic checkpoint discovery; phase priority precedes metric values."""
import logging
from pathlib import Path


def resolve_checkpoint(exp_dir, fold, model_name=None, prefer='best',
                       include_pretrain=False, allow_latest=False):
    """Return (path, kind). Latest fallback requires explicit authorization.

    The first phase containing any checkpoint owns the fold. This prevents an
    older root/pretrain best from replacing a self-training latest checkpoint.
    Legacy names are accepted, but their validation provenance is unverified.
    """
    if prefer not in ('best', 'latest'):
        raise ValueError('prefer must be best or latest')
    best_names = list(dict.fromkeys(['best_model.pth', 'VNet_best_model.pth'] +
                                  ([f'{model_name}_best_model.pth'] if model_name else [])))
    searched = []
    for fold_name in (f'fold_{fold}', f'fold{fold}'):
        root = Path(exp_dir) / fold_name
        directories = [root / 'self_train', root]
        if include_pretrain:
            directories.append(root / 'pre_train')
        for directory in directories:
            best = [directory / name for name in best_names]
            latest = directory / 'checkpoint_latest.pth'
            searched.extend(str(p) for p in best + [latest])
            if not any(p.is_file() for p in best + [latest]):
                continue
            if prefer == 'latest':
                if latest.is_file():
                    return str(latest), 'latest'
                break
            for path in best:
                if path.is_file():
                    return str(path), 'best'
            if allow_latest and latest.is_file():
                logging.warning('[Fold %s] Best missing in %s; explicitly falling back to latest', fold, directory)
                return str(latest), 'latest'
            break
    raise FileNotFoundError(f'Fold {fold}: requested {prefer} checkpoint unavailable. Searched: {searched}')
