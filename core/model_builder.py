"""Shared model specification for training and checkpoint-based evaluation."""
import inspect
from core.registry import MODELS


def model_spec_from_config(config, num_classes, legacy=False):
    model = dict(config.get('model') or {})
    name = str(model.pop('name', 'vnet')).lower()
    method = str((config.get('method') or {}).get('name', '')).lower()
    if name == 'pick' or (name == 'vnet' and method == 'pick'):
        name = 'pick_vnet'
    if method == 'pick' and name != 'pick_vnet':
        raise ValueError('PICK requires pick_vnet; the configured backbone does not provide MIM/aux branches')
    if method == 'lefed' and name != 'lefed_net':
        raise ValueError('LeFeD requires lefed_net')
    signature = inspect.signature(MODELS.get(name).__init__)
    if legacy:
        # Older train.py forwarded only these fields, even if config held more.
        keys = {'in_channels', 'has_dropout', 'normalization', 'deep_supervision', 'base_filters', 'num_stages'}
        model = {key: value for key, value in model.items() if key in keys}
    kwargs = {key: value for key, value in model.items() if key in signature.parameters}
    kwargs.setdefault('in_channels', 1)
    kwargs['num_classes'] = num_classes
    if not legacy and 'img_shape' in signature.parameters:
        kwargs.setdefault('img_shape', tuple(config.get('data', {}).get('patch_size', (32, 160, 160))))
    return {'name': name, 'kwargs': kwargs}


def build_model_from_spec(spec, *, seed=None, deterministic=True, rank=0):
    if seed is not None:
        from utils.reproducibility import set_seed
        set_seed(seed=seed, deterministic=deterministic, rank=rank)
    return MODELS.build(spec['name'], **spec['kwargs'])
