"""Shared model specification for training and checkpoint-based evaluation."""
import inspect
from core.registry import MODELS


def model_spec_from_config(config, num_classes, legacy=False):
    model = dict(config.get('model') or {})
    name = str(model.pop('name', 'vnet')).lower()
    method_name = str((config.get('method') or {}).get('name', '')).lower()

    # Resolve method class if registered to check aliases and capability contracts
    method_cls = None
    if method_name:
        try:
            import methods  # Ensure methods are registered
            from core.registry import METHODS
            method_cls = METHODS.get(method_name)
        except (KeyError, ImportError):
            method_cls = None

    # 1. Alias resolution
    if method_cls is not None and hasattr(method_cls, 'DEFAULT_MODEL_ALIAS'):
        name = method_cls.DEFAULT_MODEL_ALIAS.get(name, name)
    elif name == 'pick' or (name == 'vnet' and method_name == 'pick'):
        name = 'pick_vnet'
    elif name == 'lefed':
        name = 'lefed_net'

    model_cls = MODELS.get(name)
    if model_cls is None:
        raise KeyError(f"Model '{name}' is not registered in MODELS registry")

    # 2. Capability verification
    if method_cls is not None and hasattr(method_cls, 'REQUIRED_MODEL_CAPABILITIES'):
        req_caps = set(getattr(method_cls, 'REQUIRED_MODEL_CAPABILITIES', set()))
        model_caps = set(getattr(model_cls, 'CAPABILITIES', {'segmentation'}))
        missing = req_caps - model_caps
        if missing:
            if method_name == 'pick':
                raise ValueError('PICK requires pick_vnet; the configured backbone does not provide MIM/aux branches')
            elif method_name == 'lefed':
                raise ValueError('LeFeD requires lefed_net')
            else:
                raise ValueError(f"Method '{method_name}' requires capabilities {sorted(list(missing))} not provided by model '{name}'")
    elif method_name == 'pick' and name != 'pick_vnet':
        raise ValueError('PICK requires pick_vnet; the configured backbone does not provide MIM/aux branches')
    elif method_name == 'lefed' and name != 'lefed_net':
        raise ValueError('LeFeD requires lefed_net')

    signature = inspect.signature(model_cls.__init__)
    explicit_params = {
        k: p for k, p in signature.parameters.items()
        if k != 'self' and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    if legacy:
        # Older train.py forwarded only these fields, even if config held more.
        keys = {'in_channels', 'has_dropout', 'normalization', 'deep_supervision', 'base_filters', 'num_stages'}
        kwargs = {key: value for key, value in model.items() if key in keys and key in signature.parameters}
    else:
        # Strict parameter validation for modern execution (reject typos/unknown kwargs)
        unsupported = set(model.keys()) - set(explicit_params.keys())
        if unsupported:
            raise ValueError(
                f"Unsupported model parameter(s) for '{name}': {sorted(list(unsupported))}. "
                f"Allowed parameters are: {sorted(list(explicit_params.keys()))}"
            )
        kwargs = dict(model)
    kwargs.setdefault('in_channels', 1)
    kwargs['num_classes'] = num_classes
    if not legacy and 'img_shape' in explicit_params:
        kwargs.setdefault('img_shape', tuple(config.get('data', {}).get('patch_size', (32, 160, 160))))
    return {'name': name, 'kwargs': kwargs}


def build_model_from_spec(spec, *, seed=None, deterministic=True, rank=0):
    if seed is not None:
        from utils.reproducibility import set_seed
        set_seed(seed=seed, deterministic=deterministic, rank=rank)
    return MODELS.build(spec['name'], **spec['kwargs'])
