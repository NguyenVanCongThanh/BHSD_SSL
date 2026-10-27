"""Build evaluation models from saved training configuration and load strictly."""
import logging
import torch
from core.model_builder import model_spec_from_config, build_model_from_spec
from utils.checkpoint import load_checkpoint, validate_checkpoint_context


def build_evaluation_model(path, model_builder, model_name=None, num_classes=6, *, fold=None, task=None, dataset_name=None):
    state = torch.load(path, map_location='cpu', weights_only=True)
    validate_checkpoint_context(state, fold=fold, task=task, dataset_name=dataset_name)
    config = state.get('config') or {}
    saved_classes = (config.get('data') or {}).get('num_classes')
    if saved_classes is not None and int(saved_classes) != num_classes:
        raise ValueError(f'Evaluation expects {num_classes} classes but checkpoint config has {saved_classes}; select the matching dataset/task')
    model_config = dict(config.get('model') or {})
    saved_name = model_config.pop('name', None)
    if saved_name and config.get('method', {}).get('name') == 'pick' and saved_name == 'vnet':
        saved_name = 'pick_vnet'
    requested = 'pick_vnet' if model_name == 'pick' else model_name
    if saved_name and requested and saved_name.lower() != requested.lower():
        raise ValueError(f'Model override {requested} conflicts with checkpoint model {saved_name}')
    name = saved_name or requested
    if not name:
        raise ValueError('Legacy checkpoint has no model config; specify --model explicitly')
    if saved_name:
        spec = state.get('model_spec') or model_spec_from_config(config, num_classes, legacy=True)
        if spec['kwargs'].get('num_classes') != num_classes:
            raise ValueError('Saved model specification conflicts with evaluation class count')
        model = build_model_from_spec(spec)
    else:
        logging.warning('Legacy checkpoint: using explicit model %s with default architecture settings', name)
        model = model_builder(name, in_channels=1, num_classes=num_classes)
    metadata = load_checkpoint(path, model=model, device='cpu', strict=True, prefer_ema=True, checkpoint_state=state)
    kind = metadata.get('checkpoint_kind', 'unverified')
    logging.info('Loaded %s: provenance=%s, weights=%s', path, kind,
                 'ema_net' if 'ema_net' in metadata else 'net')
    return model, metadata, name
