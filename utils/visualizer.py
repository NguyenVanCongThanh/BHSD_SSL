"""Deterministic validation views; rendering never changes inference or metrics."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from evaluation.metrics import cal_multiclass_dice

PALETTE = {1: (230, 25, 75), 2: (60, 180, 75), 3: (255, 225, 25),
           4: (145, 30, 180), 5: (245, 130, 48)}
ERROR_COLORS = {'TP': (40, 200, 80), 'FN': (230, 40, 40),
                'FP': (255, 210, 30), 'wrong_class': (220, 40, 220)}


def standardize_2d_slice(arr, target_shape=(512, 512), is_mask=False, spacing=None):
    """Resize to a common canvas with physical aspect ratio preserved and padding."""
    physical = np.asarray(arr.shape, dtype=float) * np.asarray(spacing if spacing is not None else (1., 1.))
    factor = min(np.asarray(target_shape) / physical)
    shape = tuple(max(1, min(int(round(v * factor)), target_shape[i])) for i, v in enumerate(physical))
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()[None, None]
    mode = 'nearest' if is_mask else 'bilinear'
    resized = F.interpolate(tensor, size=shape, mode=mode,
                            align_corners=False if not is_mask else None)[0, 0].numpy()
    canvas = np.zeros(target_shape, dtype=np.uint8 if is_mask else np.float32)
    y, x = ((target_shape[i] - shape[i]) // 2 for i in range(2))
    canvas[y:y + shape[0], x:x + shape[1]] = resized
    return canvas


def display_intensity(image, metadata=None, settings=None):
    """Explicit HU conversion only; no guessing units from numeric ranges."""
    metadata, settings = metadata or {}, settings or {}
    image = np.asarray(image, dtype=np.float32)
    mode = settings.get('intensity_mode', 'percentile')
    if not np.isfinite(image).all():
        raise ValueError('Visualization image contains NaN/Inf')
    if mode == 'hu':
        if metadata.get('intensity_units') == 'HU':
            hu = image
        elif metadata.get('normalization') == 'zscore_hu':
            params = metadata.get('norm_params', {})
            mean, std = float(params['mean']), float(params['std'])
            if not np.isfinite([mean, std]).all() or std <= 0:
                raise ValueError('Invalid HU normalization metadata')
            hu = image * std + mean
        else:
            raise ValueError('HU display requires declared HU units or verified zscore_hu normalization')
        center, width = settings.get('window_center', 40.), settings.get('window_width', 80.)
        if not np.isfinite([center, width]).all() or width <= 0:
            raise ValueError('Window width must be finite and positive')
        low, high = center - width / 2, center + width / 2
        values = hu
    elif mode == 'percentile':
        bounds = settings.get('percentiles', [1., 99.])
        if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= 100:
            raise ValueError('Invalid display percentiles')
        low, high = np.percentile(image, bounds)
        values = image
    else:
        raise ValueError(f'Unknown intensity mode: {mode}')
    return np.clip((values - low) / max(float(high - low), 1e-6), 0., 1.)


def prepare_visual_record(record, settings=None):
    """Create bounded 2D media before distributed collection; retain no 3D arrays."""
    settings = settings or {}
    image, gt, pred = record['image'], record['label'], record['pred']
    if image.shape != gt.shape or gt.shape != pred.shape or image.ndim != 3:
        raise ValueError('Visualization requires matching 3D image, GT and prediction')
    case = str(record.get('case', 'sample'))
    counts = (gt > 0).sum(axis=(1, 2))
    default_z = int(np.argmax(counts)) if counts.max() else gt.shape[0] // 2
    z = int(settings.get('slice_indices', {}).get(case, default_z))
    if not 0 <= z < gt.shape[0]:
        raise ValueError(f'{case}: slice index out of bounds')
    metadata = record.get('metadata') or {}
    spacing = metadata.get('spacing')
    spacing_yx = tuple(spacing[1:]) if spacing is not None else None
    shape = tuple(settings.get('target_shape', [512, 512]))
    displayed = display_intensity(image, metadata, settings)[z]
    classes = int(record.get('num_classes', max(int(gt.max()), int(pred.max())) + 1))
    slice_dice = float(np.mean(cal_multiclass_dice(pred[z], gt[z], max(classes, 2))))
    result = {key: record[key] for key in ('case', 'dice') if key in record}
    result.update(z=z, slice_dice=slice_dice,
                  display_mode=settings.get('intensity_mode', 'percentile'),
                  image=standardize_2d_slice(displayed, shape, spacing=spacing_yx),
                  label=standardize_2d_slice(gt[z], shape, True, spacing_yx),
                  pred=standardize_2d_slice(pred[z], shape, True, spacing_yx))
    return result


def extract_best_slice(image_3d, mask_3d, pred_3d, target_shape=(512, 512)):
    record = prepare_visual_record({'image': image_3d, 'label': mask_3d, 'pred': pred_3d},
                                   {'target_shape': target_shape})
    return record['image'], record['label'], record['pred'], record['z']


def comparison_panel(record, class_labels, alpha=.5):
    """CT | GT | prediction | errors, with background left untouched."""
    if not 0 <= alpha <= 1:
        raise ValueError('Overlay alpha must be in [0,1]')
    base = np.repeat((record['image'][..., None] * 255).astype(np.uint8), 3, axis=2)
    def overlay(mask):
        rgb = base.copy()
        for cls in np.unique(mask):
            if cls == 0:
                continue
            if int(cls) not in PALETTE:
                raise ValueError(f'No configured color for foreground class {cls}')
            selected = mask == cls
            rgb[selected] = ((1 - alpha) * base[selected] + alpha * np.asarray(PALETTE[int(cls)])).astype(np.uint8)
        return rgb
    gt, pred = record['label'], record['pred']
    errors = base.copy()
    masks = {'TP': (gt > 0) & (gt == pred), 'FN': (gt > 0) & (pred == 0),
             'FP': (gt == 0) & (pred > 0), 'wrong_class': (gt > 0) & (pred > 0) & (gt != pred)}
    for name, mask in masks.items():
        errors[mask] = ERROR_COLORS[name]
    panels = [base, overlay(gt), overlay(pred), errors]
    height, width = base.shape[:2]
    canvas = Image.new('RGB', (4 * width, height + 44), 'black')
    draw = ImageDraw.Draw(canvas)
    for i, (title, panel) in enumerate(zip(('CT', 'GT', 'Prediction', 'TP green | FN red | FP yellow | wrong class magenta'), panels)):
        canvas.paste(Image.fromarray(panel), (i * width, 24))
        draw.text((i * width + 4, 4), title, fill='white')
    draw.text((4, height + 27), ' | '.join(f'{i}: {name}' for i, name in class_labels.items() if i > 0), fill='white')
    return np.asarray(canvas)


def create_wandb_segmentation_samples(visual_records, class_labels, settings=None, interactive_only=False):
    import wandb
    settings = settings or {}
    images = []
    for item in visual_records:
        record = item if item['image'].ndim == 2 else prepare_visual_record(item, settings)
        volume_dice = f"{float(np.mean(record['dice'])):.4f}" if 'dice' in record else None
        caption = (f"Case: {record.get('case')} | Z: {record['z']} | "
                   f"volume Dice 3D: {volume_dice} | slice Dice 2D: {record['slice_dice']:.4f} | "
                   f"display: {record['display_mode']}")
        if not interactive_only:
            images.append(wandb.Image(comparison_panel(record, class_labels, settings.get('alpha', .5)), caption=caption))
        else:
            images.append(wandb.Image((record['image'] * 255).astype(np.uint8), caption=caption + ' | interactive',
                masks={name: {'mask_data': record[key], 'class_labels': class_labels}
                       for name, key in [('ground_truth', 'label'), ('predictions', 'pred')]}))
    return images


def create_wandb_visual_payload(records, class_labels, settings=None):
    settings = settings or {}
    payload = {'val/predictions': create_wandb_segmentation_samples(records, class_labels, settings)}
    if settings.get('interactive', False):
        payload['val/predictions_interactive'] = create_wandb_segmentation_samples(
            records, class_labels, settings, interactive_only=True)
    return payload
