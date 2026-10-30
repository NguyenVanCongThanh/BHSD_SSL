"""Reproducible, read-only BHSD profiling helpers. Distances use metadata spacing.

Cached rows are invalidated by input size/mtime and this module's content hash.
Unlabeled placeholder masks are audited, never treated as negative ground truth.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes, mesh_surface_area

CLASS_NAMES = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH']
MODULE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parse(value):
    if isinstance(value, (str, bytes)):
        return ast.literal_eval(value.decode() if isinstance(value, bytes) else value)
    return value


def split_audit(root):
    root = Path(root)
    names = ['train', 'val', 'test', 'train_labeled', 'train_unlabeled']
    lists = {k: (root / f'{k}.txt').read_text().split() for k in names}
    sets = {k: set(v) for k, v in lists.items()}
    available = {p.stem for p in (root / 'data').glob('*.h5')}
    assert all(len(lists[k]) == len(sets[k]) for k in names), 'Duplicate split IDs'
    assert not sets['train'] & sets['val'], 'Train/validation overlap'
    assert not sets['train_labeled'] & sets['train_unlabeled'], 'Labeled/unlabeled overlap'
    assert sets['train'] == sets['train_labeled'] | sets['train_unlabeled']
    assert sets['train'] | sets['val'] == available, 'Missing or unassigned H5 cases'
    assert sets['test'] <= available
    # Patient-level independence requires a trusted patient identifier, not just case IDs.
    return lists, available


def fingerprint(paths):
    return {'module': MODULE_HASH, 'files': [
        [str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths if p.exists()
    ]}


def morphology(mask, spacing, case_id, class_id):
    """Aggregate one case/class, including all disconnected components.

    Padded marching cubes includes boundary surfaces. Covariance uses voxel-center
    physical coordinates; one-voxel masks retain a row with undefined PCA ratios.
    """
    coords = np.argwhere(mask)
    lo, hi = coords.min(0), coords.max(0) + 1
    sub = mask[tuple(slice(a, b) for a, b in zip(lo, hi))]
    vertices, faces, _, _ = marching_cubes(np.pad(sub.astype(np.uint8), 1), .5, spacing=spacing)
    area_mm2 = float(mesh_surface_area(vertices, faces))
    volume_mm3 = len(coords) * float(np.prod(spacing))
    _, n_components = ndimage.label(sub, structure=np.ones((3, 3, 3)))
    eig = np.linalg.eigvalsh(np.cov((coords * spacing).T))[::-1] if len(coords) >= 3 else np.full(3, np.nan)
    eig = np.maximum(eig, 0)
    return dict(case_id=case_id, class_id=class_id, class_name=CLASS_NAMES[class_id-1],
                voxels=len(coords), vol_ml=volume_mm3/1000, surface_area_cm2=area_mm2/100,
                sphericity=np.pi**(1/3)*(6*volume_mm3)**(2/3)/area_mm2,
                savr=10*area_mm2/volume_mm3, savr_unit='cm^-1',
                elongation=1-np.sqrt(eig[1]/eig[0]) if eig[0] > 0 else np.nan,
                flatness=1-np.sqrt(eig[2]/eig[1]) if eig[1] > 0 else np.nan,
                extent=len(coords)/np.prod(hi-lo), components_26=n_components,
                pca_defined=bool(eig[1] > 0))


def profile_case(case_id, root, raw_root, cache_dir, split):
    root, raw_root, cache_dir = map(Path, (root, raw_root, cache_dir))
    hp = root / 'data' / f'{case_id}.h5'
    gp = raw_root / 'ground truths' / f'{case_id}.nii.gz'
    ip = raw_root / 'images' / f'{case_id}.nii.gz'
    labeled = split != 'train_unlabeled'
    key = fingerprint([hp] + ([gp, ip] if labeled else []))
    cp = cache_dir / f'{case_id}.json'
    if cp.exists():
        cached = json.loads(cp.read_text())
        if cached['fingerprint'] == key and cached['summary']['split'] == split:
            return cached
    row = dict(case_id=case_id, split=split, is_labeled=labeled)
    errors, morph, retention = [], [], []
    with h5py.File(hp) as f:
        attrs = dict(f.attrs)
        image, label = f['image'], f['label']
        shape = image.shape
        if image.ndim != 3 or label.shape != shape:
            raise ValueError(f'{case_id}: image/label shape mismatch')
        spacing = np.array(parse(attrs['target_spacing']), dtype=float)
        original = np.array(parse(attrs['orig_shape']), dtype=int)
        orig_spacing = np.array(parse(attrs['orig_spacing']), dtype=float)
        bbox = np.array(parse(attrs['bbox_used']), dtype=int)
        cropped = np.array(parse(attrs['cropped_shape']), dtype=int)
        if not (np.isfinite(spacing).all() and np.all(spacing > 0) and np.isfinite(orig_spacing).all() and np.all(orig_spacing > 0)):
            raise ValueError(f'{case_id}: invalid spacing')
        checks = {
            'final_shape': tuple(parse(attrs['final_shape'])) == shape,
            'cropped_shape': np.array_equal(bbox[:, 1]-bbox[:, 0], cropped),
            'bbox_bounds': bool(np.all(bbox[:, 0] >= 0) and np.all(bbox[:, 1] <= original) and np.all(cropped > 0)),
            'is_labeled': bool(parse(attrs['is_labeled'])) == labeled,
            'has_label': bool(parse(attrs['has_label'])) == labeled,
            'case_id': attrs['case_id'] == case_id,
            'resampled_shape': bool(np.all(np.abs(np.array(shape)-np.round(cropped*orig_spacing/spacing)) <= 1)),
        }
        errors.extend(k for k, ok in checks.items() if not ok)
        for j, axis in enumerate('zyx'):
            row.update({f'shape_{axis}': shape[j], f'orig_shape_{axis}': original[j],
                        f'orig_spacing_{axis}': orig_spacing[j], f'target_spacing_{axis}': spacing[j],
                        f'bbox_{axis}_len': cropped[j]})
        n, total, squares, nonfinite = 0, 0., 0., 0
        minimum, maximum = np.inf, -np.inf
        # Slab-wise reading bounds memory even for large scans.
        for z in range(0, shape[0], 8):
            slab = np.asarray(image[z:z+8], dtype=np.float64)
            good = np.isfinite(slab)
            nonfinite += int((~good).sum())
            vals = slab[good]
            n += vals.size
            total += vals.sum()
            squares += np.square(vals).sum()
            if vals.size:
                minimum, maximum = min(minimum, vals.min()), max(maximum, vals.max())
        row.update(total_voxels=int(np.prod(shape)), img_min=minimum, img_max=maximum,
                   img_mean=total/n if n else np.nan,
                   img_std=np.sqrt(max(0, squares/n-(total/n)**2)) if n else np.nan,
                   nonfinite_voxels=nonfinite,
                   crop_reduction_pct=100*(1-np.prod(cropped)/np.prod(original)))
        if nonfinite:
            errors.append('nonfinite_image')
        clip = parse(attrs['clip_range'])
        norm = parse(attrs['norm_params'])
        row.update(clip_min=clip[0], clip_max=clip[1], norm_mean=norm['mean'], norm_std=norm['std'])
        gt = label[:]
        unique = np.unique(gt)
        if not np.isin(unique, np.arange(6)).all():
            errors.append('label_values')
        row['label_values'] = ','.join(map(str, unique))
        if not labeled and np.any(gt != 0):
            errors.append('nonzero_unlabeled_placeholder')
        row['metadata_errors'] = ';'.join(errors)
        for c in range(6):
            count = int(np.count_nonzero(gt == c)) if labeled else None
            row[f'c{c}_voxels'] = count
            if c:
                row[f'c{c}_ml'] = count*np.prod(spacing)/1000 if labeled else None
                row[f'has_c{c}'] = bool(count) if labeled else None
        row['lesion_voxels'] = int(np.count_nonzero(gt > 0)) if labeled else None
        row['lesion_ml'] = row['lesion_voxels']*np.prod(spacing)/1000 if labeled else None
        row['has_lesion'] = bool(row['lesion_voxels']) if labeled else None
        row['num_lesion_types'] = sum(row[f'has_c{c}'] for c in range(1, 6)) if labeled else None
        row['lesion_touches_h5_boundary'] = bool(any(np.any(face > 0) for face in
             (gt[0], gt[-1], gt[:, 0], gt[:, -1], gt[:, :, 0], gt[:, :, -1]))) if labeled else None
        if labeled:
            for c in range(1, 6):
                if row[f'has_c{c}']:
                    morph.append(morphology(gt == c, spacing, case_id, c))
            if not gp.exists() or not ip.exists():
                row['raw_status'] = 'missing_raw_pair'
            else:
                raw_n = nib.load(gp)
                raw_i = nib.load(ip)
                raw = np.asarray(raw_n.dataobj).transpose(2, 1, 0)
                # BHSD H5 metadata is Z,Y,X; require shape/spacing agreement.
                geometry_ok = (np.array_equal(raw.shape, original) and raw_n.shape == raw_i.shape
                    and np.allclose(raw_n.affine, raw_i.affine)
                    and np.allclose(raw_n.header.get_zooms()[::-1], orig_spacing, rtol=1e-4, atol=1e-4))
                row['raw_status'] = 'metadata_consistent' if geometry_ok else 'geometry_mismatch'
                row['raw_axis_codes'] = ''.join(nib.aff2axcodes(raw_n.affine))
                row['raw_affine_voxel_mm3'] = abs(np.linalg.det(raw_n.affine[:3, :3]))
                row['raw_spacing_voxel_mm3'] = float(np.prod(orig_spacing))
                if geometry_ok:
                    raw_crop = raw[tuple(slice(a, b) for a, b in bbox)]
                    # Diagnostic only: exact preprocessing interpolation provenance is unavailable.
                    reconstructed = ndimage.zoom(raw_crop, np.array(shape)/cropped, order=0, prefilter=False)
                    for c in range(1, 6):
                        before = int(np.count_nonzero(raw == c))
                        after_crop = int(np.count_nonzero(raw_crop == c))
                        after = row[f'c{c}_voxels']
                        intersection = int(np.count_nonzero((reconstructed == c) & (gt == c)))
                        denom = int(np.count_nonzero(reconstructed == c)) + after
                        retention.append(dict(case_id=case_id, split=split, class_id=c, class_name=CLASS_NAMES[c-1],
                            raw_voxels=before, cropped_voxels=after_crop, h5_voxels=after,
                            cropped_out_voxels=before-after_crop, disappeared=bool(before and not after),
                            appeared=bool(after and not before),
                            raw_ml=before*np.prod(orig_spacing)/1000,
                            h5_ml=row[f'c{c}_ml'],
                            raw_affine_ml=before*row['raw_affine_voxel_mm3']/1000,
                            volume_change_pct=100*(row[f'c{c}_ml']/(before*np.prod(orig_spacing)/1000)-1) if before else np.nan,
                            diagnostic_forward_dice=2*intersection/denom if denom else np.nan))
    result = dict(fingerprint=key, summary=row, morphology=morph, retention=retention)
    cp.write_text(json.dumps(result, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    return result


def sample_case(case_id, root, trials=100, patch_sizes=((32,160,160), (32,192,192), (32,224,224)), probabilities=(0., .33, .5), seed=42):
    """Use the production RandomCrop, including padding, jitter and empty-GT fallback."""
    from data.transforms3d import RandomCrop
    with h5py.File(Path(root)/'data'/f'{case_id}.h5') as f:
        label = f['label'][:]
    # Image intensities do not affect sampling: use the label as a lightweight image.
    state = np.random.get_state()
    rows = []
    try:
        for patch in patch_sizes:
            pad = np.maximum(0, np.array(patch)-label.shape)
            padded_shape = np.array(label.shape)+pad
            # Pad once; production transform sees identical padded geometry each trial.
            padded = np.pad(label, [(int(p//2), int(p-p//2)) for p in pad])
            for probability in probabilities:
                np.random.seed(seed)
                crop = RandomCrop(tuple(patch), foreground_prob=probability)
                hits = np.zeros(6, int)
                fractions = []
                for _ in range(trials):
                    out = crop({'image': padded, 'label': padded})['label']
                    hits[0] += np.any(out > 0)
                    for c in range(1, 6):
                        hits[c] += np.any(out == c)
                    fractions.append(np.mean(out > 0))
                row = dict(case_id=case_id, patch='x'.join(map(str, patch)), foreground_prob=probability,
                           trials=trials, any_hit_rate=hits[0]/trials, fg_fraction=np.mean(fractions),
                           padding_fraction=1-np.prod(label.shape)/np.prod(padded_shape),
                           patch_voxels=int(np.prod(patch)))
                for c in range(1,6):
                    row[f'c{c}_present'] = bool(np.any(label == c))
                    row[f'c{c}_hit_rate'] = hits[c]/trials
                rows.append(row)
    finally:
        np.random.set_state(state)
    return rows


def build_tables(root, raw_root, output, workers=2):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    from itertools import repeat
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output/'eda_cache'
    cache.mkdir(exist_ok=True)
    lists, available = split_audit(root)
    membership = {case: split for split in ['train_labeled', 'val', 'train_unlabeled'] for case in lists[split]}
    cases = sorted(available)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(profile_case, cases, repeat(str(root)), repeat(str(raw_root)), repeat(str(cache)),
                            [membership[c] for c in cases], chunksize=1)
        for i, result in enumerate(iterator, 1):
            results.append(result)
            if i % 100 == 0 or i == len(cases):
                print(f'Profiled {i}/{len(cases)} H5 cases', flush=True)
    summary = pd.DataFrame([r['summary'] for r in results])
    shape = pd.DataFrame([m for r in results for m in r['morphology']])
    retention = pd.DataFrame([m for r in results for m in r['retention']])
    for name, table in [('bhsd_eda_summary', summary), ('bhsd_shape_morphology', shape), ('bhsd_preprocessing_audit', retention)]:
        table.to_csv(output/f'{name}.csv', index=False)
    manifest = dict(dataset=str(Path(root).resolve()), raw_root=str(Path(raw_root).resolve()),
                    module_sha256=MODULE_HASH, cases=len(summary), labeled=int(summary.is_labeled.sum()),
                    unlabeled=int((~summary.is_labeled).sum()),
                    split_sha256={k: hashlib.sha256((Path(root)/f'{k}.txt').read_bytes()).hexdigest() for k in lists},
                    scope='All H5 volumes; raw GT and morphology for labeled cases only')
    (output/'bhsd_eda_manifest.json').write_text(json.dumps(manifest, indent=2))
    return summary, shape, retention


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path.home()/'dataset/BHSD_fold0')
    parser.add_argument('--raw', type=Path, default=Path.home()/'dataset/BHSD/label_192/label_192')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'notebooks')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    build_tables(args.dataset, args.raw, args.output, args.workers)
