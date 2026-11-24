"""
Native Physical Space Restoration for Medical 3D Segmentation (nnU-Net Alignment).
Reconstructs volumetric segmentations back to original patient physical space:
1. Inverse resampling: from preprocessed shape back to cropped shape.
2. Inverse cropping: placing cropped volume back into original bounding box in orig_shape.
3. Loading raw clinical NIfTI Ground Truth if available.
"""

from typing import Dict, Any, Optional, Tuple
import os
import ast
import logging
import numpy as np
from scipy.ndimage import zoom


def _parse_tuple_or_list(val: Any) -> Optional[Any]:
    if val is None:
        return None
    if isinstance(val, (tuple, list)):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return None
    return val


def restore_volume_to_native_space(
    volume: np.ndarray,
    metadata: Dict[str, Any],
    is_label: bool = True
) -> np.ndarray:
    """
    Inverse transform a preprocessed volume back to original patient physical space (like nnU-Net):
    1. Inverse resample: from volume.shape back to cropped_shape
    2. Inverse crop (pad): place cropped_shape back into orig_shape at bbox_used / crop_bbox

    Raises ValueError if required geometry metadata ('orig_shape' and 'bbox_used'/'crop_bbox')
    is missing so native-space evaluation never silently compares un-restored volumes.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Native-space restoration requires a metadata dict containing 'orig_shape' and 'bbox_used'/'crop_bbox'.")

    orig_shape = _parse_tuple_or_list(metadata.get('orig_shape'))
    bbox_used = _parse_tuple_or_list(metadata.get('bbox_used', metadata.get('crop_bbox')))
    cropped_shape = _parse_tuple_or_list(metadata.get('cropped_shape'))

    if orig_shape is None or bbox_used is None:
        raise ValueError(
            "Missing required native-space geometry metadata ('orig_shape' and 'bbox_used'/'crop_bbox'). "
            "Regenerate HDF5 files with tools/convert_option1_to_h5.py so .pkl geometry attributes are persisted."
        )

    orig_shape = tuple(int(x) for x in orig_shape)
    if cropped_shape is not None:
        cropped_shape = tuple(int(x) for x in cropped_shape)
    else:
        cropped_shape = tuple(int(hi) - int(lo) for lo, hi in bbox_used)

    # 1. Inverse resampling: from volume.shape back to cropped_shape
    if tuple(volume.shape) != cropped_shape:
        zoom_factors = [c / v for c, v in zip(cropped_shape, volume.shape)]
        order = 0 if is_label else 1
        resampled = zoom(volume, zoom_factors, order=order)
    else:
        resampled = volume

    # 2. Inverse cropping: Pad into original bounding box in orig_shape
    native_vol = np.zeros(orig_shape, dtype=volume.dtype)
    (z_min, z_max), (y_min, y_max), (x_min, x_max) = bbox_used

    rz, ry, rx = resampled.shape
    z_end = min(z_min + rz, orig_shape[0])
    y_end = min(y_min + ry, orig_shape[1])
    x_end = min(x_min + rx, orig_shape[2])

    native_vol[z_min:z_end, y_min:y_end, x_min:x_end] = resampled[:(z_end - z_min), :(y_end - y_min), :(x_end - x_min)]
    return native_vol


def load_raw_nifti_ground_truth(case_name: str, raw_gt_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Load raw NIfTI ground truth from clinical dataset folder in canonical (Z, Y, X) order.
    Both nibabel and SimpleITK paths return (Z, Y, X) matching preprocessed HDF5 volumes.
    """
    if not raw_gt_dir or not os.path.isdir(raw_gt_dir):
        return None

    candidates = [
        os.path.join(raw_gt_dir, f"{case_name}.nii.gz"),
        os.path.join(raw_gt_dir, f"{case_name}.nii"),
        os.path.join(raw_gt_dir, f"{case_name}_seg.nii.gz")
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                import nibabel as nib
                # nibabel stores NIfTI arrays in (X, Y, Z); transpose to canonical (Z, Y, X)
                arr_xyz = np.asarray(nib.load(path).dataobj, dtype=np.uint8)
                return arr_xyz.transpose(2, 1, 0)
            except ImportError:
                try:
                    import SimpleITK as sitk
                    img = sitk.ReadImage(path)
                    # SimpleITK.GetArrayFromImage already returns (Z, Y, X)
                    return sitk.GetArrayFromImage(img).astype(np.uint8)
                except Exception as e:
                    logging.warning(f"Failed to load NIfTI ground truth {path}: {e}")
                    return None
            except Exception as e:
                logging.warning(f"Error reading NIfTI file {path}: {e}")
                return None
    return None


def restore_case_to_native(
    pred: np.ndarray,
    label: np.ndarray,
    metadata: Dict[str, Any],
    case_name: str,
    raw_gt_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, ...]]]:
    """
    Restore prediction and ground truth to native patient physical space.
    If raw NIfTI ground truth exists in raw_gt_dir, uses it directly and validates shape alignment;
    otherwise restores preprocessed label via inverse geometry transform.
    """
    pred_native = restore_volume_to_native_space(pred, metadata, is_label=True)

    raw_gt = load_raw_nifti_ground_truth(case_name, raw_gt_dir)
    if raw_gt is not None:
        if raw_gt.shape != pred_native.shape:
            raise ValueError(
                f"Native-space shape mismatch for case '{case_name}': "
                f"restored prediction {pred_native.shape} != raw NIfTI GT {raw_gt.shape}."
            )
        label_native = raw_gt
    else:
        label_native = restore_volume_to_native_space(label, metadata, is_label=True)

    spacing_val = metadata.get('orig_spacing', metadata.get('spacing'))
    if isinstance(spacing_val, str):
        spacing_val = _parse_tuple_or_list(spacing_val)
    if spacing_val is not None:
        spacing_val = tuple(float(x) for x in spacing_val)

    return pred_native, label_native, spacing_val

