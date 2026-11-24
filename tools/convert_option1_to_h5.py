#!/usr/bin/env python3
"""
Convert nnUNet Option 1 preprocessed data (.b2nd / .pkl) to HDF5 (.h5) format for PICK.
This script is strictly READ-ONLY on input files and writes new .h5 files to output_dir.
It preserves the original 6-class labels (0..5) so that both Binary and Multi-class
tasks can share the same dataset.
"""

import os
import sys
import glob
import json
import pickle
import argparse
import numpy as np
import h5py
from tqdm import tqdm

try:
    import blosc2
    HAS_BLOSC2 = True
except ImportError:
    HAS_BLOSC2 = False


def load_b2nd_array(file_path):
    """Load an array from a blosc2 file (.b2nd)."""
    if not HAS_BLOSC2:
        raise ImportError(
            "The 'blosc2' Python library is required to read .b2nd files. "
            "Please install it with: pip install blosc2"
        )
    try:
        arr = blosc2.open(file_path, mode='r')
        return arr[:]
    except Exception as e:
        try:
            return blosc2.load_array(file_path)
        except Exception:
            raise RuntimeError(f"Failed to read blosc2 file {file_path}: {e}")


def _to_serializable(val):
    """Convert numpy scalars/arrays and nested tuples into plain JSON/literal-safe Python values."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (list, tuple)):
        return [_to_serializable(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}
    return val


def extract_pkl_metadata(pkl_path, final_shape=None, target_spacing=None):
    """
    Extract native-space spatial geometry and orientation metadata from nnU-Net v2 case .pkl.
    Persists canonical keys required by BHSDDataset and evaluation/native_space.py:
      - orig_shape, bbox_used, crop_bbox, cropped_shape, final_shape
      - orig_spacing, target_spacing
      - itk_origin, itk_spacing, itk_direction, affine (when present)
    """
    attrs = {}
    if final_shape is not None:
        attrs["final_shape"] = repr(tuple(int(x) for x in final_shape))
    if target_spacing is not None:
        attrs["target_spacing"] = repr(tuple(float(x) for x in target_spacing))

    if not pkl_path or not os.path.exists(pkl_path):
        return attrs

    with open(pkl_path, "rb") as f:
        props = pickle.load(f)
    if not isinstance(props, dict):
        return attrs

    orig_shape = props.get("shape_before_cropping", props.get("orig_shape"))
    if orig_shape is not None:
        attrs["orig_shape"] = repr(tuple(int(x) for x in orig_shape))

    bbox = props.get("bbox_used_for_cropping", props.get("bbox_used", props.get("crop_bbox")))
    if bbox is not None:
        bbox_clean = [[int(lo), int(hi)] for lo, hi in bbox]
        attrs["bbox_used"] = repr(bbox_clean)
        attrs["crop_bbox"] = repr(bbox_clean)

    cropped_shape = props.get("shape_after_cropping_and_before_resampling", props.get("cropped_shape"))
    if cropped_shape is not None:
        attrs["cropped_shape"] = repr(tuple(int(x) for x in cropped_shape))
    elif bbox is not None:
        attrs["cropped_shape"] = repr(tuple(int(hi) - int(lo) for lo, hi in bbox))

    orig_spacing = props.get("spacing", props.get("orig_spacing"))
    if orig_spacing is not None:
        attrs["orig_spacing"] = repr(tuple(float(x) for x in orig_spacing))

    # Preserve ITK / NIfTI world-coordinate orientation metadata for physical-space verification/export
    sitk_stuff = props.get("sitk_stuff") or {}
    if isinstance(sitk_stuff, dict):
        if "spacing" in sitk_stuff:
            attrs["itk_spacing"] = repr(_to_serializable(sitk_stuff["spacing"]))
        if "origin" in sitk_stuff:
            attrs["itk_origin"] = repr(_to_serializable(sitk_stuff["origin"]))
        if "direction" in sitk_stuff:
            attrs["itk_direction"] = repr(_to_serializable(sitk_stuff["direction"]))
    if "affine" in props and props["affine"] is not None:
        attrs["affine"] = repr(_to_serializable(props["affine"]))

    return attrs


def convert_dataset(input_dir, output_dir, splits_file=None, fold=0, target_spacing=None):
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    
    # 1. Discover cases from .b2nd files
    all_b2nd = glob.glob(os.path.join(input_dir, "*.b2nd"))
    seg_files = set(glob.glob(os.path.join(input_dir, "*_seg.b2nd")))
    image_files = [f for f in all_b2nd if f not in seg_files]
    
    print(f"[INFO] Found {len(image_files)} image files and {len(seg_files)} segmentation files in {input_dir}")
    if len(image_files) == 0:
        print("[ERROR] No image .b2nd files found! Please verify the input directory path.")
        return

    converted_cases = []
    failed_cases = []

    for img_path in tqdm(image_files, desc="Converting to .h5"):
        case_id = os.path.basename(img_path).replace(".b2nd", "")
        seg_path = os.path.join(input_dir, f"{case_id}_seg.b2nd")
        
        if not os.path.exists(seg_path):
            print(f"[WARN] Segmentation file missing for {case_id}, skipping.")
            failed_cases.append(case_id)
            continue
            
        out_h5_path = os.path.join(output_dir, "data", f"{case_id}.h5")
        
        try:
            img_data = load_b2nd_array(img_path)
            seg_data = load_b2nd_array(seg_path)
            
            if img_data.ndim == 4 and img_data.shape[0] == 1:
                img_data = img_data[0]
            if seg_data.ndim == 4 and seg_data.shape[0] == 1:
                seg_data = seg_data[0]
                
            img_data = img_data.astype(np.float32)
            seg_data = seg_data.astype(np.uint8)
            pkl_path = os.path.join(input_dir, f"{case_id}.pkl")
            h5_attrs = extract_pkl_metadata(
                pkl_path,
                final_shape=img_data.shape,
                target_spacing=target_spacing,
            )

            with h5py.File(out_h5_path, 'w') as h5f:
                h5f.create_dataset('image', data=img_data, compression='gzip')
                h5f.create_dataset('label', data=seg_data, compression='gzip')
                for attr_k, attr_v in h5_attrs.items():
                    h5f.attrs[attr_k] = attr_v
                
            converted_cases.append(case_id)
        except Exception as e:
            print(f"[ERROR] Failed to convert {case_id}: {e}")
            failed_cases.append(case_id)

    print(f"\n[SUMMARY] Successfully converted: {len(converted_cases)} cases.")
    if failed_cases:
        print(f"[SUMMARY] Failed: {len(failed_cases)} cases.")

    # 2. Generate Split text files for all 5 folds
    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    if splits_file and os.path.exists(splits_file):
        print(f"[INFO] Parsing all folds from {splits_file}")
        with open(splits_file, 'r') as sf:
            splits_data = json.load(sf)

        if isinstance(splits_data, list):
            num_folds = len(splits_data)
            print(f"[INFO] Found {num_folds} folds in {splits_file}")
            for f_idx, fold_info in enumerate(splits_data):
                raw_train = set(fold_info.get('train', []))
                raw_val = set(fold_info.get('val', []))

                f_train = [c for c in converted_cases if c in raw_train]
                f_val = [c for c in converted_cases if c in raw_val]

                f_train_path = os.path.join(splits_dir, f"fold{f_idx}_train.txt")
                f_val_path = os.path.join(splits_dir, f"fold{f_idx}_val.txt")

                with open(f_train_path, 'w') as f:
                    for c in sorted(f_train):
                        f.write(f"{c}\n")
                with open(f_val_path, 'w') as f:
                    for c in sorted(f_val):
                        f.write(f"{c}\n")

                print(f"  -> Fold {f_idx}: {len(f_train)} train, {len(f_val)} val cases saved to splits/")

            default_fold = fold if fold < num_folds else 0
            default_train_src = os.path.join(splits_dir, f"fold{default_fold}_train.txt")
            default_val_src = os.path.join(splits_dir, f"fold{default_fold}_val.txt")
            import shutil
            shutil.copy(default_train_src, os.path.join(output_dir, "train.txt"))
            shutil.copy(default_val_src, os.path.join(output_dir, "val.txt"))
            shutil.copy(default_val_src, os.path.join(output_dir, "test.txt"))
    else:
        print("[INFO] Splits file not provided or found. Generating default 80/20 train/val split.")
        converted_cases_sorted = sorted(converted_cases)
        split_idx = int(len(converted_cases_sorted) * 0.8)
        train_cases = converted_cases_sorted[:split_idx]
        val_cases = converted_cases_sorted[split_idx:]

        with open(os.path.join(output_dir, "train.txt"), 'w') as f:
            for c in train_cases:
                f.write(f"{c}\n")
        with open(os.path.join(output_dir, "val.txt"), 'w') as f:
            for c in val_cases:
                f.write(f"{c}\n")
        with open(os.path.join(output_dir, "test.txt"), 'w') as f:
            for c in val_cases:
                f.write(f"{c}\n")

    print(f"[SUCCESS] Data conversion and 5-fold splits completed! All files saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Option 1 nnUNet preprocessed data to HDF5 for PICK")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.environ.get("MEDSSL_OPT1_INPUT_DIR"),
        help="Path to nnUNet Option 1 preprocessed 3d_fullres directory (or set $MEDSSL_OPT1_INPUT_DIR)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("MEDSSL_H5_OUTPUT_DIR"),
        help="Path to directory where .h5 and split files will be created (or set $MEDSSL_H5_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        default=os.environ.get("MEDSSL_OPT1_SPLITS_FILE"),
        help="Path to splits_multilabel.json or splits_final.json (or set $MEDSSL_OPT1_SPLITS_FILE)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index to extract for train/val split (default: 0)",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help="Optional preprocessed target voxel spacing (Z Y X) in mm to persist in HDF5 attrs.",
    )
    args = parser.parse_args()

    if not args.input_dir or not args.output_dir:
        parser.error("--input_dir and --output_dir are required (or set MEDSSL_OPT1_INPUT_DIR / MEDSSL_H5_OUTPUT_DIR).")

    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits_file=args.splits_file,
        fold=args.fold,
        target_spacing=tuple(args.target_spacing) if args.target_spacing else None,
    )


if __name__ == "__main__":
    main()
