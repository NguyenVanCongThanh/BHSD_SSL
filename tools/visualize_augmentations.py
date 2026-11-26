#!/usr/bin/env python3
"""
Visualize all 3D data augmentation transforms used in MedSSL-BHSD.

Generates three PNG figures from sample BHSD .h5 volumes:
1. aug_01_geometric.png: Isolated geometric transforms (with GT mask overlay).
2. aug_02_intensity_noise.png: Isolated intensity, noise, blur, and low-res transforms
   rendered with a fixed intensity window from the original patch.
3. aug_03_nnunet_pipeline_samples.png: Composite samples from the full 'nnunet'
   training pipeline (`build_bhsd_transforms`).
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.constants import BHSD_CLASS_ABBREVS, DEFAULT_AUGMENTATION_CONFIG
from data.dataloader_factory import build_bhsd_transforms
from data.transforms3d import (
    ContrastTransform3D,
    GammaTransform,
    GaussianBlur3D,
    GaussianNoise,
    MultiplicativeBrightness3D,
    RandomAxialRotate3D,
    RandomCrop,
    RandomRotFlip,
    RandomScaling3D,
    SimulateLowResolution3D,
)

# Distinct colors for classes 1..5 (EDH, IPH, IVH, SAH, SDH)
CLASS_COLORS = [
    (0.0, 0.0, 0.0, 0.0),      # 0: Background (transparent)
    (0.95, 0.25, 0.25, 0.55),  # 1: EDH - Red
    (0.20, 0.85, 0.35, 0.55),  # 2: IPH - Green
    (0.20, 0.60, 1.00, 0.55),  # 3: IVH - Blue
    (1.00, 0.85, 0.15, 0.55),  # 4: SAH - Yellow
    (0.85, 0.30, 0.95, 0.55),  # 5: SDH - Magenta
]
MASK_CMAP = ListedColormap(CLASS_COLORS)


def load_h5_case(case_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load (image, label) arrays from a BHSD .h5 or .npz file."""
    if case_path.suffix == ".npz":
        with np.load(case_path) as data:
            return data["image"].astype(np.float32), data["label"].astype(np.uint8)
    import h5py
    with h5py.File(case_path, "r") as f:
        image = f["image"][:].astype(np.float32)
        label = f["label"][:].astype(np.uint8)
    return image, label


def pick_best_axial_slice(label: np.ndarray) -> int:
    """Return the axial slice index with maximum foreground area (or center slice)."""
    fg_counts = np.sum(label > 0, axis=(1, 2))
    if int(fg_counts.max()) > 0:
        return int(np.argmax(fg_counts))
    return label.shape[0] // 2


def copy_sample(image: np.ndarray, label: np.ndarray) -> Dict[str, np.ndarray]:
    """Create a fresh sample dictionary copy so transforms do not mutate inputs."""
    return {"image": image.copy(), "label": label.copy()}


def build_legend_handles(present_classes: Sequence[int]) -> List[Patch]:
    """Build legend patches for foreground classes present in the sample."""
    handles = []
    for cls_id in present_classes:
        if 1 <= cls_id <= len(BHSD_CLASS_ABBREVS):
            handles.append(
                Patch(
                    facecolor=CLASS_COLORS[cls_id][:3],
                    edgecolor="white",
                    label=f"C{cls_id}: {BHSD_CLASS_ABBREVS[cls_id - 1]}",
                )
            )
    return handles


def render_slice_panel(
    ax: plt.Axes,
    img_2d: np.ndarray,
    lbl_2d: Optional[np.ndarray],
    title: str,
    vmin: float,
    vmax: float,
    subtitle: str = "",
) -> None:
    """Render a 2D axial CT slice with optional segmentation mask overlay."""
    ax.imshow(img_2d, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    if lbl_2d is not None and np.any(lbl_2d > 0):
        ax.imshow(lbl_2d, cmap=MASK_CMAP, vmin=0, vmax=5, interpolation="nearest")
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=9.5, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_geometric_grid(
    image: np.ndarray,
    label: np.ndarray,
    case_id: str,
    patch_size: Tuple[int, int, int],
    out_path: Path,
    seed: int,
) -> None:
    """Visualize isolated geometric augmentations with GT mask overlay."""
    np.random.seed(seed)
    transforms = [
        ("Original", None),
        ("RandomRotFlip\n(rot90 + 3D flip)", RandomRotFlip(mirror_z=True)),
        ("RandomAxialRotate3D\n(angle ~ [-180, 180], p=1)", RandomAxialRotate3D(deg_range=(35.0, 45.0), prob=1.0)),
        ("RandomScaling3D\n(scale=0.85, zoom out)", RandomScaling3D(scale_range=(0.85, 0.85), prob=1.0, pad_to_min=True)),
        ("RandomScaling3D\n(scale=1.15, zoom in)", RandomScaling3D(scale_range=(1.15, 1.15), prob=1.0, pad_to_min=True)),
        (f"RandomCrop FG\n(fg_prob=1.0, {patch_size})", RandomCrop(patch_size, foreground_prob=1.0, pad_to_min=True)),
        (f"RandomCrop BG/Rand\n(fg_prob=0.0, {patch_size})", RandomCrop(patch_size, foreground_prob=0.0, pad_to_min=True)),
    ]

    vmin, vmax = np.percentile(image, [1.0, 99.0])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), facecolor="#111111")
    axes_flat = axes.flatten()

    for idx, (name, tfm) in enumerate(transforms):
        ax = axes_flat[idx]
        ax.set_facecolor("#111111")
        sample = copy_sample(image, label)
        if tfm is not None:
            sample = tfm(sample)
        z_idx = pick_best_axial_slice(sample["label"])
        fg_voxels = int(np.sum(sample["label"][z_idx] > 0))
        render_slice_panel(
            ax,
            sample["image"][z_idx],
            sample["label"][z_idx],
            title=name,
            subtitle=f"shape={sample['image'].shape} | z={z_idx} | fg_px={fg_voxels}",
            vmin=vmin,
            vmax=vmax,
        )
        ax.title.set_color("white")

    # Last cell for legend & metadata
    legend_ax = axes_flat[-1]
    legend_ax.set_facecolor("#1a1a1a")
    legend_ax.axis("off")
    present = sorted(int(c) for c in np.unique(label) if c > 0)
    handles = build_legend_handles(present)
    if handles:
        legend_ax.legend(
            handles=handles,
            loc="center",
            frameon=True,
            facecolor="#262626",
            edgecolor="#555555",
            labelcolor="white",
            fontsize=10,
            title="Hemorrhage Classes",
            title_fontsize=10.5,
        )

    fig.suptitle(
        f"BHSD Fold 0 Optimal — Geometric Augmentations (Isolated, p=1.0) | Case: {case_id}",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_intensity_grid(
    image: np.ndarray,
    label: np.ndarray,
    case_id: str,
    patch_size: Tuple[int, int, int],
    out_path: Path,
    seed: int,
) -> None:
    """Visualize isolated intensity, noise, blur, and low-res transforms."""
    np.random.seed(seed)
    base_crop = RandomCrop(patch_size, foreground_prob=1.0, pad_to_min=True)(copy_sample(image, label))
    base_img, base_lbl = base_crop["image"], base_crop["label"]
    z_idx = pick_best_axial_slice(base_lbl)
    vmin, vmax = np.percentile(base_img, [1.0, 99.0])

    transforms = [
        ("Original Patch", None),
        ("GaussianNoise\n(std=0.10)", GaussianNoise(std_range=(0.10, 0.10), prob=1.0)),
        ("GaussianBlur3D\n(sigma=1.0, in-plane)", GaussianBlur3D(sigma_range=(1.0, 1.0), prob=1.0)),
        ("SimulateLowResolution3D\n(zoom=0.5)", SimulateLowResolution3D(zoom_range=(0.5, 0.5), prob=1.0)),
        ("MultiplicativeBrightness3D\n(factor=0.75, darker)", MultiplicativeBrightness3D(multiplier_range=(0.75, 0.75), prob=1.0)),
        ("MultiplicativeBrightness3D\n(factor=1.25, brighter)", MultiplicativeBrightness3D(multiplier_range=(1.25, 1.25), prob=1.0)),
        ("ContrastTransform3D\n(factor=0.75, lower)", ContrastTransform3D(contrast_range=(0.75, 0.75), prob=1.0)),
        ("ContrastTransform3D\n(factor=1.25, higher)", ContrastTransform3D(contrast_range=(1.25, 1.25), prob=1.0)),
        ("GammaTransform\n(gamma=0.7, normal)", GammaTransform(gamma_range=(0.7, 0.7), prob=1.0, retain_stats=True, invert_image=False)),
        ("GammaTransform\n(gamma=1.5, inverted)", GammaTransform(gamma_range=(1.5, 1.5), prob=1.0, retain_stats=True, invert_image=True)),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(19, 8.2), facecolor="#111111")
    for idx, (name, tfm) in enumerate(transforms):
        ax = axes.flatten()[idx]
        ax.set_facecolor("#111111")
        sample = copy_sample(base_img, base_lbl)
        if tfm is not None:
            sample = tfm(sample)
        slc = sample["image"][z_idx]
        render_slice_panel(
            ax,
            slc,
            None,
            title=name,
            subtitle=f"mean={slc.mean():.2f}, std={slc.std():.2f}",
            vmin=vmin,
            vmax=vmax,
        )
        ax.title.set_color("white")

    fig.suptitle(
        f"BHSD Fold 0 Optimal — Intensity, Noise & Low-Res Augmentations (Fixed Window, p=1.0) | Case: {case_id}",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_nnunet_pipeline_samples(
    image: np.ndarray,
    label: np.ndarray,
    case_id: str,
    patch_size: Tuple[int, int, int],
    out_path: Path,
    base_seed: int,
) -> None:
    """Visualize composite samples produced by the full 'nnunet' training pipeline."""
    pipeline = build_bhsd_transforms(
        patch_size=patch_size,
        extended_aug=True,
        foreground_prob=0.50,
        aug_config=dict(DEFAULT_AUGMENTATION_CONFIG),
        aug_preset="nnunet",
    )
    vmin, vmax = np.percentile(image, [1.0, 99.0])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), facecolor="#111111")
    axes_flat = axes.flatten()

    # Panel 0: Original volume at GT-max slice
    z0 = pick_best_axial_slice(label)
    axes_flat[0].set_facecolor("#111111")
    render_slice_panel(
        axes_flat[0],
        image[z0],
        label[z0],
        title="Original Volume",
        subtitle=f"shape={image.shape} | z={z0}",
        vmin=vmin,
        vmax=vmax,
    )
    axes_flat[0].title.set_color("white")

    # Panels 1..7: Random samples from full nnunet pipeline
    for i in range(1, 8):
        seed_i = base_seed + i * 17
        np.random.seed(seed_i)
        out = pipeline(copy_sample(image, label))
        aug_img = out["image"].squeeze(0).numpy()
        aug_lbl = out["label"].numpy()
        zi = pick_best_axial_slice(aug_lbl)
        ax = axes_flat[i]
        ax.set_facecolor("#111111")
        render_slice_panel(
            ax,
            aug_img[zi],
            aug_lbl[zi],
            title=f"nnU-Net Sample #{i} (seed={seed_i})",
            subtitle=f"z={zi} | mean={aug_img[zi].mean():.2f}, std={aug_img[zi].std():.2f}",
            vmin=vmin,
            vmax=vmax,
        )
        ax.title.set_color("white")

    fig.suptitle(
        f"BHSD Fold 0 Optimal — Full 'nnunet' Augmentation Pipeline Samples (patch={patch_size}) | Case: {case_id}",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MedSSL-BHSD 3D augmentations from .h5 volumes.")
    parser.add_argument("--input", type=Path, required=True, help="Path to an .h5 file or directory of .h5 files.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "docs" / "analysis" / "figures" / "augmentations",
        help="Directory to save output PNG figures.",
    )
    parser.add_argument("--patch_size", type=int, nargs=3, default=[32, 256, 256], help="Target 3D patch size (Z Y X).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible visualization.")
    args = parser.parse_args()

    if args.input.is_file():
        h5_files = [args.input]
    else:
        h5_files = sorted(args.input.glob("*.h5")) or sorted(args.input.glob("*.npz"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_size = tuple(int(x) for x in args.patch_size)

    for h5_path in h5_files:
        case_id = h5_path.stem
        image, label = load_h5_case(h5_path)
        prefix = f"{case_id}_" if len(h5_files) > 1 else ""
        plot_geometric_grid(image, label, case_id, patch_size, args.output_dir / f"{prefix}aug_01_geometric.png", args.seed)
        plot_intensity_grid(image, label, case_id, patch_size, args.output_dir / f"{prefix}aug_02_intensity_noise.png", args.seed)
        plot_nnunet_pipeline_samples(
            image, label, case_id, patch_size, args.output_dir / f"{prefix}aug_03_nnunet_pipeline_samples.png", args.seed
        )
        print(f"Saved augmentation figures for {case_id} -> {args.output_dir}")


if __name__ == "__main__":
    main()
