"""
3D Volumetric Data Augmentation Transforms for CT Scans.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F

from data.constants import (
    DEFAULT_AUGMENTATION_CONFIG,
    DEFAULT_CROP_JITTER_XY,
    DEFAULT_CROP_JITTER_Z,
    DEFAULT_FOREGROUND_OVERSAMPLE_PROB,
)


class RandomRotFlip(object):
    """Randomly rotate (90 deg) and flip the 3D volume along spatial dimensions."""
    def __init__(self, mirror_z: bool = False):
        self.mirror_z = bool(mirror_z)

    def __call__(self, sample: Dict) -> Dict:
        image, label = sample['image'], sample['label']
        # Random rotation in axial plane (Y, X)
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2))
        label = np.rot90(label, k, axes=(1, 2))

        if self.mirror_z:
            # Independent mirroring across all 3 axes (nnU-Net 3D MirrorTransform)
            for axis in (0, 1, 2):
                if np.random.rand() < 0.5:
                    image = np.flip(image, axis=axis)
                    label = np.flip(label, axis=axis)
            image = image.copy()
            label = label.copy()
        else:
            # Legacy random flip along Y or X axis
            axis = np.random.randint(1, 3)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

        sample['image'] = image
        sample['label'] = label
        return sample


class RandomAxialRotate3D(object):
    """
    Continuous in-plane (Y, X) rotation matching nnU-Net's dummy-2D rotation on anisotropic CT.
    Rotates image slices with bilinear interpolation (fill=min) and labels with nearest-neighbor.
    """
    def __init__(
        self,
        deg_range: Tuple[float, float] = (-180.0, 180.0),
        prob: float = DEFAULT_AUGMENTATION_CONFIG["rotation_prob"],
        pad_to_min: bool = True,
    ):
        self.deg_range = deg_range
        self.prob = prob
        self.pad_to_min = pad_to_min

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            from scipy.ndimage import rotate
            angle = float(np.random.uniform(self.deg_range[0], self.deg_range[1]))
            image, label = sample['image'], sample['label']
            cval = float(image.min()) if self.pad_to_min else 0.0
            rot_img = rotate(
                image,
                angle=angle,
                axes=(1, 2),
                reshape=False,
                order=1,
                mode='constant',
                cval=cval,
            )
            rot_lbl = rotate(
                label,
                angle=angle,
                axes=(1, 2),
                reshape=False,
                order=0,
                mode='constant',
                cval=0,
            )
            sample['image'] = rot_img.astype(image.dtype, copy=False)
            sample['label'] = rot_lbl.astype(label.dtype, copy=False)
        return sample


class RandomCrop(object):
    """
    Random crop of a 3D volume to target output_size (Z, Y, X).
    Supports nnUNet-style foreground oversampling: with probability `foreground_prob`,
    the crop is centered on a random foreground lesion voxel.
    Applies symmetric zero-padding if any dimension of the volume is smaller than the target.
    """
    def __init__(
        self,
        output_size: Tuple[int, int, int],
        foreground_prob: float = DEFAULT_FOREGROUND_OVERSAMPLE_PROB,
        jitter_z: int = DEFAULT_CROP_JITTER_Z,
        jitter_xy: int = DEFAULT_CROP_JITTER_XY,
        pad_to_min: bool = False,
    ):
        self.output_size = output_size
        self.foreground_prob = foreground_prob
        self.jitter_z = int(jitter_z)
        self.jitter_xy = int(jitter_xy)
        self.pad_to_min = bool(pad_to_min)

    def __call__(self, sample: Dict) -> Dict:
        image, label = sample['image'], sample['label']
        z, y, x = image.shape

        pz = max(self.output_size[0] - z, 0)
        py = max(self.output_size[1] - y, 0)
        px = max(self.output_size[2] - x, 0)

        if pz > 0 or py > 0 or px > 0:
            pad_z = (pz // 2, pz - pz // 2)
            pad_y = (py // 2, py - py // 2)
            pad_x = (px // 2, px - px // 2)
            pad_val = float(image.min()) if self.pad_to_min else 0
            image = np.pad(image, [pad_z, pad_y, pad_x], mode='constant', constant_values=pad_val)
            label = np.pad(label, [pad_z, pad_y, pad_x], mode='constant', constant_values=0)
            z, y, x = image.shape

        # nnUNet-style Foreground Oversampling
        force_fg = (np.random.random() < self.foreground_prob)
        fg_coords = None
        if force_fg and np.any(label > 0):
            fg_coords = np.argwhere(label > 0)

        if fg_coords is not None and len(fg_coords) > 0:
            target_idx = np.random.randint(0, len(fg_coords))
            cz, cy, cx = fg_coords[target_idx]
            # Center the crop around the selected foreground voxel with random jitter
            jz = np.random.randint(-self.jitter_z, self.jitter_z + 1) if self.jitter_z > 0 else 0
            jy = np.random.randint(-self.jitter_xy, self.jitter_xy + 1) if self.jitter_xy > 0 else 0
            jx = np.random.randint(-self.jitter_xy, self.jitter_xy + 1) if self.jitter_xy > 0 else 0
            z1 = int(cz - self.output_size[0] // 2 + jz)
            y1 = int(cy - self.output_size[1] // 2 + jy)
            x1 = int(cx - self.output_size[2] // 2 + jx)
            z1 = max(0, min(z1, z - self.output_size[0]))
            y1 = max(0, min(y1, y - self.output_size[1]))
            x1 = max(0, min(x1, x - self.output_size[2]))
        else:
            z1 = np.random.randint(0, max(z - self.output_size[0] + 1, 1))
            y1 = np.random.randint(0, max(y - self.output_size[1] + 1, 1))
            x1 = np.random.randint(0, max(x - self.output_size[2] + 1, 1))

        sample['image'] = image[
            z1:z1 + self.output_size[0],
            y1:y1 + self.output_size[1],
            x1:x1 + self.output_size[2]
        ]
        sample['label'] = label[
            z1:z1 + self.output_size[0],
            y1:y1 + self.output_size[1],
            x1:x1 + self.output_size[2]
        ]
        return sample


class CenterCrop(object):
    """Center crop of a 3D volume to target output_size (Z, Y, X)."""
    def __init__(self, output_size: Tuple[int, int, int]):
        self.output_size = output_size

    def __call__(self, sample: Dict) -> Dict:
        image, label = sample['image'], sample['label']
        z, y, x = image.shape

        pz = max(self.output_size[0] - z, 0)
        py = max(self.output_size[1] - y, 0)
        px = max(self.output_size[2] - x, 0)

        if pz > 0 or py > 0 or px > 0:
            pad_z = (pz // 2, pz - pz // 2)
            pad_y = (py // 2, py - py // 2)
            pad_x = (px // 2, px - px // 2)
            image = np.pad(image, [pad_z, pad_y, pad_x], mode='constant', constant_values=0)
            label = np.pad(label, [pad_z, pad_y, pad_x], mode='constant', constant_values=0)
            z, y, x = image.shape

        z1 = (z - self.output_size[0]) // 2
        y1 = (y - self.output_size[1]) // 2
        x1 = (x - self.output_size[2]) // 2

        sample['image'] = image[
            z1:z1 + self.output_size[0],
            y1:y1 + self.output_size[1],
            x1:x1 + self.output_size[2]
        ]
        sample['label'] = label[
            z1:z1 + self.output_size[0],
            y1:y1 + self.output_size[1],
            x1:x1 + self.output_size[2]
        ]
        return sample


class RandomIntensityScale(object):
    """Randomly scale image intensity by a factor in [1 - scale_range, 1 + scale_range]."""
    def __init__(self, scale_range: float = 0.1, prob: float = 0.5):
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            factor = np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
            sample['image'] = sample['image'] * factor
        return sample


class RandomScaling3D(object):
    """
    Randomly scale 3D volume by a factor within [min_scale, max_scale] (following nnUNet design).
    Uses trilinear interpolation for image and nearest neighbor for segmentation label.
    """
    def __init__(
        self,
        scale_range: Tuple[float, float] = DEFAULT_AUGMENTATION_CONFIG["scale_range"],
        prob: float = DEFAULT_AUGMENTATION_CONFIG["scale_prob"],
        pad_to_min: bool = False,
    ):
        self.scale_range = scale_range
        self.prob = prob
        self.pad_to_min = bool(pad_to_min)

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            image, label = sample['image'], sample['label']
            orig_shape = image.shape  # (Z, Y, X)
            pad_val = float(image.min()) if self.pad_to_min else 0.0

            # Convert to torch tensor for fast 3D GPU/CPU trilinear/nearest interpolation
            img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            lbl_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()

            scaled_img = F.interpolate(img_t, scale_factor=scale, mode='trilinear', align_corners=False)
            scaled_lbl = F.interpolate(lbl_t, scale_factor=scale, mode='nearest')

            # Crop or symmetric pad back to original volume shape
            sz, sy, sx = scaled_img.shape[2:]
            oz, oy, ox = orig_shape

            # Handle Z
            if sz > oz:
                start_z = (sz - oz) // 2
                scaled_img = scaled_img[:, :, start_z:start_z + oz, :, :]
                scaled_lbl = scaled_lbl[:, :, start_z:start_z + oz, :, :]
            elif sz < oz:
                pad_z1 = (oz - sz) // 2
                pad_z2 = oz - sz - pad_z1
                scaled_img = F.pad(scaled_img, (0, 0, 0, 0, pad_z1, pad_z2), mode='constant', value=pad_val)
                scaled_lbl = F.pad(scaled_lbl, (0, 0, 0, 0, pad_z1, pad_z2), mode='constant', value=0)

            # Handle Y
            if sy > oy:
                start_y = (sy - oy) // 2
                scaled_img = scaled_img[:, :, :, start_y:start_y + oy, :]
                scaled_lbl = scaled_lbl[:, :, :, start_y:start_y + oy, :]
            elif sy < oy:
                pad_y1 = (oy - sy) // 2
                pad_y2 = oy - sy - pad_y1
                scaled_img = F.pad(scaled_img, (0, 0, pad_y1, pad_y2, 0, 0), mode='constant', value=pad_val)
                scaled_lbl = F.pad(scaled_lbl, (0, 0, pad_y1, pad_y2, 0, 0), mode='constant', value=0)

            # Handle X
            if sx > ox:
                start_x = (sx - ox) // 2
                scaled_img = scaled_img[:, :, :, :, start_x:start_x + ox]
                scaled_lbl = scaled_lbl[:, :, :, :, start_x:start_x + ox]
            elif sx < ox:
                pad_x1 = (ox - sx) // 2
                pad_x2 = ox - sx - pad_x1
                scaled_img = F.pad(scaled_img, (pad_x1, pad_x2, 0, 0, 0, 0), mode='constant', value=pad_val)
                scaled_lbl = F.pad(scaled_lbl, (pad_x1, pad_x2, 0, 0, 0, 0), mode='constant', value=0)

            sample['image'] = scaled_img.squeeze(0).squeeze(0).numpy()
            sample['label'] = scaled_lbl.squeeze(0).squeeze(0).numpy().astype(label.dtype)

        return sample


class GaussianBlur3D(object):
    """
    Subtle Gaussian blurring on 3D CT volume to simulate varying scanner resolution/noise.
    Following nnUNet data augmentation specification.
    """
    def __init__(
        self,
        sigma_range: Tuple[float, float] = DEFAULT_AUGMENTATION_CONFIG["blur_sigma_range"],
        prob: float = 0.2,
    ):
        self.sigma_range = sigma_range
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            from scipy.ndimage import gaussian_filter
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            # Only blur in-plane (Y, X) or subtle isotropic
            sample['image'] = gaussian_filter(sample['image'], sigma=(0, sigma, sigma))
        return sample


class GammaTransform(object):
    """
    Non-linear intensity transformation via exponential gamma (I_new = I^gamma).
    Supports nnU-Net inverted gamma and mean/std retention (retain_stats).
    """
    def __init__(
        self,
        gamma_range: Tuple[float, float] = DEFAULT_AUGMENTATION_CONFIG["gamma_range"],
        prob: float = DEFAULT_AUGMENTATION_CONFIG["gamma_prob"],
        retain_stats: bool = True,
        invert_image: bool = False,
    ):
        self.gamma_range = gamma_range
        self.prob = prob
        self.retain_stats = retain_stats
        self.invert_image = invert_image

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            img = sample['image'].astype(np.float32, copy=False)
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            mn, mx = float(img.min()), float(img.max())
            rng = mx - mn
            if rng > 1e-4:
                orig_mean = float(img.mean())
                orig_std = float(img.std())
                norm_img = np.clip((img - mn) / rng, 0.0, 1.0)
                if self.invert_image:
                    gamma_img = 1.0 - np.power(1.0 - norm_img, gamma)
                else:
                    gamma_img = np.power(norm_img, gamma)
                out = (gamma_img * rng) + mn
                if self.retain_stats:
                    new_std = float(out.std())
                    if new_std > 1e-6:
                        out = (out - float(out.mean())) / new_std * orig_std + orig_mean
                sample['image'] = out.astype(sample['image'].dtype, copy=False)
        return sample


class GaussianNoise(object):
    """Add zero-mean Gaussian noise to image (fixed std or sampled from std_range)."""
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.02,
        prob: float = 0.3,
        std_range: Optional[Tuple[float, float]] = None,
    ):
        self.mean = mean
        self.std = std
        self.prob = prob
        self.std_range = std_range

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            sigma = (
                float(np.random.uniform(self.std_range[0], self.std_range[1]))
                if self.std_range is not None
                else float(self.std)
            )
            if sigma > 0.0:
                noise = np.random.normal(self.mean, sigma, size=sample['image'].shape)
                sample['image'] = (sample['image'] + noise).astype(sample['image'].dtype, copy=False)
        return sample


class MultiplicativeBrightness3D(object):
    """Scale intensities multiplicatively matching nnU-Net BrightnessMultiplicativeTransform."""
    def __init__(
        self,
        multiplier_range: Tuple[float, float] = DEFAULT_AUGMENTATION_CONFIG["brightness_range"],
        prob: float = DEFAULT_AUGMENTATION_CONFIG["brightness_prob"],
    ):
        self.multiplier_range = multiplier_range
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            factor = float(np.random.uniform(self.multiplier_range[0], self.multiplier_range[1]))
            sample['image'] = (sample['image'] * factor).astype(sample['image'].dtype, copy=False)
        return sample


class ContrastTransform3D(object):
    """Adjust image contrast around volume mean while optionally preserving [min, max] range."""
    def __init__(
        self,
        contrast_range: Tuple[float, float] = DEFAULT_AUGMENTATION_CONFIG["contrast_range"],
        prob: float = DEFAULT_AUGMENTATION_CONFIG["contrast_prob"],
        preserve_range: bool = True,
    ):
        self.contrast_range = contrast_range
        self.prob = prob
        self.preserve_range = preserve_range

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            img = sample['image']
            factor = float(np.random.uniform(self.contrast_range[0], self.contrast_range[1]))
            mn = float(img.mean())
            out = (img - mn) * factor + mn
            if self.preserve_range:
                out = np.clip(out, float(img.min()), float(img.max()))
            sample['image'] = out.astype(img.dtype, copy=False)
        return sample


class SimulateLowResolution3D(object):
    """Simulate lower in-plane (Y, X) resolution matching nnU-Net dummy-2D resampling."""
    def __init__(
        self,
        zoom_range: Tuple[float, float] = DEFAULT_AUGMENTATION_CONFIG["lowres_scale_range"],
        prob: float = DEFAULT_AUGMENTATION_CONFIG["lowres_prob"],
    ):
        self.zoom_range = zoom_range
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            img = sample['image']
            z, y, x = img.shape
            zoom = float(np.random.uniform(self.zoom_range[0], self.zoom_range[1]))
            target_y = max(1, int(round(y * zoom)))
            target_x = max(1, int(round(x * zoom)))
            if target_y < y or target_x < x:
                img_t = torch.from_numpy(img).unsqueeze(1).float()  # (Z, 1, Y, X)
                down = F.interpolate(img_t, size=(target_y, target_x), mode='nearest')
                up = F.interpolate(down, size=(y, x), mode='bilinear', align_corners=False)
                sample['image'] = up.squeeze(1).numpy().astype(img.dtype, copy=False)
        return sample


class ToTensor(object):
    """Convert numpy ndarrays to PyTorch Tensors and add channel dimension (1, Z, Y, X)."""
    def __call__(self, sample: Dict) -> Dict:
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample['image'] = torch.from_numpy(image)
        sample['label'] = torch.from_numpy(sample['label'].astype(np.int64))
        return sample
