"""
3D Volumetric Data Augmentation Transforms for CT Scans.
"""

from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F


class RandomRotFlip(object):
    """Randomly rotate (90 deg) and flip the 3D volume along spatial dimensions."""
    def __call__(self, sample: Dict) -> Dict:
        image, label = sample['image'], sample['label']
        # Random rotation in axial plane (Y, X)
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2))
        label = np.rot90(label, k, axes=(1, 2))

        # Random flip along Y or X axis
        axis = np.random.randint(1, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        sample['image'] = image
        sample['label'] = label
        return sample


class RandomCrop(object):
    """
    Random crop of a 3D volume to target output_size (Z, Y, X).
    Supports nnUNet-style foreground oversampling: with probability `foreground_prob`,
    the crop is centered on a random foreground lesion voxel.
    Applies symmetric zero-padding if any dimension of the volume is smaller than the target.
    """
    def __init__(self, output_size: Tuple[int, int, int], foreground_prob: float = 0.33):
        self.output_size = output_size
        self.foreground_prob = foreground_prob

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

        # nnUNet-style Foreground Oversampling
        force_fg = (np.random.random() < self.foreground_prob)
        fg_coords = None
        if force_fg and np.any(label > 0):
            fg_coords = np.argwhere(label > 0)

        if fg_coords is not None and len(fg_coords) > 0:
            target_idx = np.random.randint(0, len(fg_coords))
            cz, cy, cx = fg_coords[target_idx]
            # Center the crop around the selected foreground voxel with random jitter
            z1 = int(cz - self.output_size[0] // 2 + np.random.randint(-4, 5))
            y1 = int(cy - self.output_size[1] // 2 + np.random.randint(-16, 17))
            x1 = int(cx - self.output_size[2] // 2 + np.random.randint(-16, 17))
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
    def __init__(self, scale_range: Tuple[float, float] = (0.85, 1.15), prob: float = 0.2):
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            image, label = sample['image'], sample['label']
            orig_shape = image.shape  # (Z, Y, X)

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
                scaled_img = F.pad(scaled_img, (0, 0, 0, 0, pad_z1, pad_z2), mode='constant', value=0)
                scaled_lbl = F.pad(scaled_lbl, (0, 0, 0, 0, pad_z1, pad_z2), mode='constant', value=0)

            # Handle Y
            if sy > oy:
                start_y = (sy - oy) // 2
                scaled_img = scaled_img[:, :, :, start_y:start_y + oy, :]
                scaled_lbl = scaled_lbl[:, :, :, start_y:start_y + oy, :]
            elif sy < oy:
                pad_y1 = (oy - sy) // 2
                pad_y2 = oy - sy - pad_y1
                scaled_img = F.pad(scaled_img, (0, 0, pad_y1, pad_y2, 0, 0), mode='constant', value=0)
                scaled_lbl = F.pad(scaled_lbl, (0, 0, pad_y1, pad_y2, 0, 0), mode='constant', value=0)

            # Handle X
            if sx > ox:
                start_x = (sx - ox) // 2
                scaled_img = scaled_img[:, :, :, :, start_x:start_x + ox]
                scaled_lbl = scaled_lbl[:, :, :, :, start_x:start_x + ox]
            elif sx < ox:
                pad_x1 = (ox - sx) // 2
                pad_x2 = ox - sx - pad_x1
                scaled_img = F.pad(scaled_img, (pad_x1, pad_x2, 0, 0, 0, 0), mode='constant', value=0)
                scaled_lbl = F.pad(scaled_lbl, (pad_x1, pad_x2, 0, 0, 0, 0), mode='constant', value=0)

            sample['image'] = scaled_img.squeeze(0).squeeze(0).numpy()
            sample['label'] = scaled_lbl.squeeze(0).squeeze(0).numpy().astype(label.dtype)

        return sample


class GaussianBlur3D(object):
    """
    Subtle Gaussian blurring on 3D CT volume to simulate varying scanner resolution/noise.
    Following nnUNet data augmentation specification.
    """
    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 1.0), prob: float = 0.2):
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
    Enriches contrast around diverse lesion densities (e.g. hyperdense EDH vs isodense subacute SDH).
    Normalized to preserve foreground range.
    """
    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), prob: float = 0.2, retain_stats: bool = True):
        self.gamma_range = gamma_range
        self.prob = prob
        self.retain_stats = retain_stats

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            img = sample['image']
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            mn, mx = img.min(), img.max()
            rng = mx - mn
            if rng > 1e-4:
                # Scale to [0, 1]
                norm_img = (img - mn) / rng
                # Apply gamma
                gamma_img = np.power(np.clip(norm_img, 0, 1), gamma)
                # Rescale back to original intensity range
                sample['image'] = (gamma_img * rng) + mn
        return sample


class GaussianNoise(object):
    """Add subtle Gaussian noise to image."""
    def __init__(self, mean: float = 0.0, std: float = 0.02, prob: float = 0.3):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() < self.prob:
            noise = np.random.normal(self.mean, self.std, size=sample['image'].shape)
            sample['image'] = sample['image'] + noise
        return sample


class ToTensor(object):
    """Convert numpy ndarrays to PyTorch Tensors and add channel dimension (1, Z, Y, X)."""
    def __call__(self, sample: Dict) -> Dict:
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample['image'] = torch.from_numpy(image)
        sample['label'] = torch.from_numpy(sample['label'].astype(np.int64))
        return sample
