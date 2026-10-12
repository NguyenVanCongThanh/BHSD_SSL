"""
3D Volumetric Data Augmentation Transforms for CT Scans.
"""

from typing import Tuple, Dict
import numpy as np
import torch


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
    Applies symmetric zero-padding if any dimension of the volume is smaller than the target.
    """
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
