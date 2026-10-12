"""
3D Spatial Masking Utilities for Semi-Supervised Learning.
Includes:
- MaskGeneratorBHSD: Token-based masking for Masked Image Modeling (MIM).
- context_mask: Bounding-box context masking for CutMix.
- random_mask: Multi-patch random masking.
"""

from typing import Tuple
import numpy as np
import torch


class MaskGeneratorBHSD:
    """
    Mask Generator for Masked Image Modeling (MIM) on non-cubic 3D patches (e.g. 32 x 160 x 160).
    Divides the volume into 3D tokens and generates random binary masks with a specified mask ratio.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (32, 160, 160),
        mask_patch_size: Tuple[int, int, int] = (8, 16, 16),
        mask_ratio: float = 0.6
    ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio

        assert input_size[0] % mask_patch_size[0] == 0, f"{input_size[0]} not divisible by {mask_patch_size[0]}"
        assert input_size[1] % mask_patch_size[1] == 0, f"{input_size[1]} not divisible by {mask_patch_size[1]}"
        assert input_size[2] % mask_patch_size[2] == 0, f"{input_size[2]} not divisible by {mask_patch_size[2]}"

        self.rand_size = (
            input_size[0] // mask_patch_size[0],
            input_size[1] // mask_patch_size[1],
            input_size[2] // mask_patch_size[2],
        )
        self.token_count = self.rand_size[0] * self.rand_size[1] * self.rand_size[2]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self) -> np.ndarray:
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape(self.rand_size)
        mask = (
            mask.repeat(self.mask_patch_size[0], axis=0)
            .repeat(self.mask_patch_size[1], axis=1)
            .repeat(self.mask_patch_size[2], axis=2)
        )
        return mask


def context_mask(
    img: torch.Tensor,
    mask_ratio: float = 2 / 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate context mask for CutMix.
    
    Args:
        img: 5D Tensor (B, C, D, H, W)
        mask_ratio: fraction of volume dimensions to cut
        
    Returns:
        mask: 3D binary tensor of shape (D, H, W) on GPU/same device
        loss_mask: 4D binary tensor of shape (B, D, H, W) on GPU/same device
    """
    batch_size = img.shape[0]
    img_x, img_y, img_z = img.shape[2], img.shape[3], img.shape[4]
    device = img.device

    loss_mask = torch.ones(batch_size, img_x, img_y, img_z, device=device)
    mask = torch.ones(img_x, img_y, img_z, device=device)

    patch_pixel_x = int(img_x * mask_ratio)
    patch_pixel_y = int(img_y * mask_ratio)
    patch_pixel_z = int(img_z * mask_ratio)

    w = np.random.randint(0, max(img_x - patch_pixel_x, 1))
    h = np.random.randint(0, max(img_y - patch_pixel_y, 1))
    z = np.random.randint(0, max(img_z - patch_pixel_z, 1))

    mask[w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 0
    loss_mask[:, w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 0

    return mask.long(), loss_mask.long()


def random_mask(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate 27-grid multi-patch random mask for 3D volumes."""
    batch_size = img.shape[0]
    img_x, img_y, img_z = img.shape[2], img.shape[3], img.shape[4]
    device = img.device

    loss_mask = torch.ones(batch_size, img_x, img_y, img_z, device=device)
    mask = torch.ones(img_x, img_y, img_z, device=device)

    patch_pixel_x = int(img_x * 2 / 3)
    patch_pixel_y = int(img_y * 2 / 3)
    patch_pixel_z = int(img_z * 2 / 3)

    mask_size_x = int(patch_pixel_x / 3) + 1
    mask_size_y = int(patch_pixel_y / 3) + 1
    mask_size_z = int(patch_pixel_z / 3)

    size_x, size_y, size_z = int(img_x / 3), int(img_y / 3), int(img_z / 3)
    for xs in range(3):
        for ys in range(3):
            for zs in range(3):
                w = np.random.randint(xs * size_x, max((xs + 1) * size_x - mask_size_x, xs * size_x + 1))
                h = np.random.randint(ys * size_y, max((ys + 1) * size_y - mask_size_y, ys * size_y + 1))
                z = np.random.randint(zs * size_z, max((zs + 1) * size_z - mask_size_z, zs * size_z + 1))
                mask[w:w + mask_size_x, h:h + mask_size_y, z:z + mask_size_z] = 0
                loss_mask[:, w:w + mask_size_x, h:h + mask_size_y, z:z + mask_size_z] = 0

    return mask.long(), loss_mask.long()
