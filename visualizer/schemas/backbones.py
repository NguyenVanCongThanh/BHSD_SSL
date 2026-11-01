"""
Declarative 3D Backbone Specifications from models/backbones/*.py.
Used by SymbolicShapePropagator to compute exact 5D tensor dimensions (B, C, Z, Y, X)
at every encoder stage, skip connection, bottleneck, decoder stage, and deep supervision head.
"""

from typing import Any, Dict, List


def get_backbone_specs() -> Dict[str, Dict[str, Any]]:
    return {
        "nnunet3d": {
            "id": "nnunet3d",
            "name": "nnU-Net 3D (models/backbones/nnunet3d.py)",
            "file": "models/backbones/nnunet3d.py",
            "norm": "InstanceNorm3d(affine=True)",
            "activation": "LeakyReLU(negative_slope=0.01)",
            "downsample_op": "Strided Conv3d (use_strided_conv=True) or MaxPool3d",
            "upsample_op": "ConvTranspose3d",
            "supports_num_stages": [5, 6, 7],
            "supports_anisotropic": True,
            "supports_deep_supervision": True,
            "description": (
                "Kiến trúc 3D U-Net chuẩn nnU-Net v2 (PlainConvUNet Option1_CTClip & spacing05). "
                "Khi num_stages=7 và anisotropic_kernels=True, trục Z chỉ bị giảm 4x tổng cộng "
                "(ở pool3 và pool4), trong khi mặt phẳng XY giảm 64x, bảo toàn các lát cắt xuất huyết mỏng (SAH, SDH, IVH)."
            ),
            "stage_rules": {
                "base_filters": 32,
                "max_filters": 320,
                "channels_by_stages": {
                    "5": [32, 64, 128, 256, 320],
                    "6": [32, 64, 128, 256, 320, 320],
                    "7": [32, 64, 128, 256, 320, 320, 320],
                },
                "kernels_anisotropic": [
                    [1, 3, 3],
                    [1, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                ],
                "kernels_isotropic": [
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                ],
                "pool_strides_anisotropic": [
                    [1, 2, 2],  # pool1
                    [1, 2, 2],  # pool2
                    [2, 2, 2],  # pool3 (Z halved)
                    [2, 2, 2],  # pool4 (Z halved)
                    [1, 2, 2],  # pool5
                    [1, 2, 2],  # pool6
                ],
                "pool_strides_isotropic": [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [1, 2, 2],
                    [1, 2, 2],
                ],
            },
        },
        "vnet": {
            "id": "vnet",
            "name": "V-Net 3D Residual (models/backbones/vnet.py)",
            "file": "models/backbones/vnet.py",
            "norm": "BatchNorm3d (running_mean & running_var buffers)",
            "activation": "ReLU(inplace=True)",
            "downsample_op": "Conv3d(stride=2, kernel=2)",
            "upsample_op": "ConvTranspose3d(stride=2, kernel=2)",
            "supports_num_stages": [5],
            "supports_anisotropic": False,
            "supports_deep_supervision": False,
            "description": (
                "Mạng V-Net 5 tầng với residual blocks (1-2-3-3-3 convs), BatchNorm3d và Dropout3d ở bottleneck/decoder. "
                "Lưu ý: BatchNorm3d có buffers (running_mean, running_var) được copy trực tiếp từ Student sang Teacher trong update_ema_variables."
            ),
            "stage_rules": {
                "base_filters": 16,
                "max_filters": 256,
                "channels_by_stages": {
                    "5": [16, 32, 64, 128, 256],
                },
                "kernels_anisotropic": [[3, 3, 3]] * 5,
                "kernels_isotropic": [[3, 3, 3]] * 5,
                "pool_strides_anisotropic": [[2, 2, 2]] * 4,
                "pool_strides_isotropic": [[2, 2, 2]] * 4,
            },
        },
        "unet3d": {
            "id": "unet3d",
            "name": "Standard 3D U-Net (models/backbones/unet3d.py)",
            "file": "models/backbones/unet3d.py",
            "norm": "BatchNorm3d",
            "activation": "ReLU(inplace=True)",
            "downsample_op": "MaxPool3d(kernel=2, stride=2)",
            "upsample_op": "ConvTranspose3d(kernel=2, stride=2)",
            "supports_num_stages": [5],
            "supports_anisotropic": False,
            "supports_deep_supervision": False,
            "description": "3D U-Net cổ điển 5 tầng [32, 64, 128, 256, 512] với downsampling đẳng hướng (2, 2, 2) ở cả 4 bước.",
            "stage_rules": {
                "base_filters": 32,
                "max_filters": 512,
                "channels_by_stages": {
                    "5": [32, 64, 128, 256, 512],
                },
                "kernels_anisotropic": [[3, 3, 3]] * 5,
                "kernels_isotropic": [[3, 3, 3]] * 5,
                "pool_strides_anisotropic": [[2, 2, 2]] * 4,
                "pool_strides_isotropic": [[2, 2, 2]] * 4,
            },
        },
    }

