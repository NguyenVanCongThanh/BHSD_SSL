"""
Data package for MedSSL-BHSD.
Uses PEP 562 lazy imports so importing submodules (e.g. data.constants, data.samplers)
does not eagerly pull heavy optional dependencies (h5py, scipy, torchvision) or create
circular imports with core.
"""

import importlib

_LAZY_EXPORTS = {
    "BHSDDataset": ("data.dataset_bhsd", "BHSDDataset"),
    "BHSD_MULTICLASS_NAMES": ("core.constants", "BHSD_MULTICLASS_NAMES"),
    "BHSD_BINARY_NAMES": ("core.constants", "BHSD_BINARY_NAMES"),
    "TwoStreamBatchSampler": ("data.samplers", "TwoStreamBatchSampler"),
    "DistributedTwoStreamBatchSampler": ("data.samplers", "DistributedTwoStreamBatchSampler"),
    "RandomRotFlip": ("data.transforms3d", "RandomRotFlip"),
    "RandomCrop": ("data.transforms3d", "RandomCrop"),
    "CenterCrop": ("data.transforms3d", "CenterCrop"),
    "ToTensor": ("data.transforms3d", "ToTensor"),
    "RandomIntensityScale": ("data.transforms3d", "RandomIntensityScale"),
    "GaussianNoise": ("data.transforms3d", "GaussianNoise"),
    "RandomScaling3D": ("data.transforms3d", "RandomScaling3D"),
    "GaussianBlur3D": ("data.transforms3d", "GaussianBlur3D"),
    "GammaTransform": ("data.transforms3d", "GammaTransform"),
    "MaskGeneratorBHSD": ("data.masking", "MaskGeneratorBHSD"),
    "context_mask": ("data.masking", "context_mask"),
    "random_mask": ("data.masking", "random_mask"),
    "build_bhsd_transforms": ("data.dataloader_factory", "build_bhsd_transforms"),
    "build_bhsd_dataloaders": ("data.dataloader_factory", "build_bhsd_dataloaders"),
    "DataLoaderBundle": ("data.dataloader_factory", "DataLoaderBundle"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'data' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
