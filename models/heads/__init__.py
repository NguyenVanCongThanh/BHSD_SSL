"""
Specialized Prediction Heads for Multi-Task and Semi-Supervised Learning.
"""

from models.heads.seg_head import SegmentationHead
from models.heads.mim_head import MIMReconstructionHead

__all__ = [
    "SegmentationHead",
    "MIMReconstructionHead",
]
