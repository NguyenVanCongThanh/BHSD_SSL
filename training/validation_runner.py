"""
Validation Runner for 3D Medical Image Segmentation.
Encapsulates sliding window inference orchestration, metrics aggregation,
and visual artifact packaging across single and distributed processes.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from evaluation.sliding_window import evaluate_dataset_sliding_window


class ValidationRunner:
    """
    Coordinates dataset validation via sliding-window inference,
    metric calculation, and visualization extraction.
    """
    def __init__(
        self,
        val_dataset: Any,
        num_classes: int,
        is_binary: bool,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        options: Optional[Dict[str, Any]] = None,
        visual_settings: Optional[Dict[str, Any]] = None,
        num_vis_samples: int = 0
    ):
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.is_binary = is_binary
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.options = options or {}
        self.visual_settings = visual_settings or {}
        self.num_vis_samples = num_vis_samples
        self.last_details: Dict[str, Any] = {}

    def run(
        self,
        model: nn.Module,
        return_visuals: bool = False,
        weights_source: str = 'net'
    ) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
        self.last_details = {}
        count = self.num_vis_samples if return_visuals else 0
        result = evaluate_dataset_sliding_window(
            model=model,
            val_dataset=self.val_dataset,
            num_classes=self.num_classes,
            is_binary=self.is_binary,
            device=self.device,
            log_progress=(self.rank == 0),
            return_visual_samples=count,
            rank=self.rank,
            world_size=self.world_size,
            details_sink=self.last_details,
            visual_case_ids=self.visual_settings.get('case_ids'),
            visual_settings=self.visual_settings,
            **self.options
        )
        self.last_details['weights_source'] = weights_source
        return result

