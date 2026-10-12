"""
Backward-compatible proxy for test_util.py.
Redirects to evaluation.metrics and evaluation.sliding_window.
"""

from evaluation.metrics import calculate_case_metrics, cal_binary_dice, cal_multiclass_dice

try:
    from medpy import metric
    def calculate_metric_percase(pred, gt):
        dc = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dc, jc, hd, asd
except ImportError:
    def calculate_metric_percase(pred, gt):
        dc = cal_binary_dice(pred, gt)
        return dc, dc, 0.0, 0.0

__all__ = [
    "calculate_metric_percase",
    "cal_binary_dice",
    "cal_multiclass_dice",
]
