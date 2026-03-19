"""
MONAI metric wrappers for multi-class Dice evaluation.
"""

import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction


def build_metrics(num_classes: int, include_background: bool = False):
    """
    Returns (dice_metric, post_pred, post_label) tuple.
    post_pred / post_label convert logits/labels to one-hot for metric computation.
    """
    dice_metric = DiceMetric(
        include_background=include_background,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )
    post_pred  = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)
    return dice_metric, post_pred, post_label


def aggregate_dice(dice_metric) -> tuple[float, torch.Tensor]:
    """
    Returns (mean_dice_scalar, per_class_dice_tensor).
    Resets metric buffer after aggregation.
    """
    result, not_nans = dice_metric.aggregate()   # (num_classes,)
    dice_metric.reset()
    mean_dice = result[not_nans.bool()].mean().item()
    return mean_dice, result