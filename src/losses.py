"""
Combined Dice + CrossEntropy loss with class weighting.
"""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss


def build_loss(
    num_classes: int,
    class_weights: torch.Tensor | None = None,
    lambda_dice: float = 0.5,
    lambda_ce: float = 0.5,
    ignore_background: bool = False,
) -> nn.Module:
    """
    DiceCELoss — standard for multi-class medical image segmentation.
    Newer MONAI passes class weights via the separate CE weight parameter.
    """
    # Build CE loss separately with weights, then wrap in DiceCELoss
    if class_weights is not None:
        ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    else:
        ce_loss = nn.CrossEntropyLoss()

    class WeightedDiceCELoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.dice_ce = DiceCELoss(
                include_background=not ignore_background,
                to_onehot_y=True,
                softmax=True,
                lambda_dice=lambda_dice,
                lambda_ce=lambda_ce,
                reduction="mean",
            )
            self.ce = ce_loss
            self.lam_dice = lambda_dice
            self.lam_ce   = lambda_ce

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # DiceCELoss expects target shape (B, 1, H, W, D)
            dice_loss = self.dice_ce(pred, target)

            if class_weights is not None:
                # CE expects (B, H, W, D) long target
                target_long = target.squeeze(1).long()
                ce_loss_val = self.ce(pred, target_long)
                return self.lam_dice * dice_loss + self.lam_ce * ce_loss_val

            return dice_loss

    return WeightedDiceCELoss()