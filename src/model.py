"""
Model factory — supports SwinUNETR (primary) and nnUNet-style UNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.nets import UNet
from monai.networks.layers import Norm


def build_model(
    model_name: str,
    num_classes: int,
    patch_size: list[int],
    in_channels: int = 1,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "swin_unetr":
        # MONAI >= 1.3 removed img_size; use feature_size + spatial_dims only
        try:
            from monai.networks.nets import SwinUNETR
            model = SwinUNETR(
                in_channels=in_channels,
                out_channels=num_classes,
                feature_size=48,
                use_checkpoint=True,
                spatial_dims=3,
            )
        except TypeError:
            # fallback for older MONAI that still needs img_size
            from monai.networks.nets import SwinUNETR
            model = SwinUNETR(
                img_size=tuple(patch_size),
                in_channels=in_channels,
                out_channels=num_classes,
                feature_size=48,
                use_checkpoint=True,
                spatial_dims=3,
            )
        return model

    elif model_name in ("unet3d", "nnunet"):
        # nnUNet-style: deeper channels, residual units, instance norm
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=(32, 64, 128, 256, 320),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.INSTANCE,
            dropout=0.0,
            act="LEAKYRELU",
        )
        return model

    else:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Choose 'swin_unetr', 'unet3d', or 'nnunet'."
        )


class ModelWithSlidingWindow(nn.Module):
    """Wraps any model with MONAI sliding_window_inference for full-volume inference."""

    def __init__(self, model: nn.Module, patch_size: list[int],
                 sw_batch_size: int = 2, overlap: float = 0.5):
        super().__init__()
        self.model = model
        self.patch_size = tuple(patch_size)
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from monai.inferers import sliding_window_inference
        return sliding_window_inference(
            inputs=x,
            roi_size=self.patch_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.model,
            overlap=self.overlap,
            mode="gaussian",
        )