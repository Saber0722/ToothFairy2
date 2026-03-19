"""
MONAI transform pipelines for train / val / test.
MONAI 1.5.2 — uses ITKReader for .mha files.
"""

import numpy as np
import SimpleITK as sitk
import torch

from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    SpatialPadd,
    ToTensord,
)
from monai.data.image_reader import ITKReader


def get_train_transforms(patch_size: list[int], num_samples: int = 2):
    return Compose([
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
            reader=ITKReader(fallback_only=False),
        ),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image"], dtype="float32"),
        EnsureTyped(keys=["label"], dtype="int64"),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
            mode=("constant", "constant"),
            constant_values=(-1, 0),
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=3,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=-0.9,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.3,
            rotate_range=(0.15, 0.15, 0.15),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms(patch_size: list[int]):
    return Compose([
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
            reader=ITKReader(fallback_only=False),
        ),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image"], dtype="float32"),
        EnsureTyped(keys=["label"], dtype="int64"),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
            mode=("constant", "constant"),
            constant_values=(-1, 0),
        ),
        # Extract multiple random patches — same as train but no augmentation
        RandSpatialCropSamplesd(
            keys=["image", "label"],
            roi_size=patch_size,
            num_samples=4,
            random_center=True,
            random_size=False,
        ),
        ToTensord(keys=["image", "label"]),
    ])


def get_test_transforms():
    return Compose([
        LoadImaged(
            keys=["image"],
            image_only=False,
            reader=ITKReader(fallback_only=False),
        ),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        EnsureTyped(keys=["image"], dtype="float32"),
        ToTensord(keys=["image"]),
    ])