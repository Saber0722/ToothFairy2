"""
MONAI CacheDataset wrappers for train / val / test splits.
"""

from pathlib import Path

from monai.data import CacheDataset, DataLoader, Dataset

from src.transforms import get_train_transforms, get_val_transforms, get_test_transforms
from src.utils import load_manifest


def build_loader(
    splits_dir: Path,
    split: str,
    patch_size: list[int],
    batch_size: int,
    num_workers: int,
    num_samples: int = 2,
    cache_rate: float = 0.0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for the given split.
    cache_rate: fraction of dataset to cache in RAM (0.0 = no cache, safe for large datasets)
    """
    manifest = load_manifest(splits_dir, split)

    if split == "train":
        transforms = get_train_transforms(patch_size, num_samples=num_samples)
        shuffle = True
        drop_last = True
    elif split == "val":
        transforms = get_val_transforms(patch_size)
        shuffle = False
        drop_last = False
    else:
        transforms = get_val_transforms(patch_size)
        shuffle = False
        drop_last = False

    if cache_rate > 0:
        ds = CacheDataset(data=manifest,
                          transform=transforms,
                          cache_rate=cache_rate,
                          num_workers=num_workers)
    else:
        ds = Dataset(data=manifest, transform=transforms)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    return loader


def build_inference_loader(
    file_list: list[dict],
    num_workers: int = 2,
) -> DataLoader:
    """Inference-only loader — no labels."""
    ds = Dataset(data=file_list, transform=get_test_transforms())
    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=num_workers, pin_memory=True)