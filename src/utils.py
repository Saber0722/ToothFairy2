import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_label_mapping(splits_dir: Path) -> tuple[dict, dict, int]:
    with open(splits_dir / "label_mapping.json") as f:
        m = json.load(f)
    label_to_idx = {int(k): int(v) for k, v in m["label_to_idx"].items()}
    idx_to_label = {int(k): int(v) for k, v in m["idx_to_label"].items()}
    return label_to_idx, idx_to_label, int(m["num_classes"])


def load_class_weights(splits_dir: Path, device: torch.device) -> torch.Tensor:
    with open(splits_dir / "class_weights.json") as f:
        w = json.load(f)
    weights = torch.tensor([w[str(i)] for i in range(len(w))],
                           dtype=torch.float32, device=device)
    return weights


def load_manifest(splits_dir: Path, split: str) -> list[dict]:
    with open(splits_dir / f"{split}.json") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)