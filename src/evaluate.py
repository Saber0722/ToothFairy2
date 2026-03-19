"""
Evaluation script — runs on test set, computes per-class Dice + Hausdorff.
Usage:
    python -m src.evaluate --checkpoint results/models/best.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from tqdm import tqdm

from src.dataset import build_loader
from src.model import build_model
from src.utils import (get_logger, load_config, load_label_mapping,
                        load_manifest, set_seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--checkpoint", default="results/models/best.pth",
                   help="Path to trained .pth checkpoint")
    p.add_argument("--split",      default="test",
                   help="Which split to evaluate: val | test")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg  = load_config(args.config)
    root = Path(".")
    set_seed(cfg["project"]["seed"])

    splits_dir  = root / cfg["paths"]["splits"]
    graphs_dir  = root / cfg["paths"]["graphs"]
    graphs_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("evaluate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    _, idx_to_label, num_classes = load_label_mapping(splits_dir)
    patch_size = cfg["training"]["patch_size"]

    #  load model 
    model_name = cfg["training"]["model"]
    model      = build_model(model_name, num_classes, patch_size)
    ckpt       = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model      = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")

    #  metrics 
    dice_metric = DiceMetric(
        include_background=False,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )
    post_pred  = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    #  loader 
    loader = build_loader(splits_dir, args.split, patch_size,
                          batch_size=1,
                          num_workers=cfg["training"]["num_workers"],
                          cache_rate=0.0)

    per_case_dice = []
    manifest      = load_manifest(splits_dir, args.split)

    for i, batch in enumerate(tqdm(loader, desc=f"Evaluating {args.split}")):
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # sliding window inference for full-volume prediction
        logits = sliding_window_inference(
            inputs=imgs,
            roi_size=tuple(patch_size),
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )

        preds_onehot  = [post_pred(p)  for p in logits]
        labels_onehot = [post_label(l) for l in labels]
        dice_metric(y_pred=preds_onehot, y=labels_onehot)

        # per-case dice
        case_result, _ = dice_metric.aggregate()
        dice_metric.reset()
        case_mean = case_result.nanmean().item()
        per_case_dice.append({
            "case_id": manifest[i]["case_id"],
            "mean_dice": case_mean,
            **{f"dice_cls{j}": case_result[j].item()
               for j in range(len(case_result))},
        })

    #  aggregate 
    results_df = pd.DataFrame(per_case_dice)
    mean_dice  = results_df["mean_dice"].mean()
    logger.info(f"\nOverall Mean Dice ({args.split}): {mean_dice:.4f}")

    # per-class summary
    dice_cols = [c for c in results_df.columns if c.startswith("dice_cls")]
    class_means = results_df[dice_cols].mean()
    logger.info("\nPer-class Dice (mean over cases):")
    for col, val in class_means.items():
        cls_idx  = int(col.replace("dice_cls", ""))
        raw_lbl  = idx_to_label.get(cls_idx + 1, cls_idx + 1)  # +1: bg excluded
        logger.info(f"  cls={cls_idx:2d} (FDI={raw_lbl:2d}): {val:.4f}")

    #  save 
    out_csv = graphs_dir / f"eval_{args.split}_results.csv"
    results_df.to_csv(out_csv, index=False)
    logger.info(f"\nResults saved to {out_csv}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "overall_mean_dice": mean_dice,
        "per_class_dice": class_means.to_dict(),
    }
    with open(graphs_dir / f"eval_{args.split}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()