"""
Main training script — memory-optimised for 8GB GPU.
Usage: python -m src.train
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from src.dataset import build_loader
from src.losses import build_loss
from src.metrics import aggregate_dice, build_metrics
from src.model import build_model
from src.utils import (count_parameters, get_logger, load_class_weights,
                        load_config, load_label_mapping, set_seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--model",   default=None)
    p.add_argument("--resume",  default=None)
    return p.parse_args()


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train_one_epoch(model, loader, optimizer, loss_fn, scaler,
                    device, use_amp, epoch):
    model.train()
    epoch_loss = 0.0
    steps = 0

    for batch in tqdm(loader, desc=f"  Train E{epoch:04d}", leave=False):
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(imgs)          # (B, C, D, H, W) on patch
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        steps += 1

        # free cache after every step on small GPU
        torch.cuda.empty_cache()

    return epoch_loss / max(steps, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, dice_metric, post_pred, post_label,
             device, use_amp, epoch):
    """
    Patch-based validation — same as training but no grad, no augmentation.
    Avoids full-volume sliding window OOM on 8GB GPU.
    """
    model.eval()
    val_loss = 0.0
    steps    = 0

    for batch in tqdm(loader, desc=f"  Val   E{epoch:04d}", leave=False):
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = loss_fn(logits, labels)

        val_loss += loss.item()
        steps    += 1

        # compute dice on CPU to save GPU RAM
        logits_cpu = logits.detach().cpu()
        labels_cpu = labels.detach().cpu()
        preds_onehot  = [post_pred(p)  for p in logits_cpu]
        labels_onehot = [post_label(l) for l in labels_cpu]
        dice_metric(y_pred=preds_onehot, y=labels_onehot)

        del logits, logits_cpu, labels_cpu
        torch.cuda.empty_cache()

    mean_dice, per_class_dice = aggregate_dice(dice_metric)
    return val_loss / max(steps, 1), mean_dice, per_class_dice


def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    root  = Path(".")
    set_seed(cfg["project"]["seed"])

    splits_dir = root / cfg["paths"]["splits"]
    models_dir = root / cfg["paths"]["models"]
    graphs_dir = root / cfg["paths"]["graphs"]
    models_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train", log_file=models_dir / "train.log")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["training"]["amp"] and device.type == "cuda"
    logger.info(f"Device: {device} | AMP: {use_amp}")

    _, _, num_classes = load_label_mapping(splits_dir)
    class_weights     = load_class_weights(splits_dir, device)
    logger.info(f"Num classes: {num_classes}")

    model_name = args.model or cfg["training"]["model"]
    patch_size = cfg["training"]["patch_size"]
    model      = build_model(model_name, num_classes, patch_size).to(device)
    logger.info(f"Model: {model_name} | Params: {count_parameters(model):,}")

    # loss only used during training on patches — safe size
    loss_fn = build_loss(num_classes, class_weights=class_weights,
                         ignore_background=True)

    dice_metric, post_pred, post_label = build_metrics(num_classes)

    lr        = cfg["training"]["lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["num_epochs"], eta_min=lr * 1e-2)
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_dice   = 0.0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    if args.resume and Path(args.resume).exists():
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        history     = ckpt.get("history", history)
        logger.info(f"Resumed from epoch {ckpt['epoch']} | best dice={best_dice:.4f}")

    # train loader: patches (memory safe)
    train_loader = build_loader(
        splits_dir, "train", patch_size,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        num_samples=2, cache_rate=0.0,
    )
    # val loader — same patch-based as train, no augmentation
    val_loader = build_loader(
        splits_dir, "val", patch_size,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        num_samples=4,       # 4 patches per volume
        cache_rate=0.0,
    )

    num_epochs = cfg["training"]["num_epochs"]
    logger.info(f"Starting training for {num_epochs} epochs")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, use_amp, epoch)

        # validate every epoch (use every 2 if still slow)
        val_loss, val_dice, per_class = validate(
            model, val_loader, loss_fn, dice_metric,
            post_pred, post_label,
            device, use_amp, epoch,
        )

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        logger.info(
            f"Epoch {epoch:04d}/{num_epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val Dice={val_dice:.4f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e} | "
            f"Time={elapsed:.1f}s"
        )

        state = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice, "history": history, "config": cfg,
        }
        save_checkpoint(state, models_dir / "latest.pth")

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(state, models_dir / "best.pth")
            logger.info(f"  ★ New best Dice: {best_dice:.4f}")

        if epoch % 10 == 0:
            with open(models_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    with open(models_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. Best Val Dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()