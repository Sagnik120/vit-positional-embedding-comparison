"""
train.py

Single training entry point for BOTH variants:

    python -m common.train --model original --out results/baseline
    python -m common.train --model modified --out results/modified_rope

Everything (model size, optimizer, schedule, augmentation, seed) comes from
common/config.py and is IDENTICAL between the two runs -- the only thing
that differs is which model class gets instantiated.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.config import ModelConfig, TrainConfig, DataConfig
from common.dataset import get_dataloaders
from common.utils import (
    set_seed, get_device, CSVLogger, save_checkpoint, write_json,
    count_parameters,
)


def build_model(model_name: str, model_cfg: ModelConfig):
    if model_name == "original":
        from vit_original.vit import ViT
    elif model_name == "modified":
        from vit_modified.vit import ViT
    else:
        raise ValueError(f"Unknown --model {model_name!r}, expected 'original' or 'modified'")

    return ViT(
        image_size=model_cfg.image_size,
        patch_size=model_cfg.patch_size,
        num_classes=model_cfg.num_classes,
        dim=model_cfg.dim,
        depth=model_cfg.depth,
        heads=model_cfg.heads,
        mlp_dim=model_cfg.mlp_dim,
        pool=model_cfg.pool,
        dim_head=model_cfg.dim_head,
        dropout=model_cfg.dropout,
        emb_dropout=model_cfg.emb_dropout,
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_count += x.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["original", "modified"], required=True)
    parser.add_argument("--out", required=True, help="e.g. results/baseline")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    data_cfg = DataConfig()
    if args.epochs is not None:
        train_cfg.epochs = args.epochs

    set_seed(train_cfg.seed)
    device = get_device(args.device)
    print(f"[info] using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(data_cfg, train_cfg)

    model = build_model(args.model, model_cfg).to(device)
    n_params = count_parameters(model)
    print(f"[info] model={args.model} params={n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    def lr_lambda(epoch):
        if epoch < train_cfg.warmup_epochs:
            return (epoch + 1) / train_cfg.warmup_epochs
        progress = (epoch - train_cfg.warmup_epochs) / max(
            1, train_cfg.epochs - train_cfg.warmup_epochs
        )
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = os.path.join(args.out, "logs", "train_log.csv")
    logger = CSVLogger(log_path, fieldnames=[
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_sec"
    ])

    best_val_acc = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    ckpt_path = os.path.join(args.out, "checkpoints", "best.pt")

    for epoch in range(train_cfg.epochs):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step()
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        logger.log({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
            "epoch_time_sec": elapsed,
        })
        print(f"[{args.model}] epoch {epoch:03d} | train_loss {train_loss:.4f} "
              f"train_acc {train_acc:.4f} | val_loss {val_loss:.4f} "
              f"val_acc {val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(model, optimizer, epoch, best_val_acc, ckpt_path)
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= train_cfg.early_stopping_patience:
            print(f"[info] early stopping at epoch {epoch} "
                  f"(no val improvement for {train_cfg.early_stopping_patience} epochs)")
            break

    write_json({
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "num_params": n_params,
        "checkpoint_path": ckpt_path,
    }, os.path.join(args.out, "metrics", "best_val_checkpoint_info.json"))

    print(f"[done] best val_acc={best_val_acc:.4f} at epoch {best_epoch}. "
          f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
