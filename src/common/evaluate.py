"""
evaluate.py

Loads a variant's best checkpoint and computes Top-1 test accuracy on the
official CIFAR-10 test set (10,000 images, never used in training/val).

Usage:
    python -m common.evaluate --model original --out results/baseline
    python -m common.evaluate --model modified --out results/modified_rope
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.config import ModelConfig, TrainConfig, DataConfig
from common.dataset import get_dataloaders
from common.utils import get_device, load_checkpoint, write_json
from common.train import build_model, run_epoch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["original", "modified"], required=True)
    parser.add_argument("--out", required=True, help="e.g. results/baseline")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    data_cfg = DataConfig()

    device = get_device(args.device)
    _, _, test_loader = get_dataloaders(data_cfg, train_cfg)

    model = build_model(args.model, model_cfg).to(device)
    ckpt_path = os.path.join(args.out, "checkpoints", "best.pt")
    ckpt = load_checkpoint(model, ckpt_path, device)
    print(f"[info] loaded checkpoint from epoch {ckpt['epoch']} "
          f"(best_val_acc={ckpt['best_val_acc']:.4f})")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None,
                                     device=device, train=False)

    result = {
        "model": args.model,
        "test_loss": test_loss,
        "top1_test_accuracy": test_acc,
        "top1_test_accuracy_pct": round(test_acc * 100, 2),
        "checkpoint_epoch": ckpt["epoch"],
        "best_val_acc": ckpt["best_val_acc"],
    }
    out_path = os.path.join(args.out, "metrics", "test_accuracy.json")
    write_json(result, out_path)
    print(f"[done] Top-1 test accuracy: {result['top1_test_accuracy_pct']}% "
          f"-> written to {out_path}")


if __name__ == "__main__":
    main()
