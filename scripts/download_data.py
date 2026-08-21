"""
download_data.py

Downloads CIFAR-10 into data/cifar10 (idempotent -- torchvision skips
re-download if already present and verified).

Run this once before training:
    python scripts/download_data.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from torchvision import datasets
from common.config import DataConfig


def main():
    cfg = DataConfig()
    os.makedirs(cfg.data_root, exist_ok=True)

    print(f"[info] downloading CIFAR-10 train split to {cfg.data_root} ...")
    train_set = datasets.CIFAR10(root=cfg.data_root, train=True, download=True)
    print(f"[info] train set size: {len(train_set)}")

    print(f"[info] downloading CIFAR-10 test split to {cfg.data_root} ...")
    test_set = datasets.CIFAR10(root=cfg.data_root, train=False, download=True)
    print(f"[info] test set size: {len(test_set)}")

    print("[done] CIFAR-10 ready.")


if __name__ == "__main__":
    main()
