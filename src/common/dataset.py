"""
dataset.py

CIFAR-10 loading + augmentation. Identical for both the original and
modified ViT runs -- imported by train.py, never duplicated.

Split: CIFAR-10's official 50,000 train images are split into
train / val (val_fraction from config), official 10,000 test images are
used ONLY at the very end for the Top-1 test accuracy deliverable.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from common.config import DataConfig, TrainConfig


def build_transforms(cfg: DataConfig):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=cfg.random_crop_padding),
        transforms.RandomHorizontalFlip(p=cfg.horizontal_flip_prob),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])
    return train_tf, eval_tf


def get_dataloaders(data_cfg: DataConfig, train_cfg: TrainConfig):
    train_tf, eval_tf = build_transforms(data_cfg)

    full_train = datasets.CIFAR10(
        root=data_cfg.data_root, train=True, download=True, transform=train_tf
    )
    # A second copy with eval-time transforms, so the held-out val split
    # is NOT augmented (important for a clean val-loss/val-acc signal).
    full_train_eval_tf = datasets.CIFAR10(
        root=data_cfg.data_root, train=True, download=True, transform=eval_tf
    )

    n_total = len(full_train)
    n_val = int(n_total * train_cfg.val_fraction)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(train_cfg.seed)
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    train_set = torch.utils.data.Subset(full_train, train_indices.indices)
    val_set = torch.utils.data.Subset(full_train_eval_tf, val_indices.indices)

    test_set = datasets.CIFAR10(
        root=data_cfg.data_root, train=False, download=True, transform=eval_tf
    )

    train_loader = DataLoader(
        train_set, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick standalone sanity check: `python -m common.dataset`
    data_cfg = DataConfig()
    train_cfg = TrainConfig()
    train_loader, val_loader, test_loader = get_dataloaders(data_cfg, train_cfg)
    xb, yb = next(iter(train_loader))
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}, "
          f"test batches: {len(test_loader)}")
    print(f"sample batch: x={tuple(xb.shape)}, y={tuple(yb.shape)}, "
          f"y range=({yb.min().item()},{yb.max().item()})")
