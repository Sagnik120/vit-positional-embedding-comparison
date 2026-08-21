import os

# Single source of truth for hyperparameters shared by BOTH the original and
# modified ViT training runs. Keeping this identical (except for the PE
# mechanism itself, which lives entirely inside the model classes) is what
# makes the original-vs-modified comparison valid.
# 
# Do not hardcode these values anywhere else -- import from here.
# """

from dataclasses import dataclass

# Get repository root absolute path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))


@dataclass
class ModelConfig:
    image_size: int = 32       # CIFAR-10 native resolution
    patch_size: int = 4        # -> 8x8 = 64 patches
    num_classes: int = 10
    dim: int = 256              # transformer hidden size D
    depth: int = 6               # number of transformer blocks
    heads: int = 4
    mlp_dim: int = 512          # 4 * dim / 2, kept modest for CPU/MPS speed
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.1
    pool: str = "cls"           # match paper's [class] token approach


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0
    seed: int = 42
    num_workers: int = 4
    val_fraction: float = 0.1   # held out from the 50k train set
    early_stopping_patience: int = 20  # epochs with no val-acc improvement


@dataclass
class DataConfig:
    data_root: str = os.path.join(_repo_root, "data", "cifar10")
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)
    random_crop_padding: int = 4
    horizontal_flip_prob: float = 0.5
