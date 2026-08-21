# 01. Task Definition

## Assignment

Train two variants of ViT (original paper's positional embedding vs. one
alternative PE scheme of choice), everything else identical, and compare.
Existing ViT implementation allowed (no need to implement the transformer
itself from scratch). Dataset must be one reported in the ViT paper.

## Dataset decision: CIFAR-10

- Reported in the paper (Table 2, Table 5, Figure 3).
- 50,000 train / 10,000 test images, 10 classes — large enough per class
  (5,000/class) to train a small ViT from scratch without immediately
  collapsing to random-guess accuracy, unlike Oxford Pets/Flowers (a few
  thousand images total) or CIFAR-100 (500/class).
- Small 32×32 resolution keeps compute tractable on a laptop.

## Model size decision: small custom-scale ViT, not ViT-Base

ViT-Base (86M params) is designed for JFT-300M-scale pretraining. Training
it from scratch on 50k CIFAR-10 images would badly overfit regardless of
PE choice and would not run in reasonable time on a MacBook. Used a scaled
down configuration instead (see `src/common/config.py`):
image_size=32, patch_size=4 (64 patches), dim=256, depth=6, heads=4,
dim_head=64, mlp_dim=512 → ~3.2M parameters.

## Hardware decision

MacBook M5, 24GB unified memory, using PyTorch's MPS backend
(`torch.device("mps")`). No discrete GPU needed at this model/dataset
scale. See `03_environment_setup.md`.

## Overfitting consideration

A small ViT trained from scratch on CIFAR-10 without pretraining is
expected to overfit to some degree (paper Section 4.3 discusses this
directly). This does not invalidate the original-vs-modified comparison:
both variants are trained under identical conditions, and the comparison
is read from best-val-accuracy checkpoints, generalization gap, and loss
curve shape rather than from raw train accuracy. See `report/discussion.md`
for the actual analysis once training completes.
