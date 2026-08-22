# 05. Training Log

## Shared Hyperparameters (from `src/common/config.py`)

- **Image Size:** 32x32 (CIFAR-10)
- **Patch Size:** 4x4 (64 patches total)
- **Embedding Dim:** 256
- **Encoder Depth:** 6 layers
- **Attention Heads:** 4
- **MLP Hidden Dim:** 512
- **Dropout / Emb Dropout:** 0.1 / 0.1
- **Optimizer:** AdamW (learning rate = 3e-4, weight decay = 0.05)
- **Warmup:** 5 epochs
- **Early Stopping Patience:** 20 epochs

---

## Run: Original ViT (`results/baseline/`)

- **Command:** `bash scripts/train_baseline.sh`
- **Device used:** CUDA (Google Colab T4 GPU)
- **Epochs run:** 100 epochs (completed full run)
- **Best val accuracy:** 0.8428 at Epoch 80
- **Test set accuracy:** 83.73%

---

## Run: Modified ViT — RoPE (`results/modified_rope/`)

- **Command:** `bash scripts/train_modified.sh`
- **Device used:** CUDA (Google Colab T4 GPU)
- **Epochs run:** 100 epochs (completed full run)
- **Best val accuracy:** 0.8706 at Epoch 90
- **Test set accuracy:** 86.77%
