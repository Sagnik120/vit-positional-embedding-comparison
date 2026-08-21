#!/bin/bash
# Trains the MODIFIED ViT (2D Rotary Position Embedding, applied per-layer).
set -e
cd "$(dirname "$0")/../src"
echo "[info] Starting Modified ViT (RoPE) training (100 epochs)... please wait, this takes ~1 hour and prints at the end of each epoch."
python -m common.train --model modified --out ../results/modified_rope "$@"
