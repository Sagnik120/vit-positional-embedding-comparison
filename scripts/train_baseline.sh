#!/bin/bash
# Trains the ORIGINAL ViT (learned additive 1D positional embedding).
set -e
cd "$(dirname "$0")/../src"
echo "[info] Starting Baseline ViT training (100 epochs)... please wait, this takes ~1 hour and prints at the end of each epoch."
python -m common.train --model original --out ../results/baseline "$@"
