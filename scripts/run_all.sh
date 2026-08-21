#!/bin/bash
# Full pipeline: download data -> diagnostics -> train both variants ->
# evaluate both on test set -> generate comparison plots/tables.
set -e
cd "$(dirname "$0")/.."

echo "== 1. Downloading CIFAR-10 =="
python scripts/download_data.py

echo "== 2. Running diagnostics =="
python scripts/run_diagnostics.py

echo "== 3. Training original ViT =="
bash scripts/train_baseline.sh

echo "== 4. Training modified ViT (RoPE) =="
bash scripts/train_modified.sh

echo "== 5. Evaluating both on test set =="
cd src
python -m common.evaluate --model original --out ../results/baseline
python -m common.evaluate --model modified --out ../results/modified_rope
cd ..

echo "== 6. Generating comparison plots and tables =="
python scripts/evaluate_all.py

echo "== DONE. See results/comparison/ for final outputs. =="
