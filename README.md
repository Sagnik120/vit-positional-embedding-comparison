# ViT Positional Embedding Comparison

Comparing the **original ViT positional embedding** (learned additive 1D,
as in Dosovitskiy et al., 2021, *"An Image is Worth 16x16 Words"*) against
a **modified 2D Rotary Position Embedding (RoPE)** variant, trained from
scratch on CIFAR-10.

- GitHub: `Sagnik120/vit-positional-embedding-comparison`
- Author's Hugging Face: `Sagnik120`
- Author's Kaggle: `chandrasagnik027`

See `CHANGES.md` for the exact line-numbered diff between the two model
implementations, and `docs/` for the full process log.

---

## 1. Setup

```bash
git clone https://github.com/Sagnik120/vit-positional-embedding-comparison.git
cd vit-positional-embedding-comparison

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

See `docs/03_environment_setup.md` for MacBook (Apple Silicon / MPS)
specific notes.

## 2. Download CIFAR-10

```bash
python scripts/download_data.py
```

## 3. Run diagnostics (do this before any real training run)

```bash
python scripts/run_diagnostics.py
```

This validates imports, device selection, config sanity, dataset shapes,
forward/backward passes for BOTH models, output-shape equality, parameter
count parity, RoPE numerical correctness, one-batch overfit smoke tests,
and a full mini train/val loop — end to end, in under a minute. All 15
checks must PASS before proceeding.

## 4. Train both variants

```bash
bash scripts/train_baseline.sh      # original ViT -> results/baseline/
bash scripts/train_modified.sh      # modified ViT (RoPE) -> results/modified_rope/
```

Or run everything (download → diagnostics → train both → evaluate → plots)
in one command:

```bash
bash scripts/run_all.sh
```

## 5. Evaluate on the test set

```bash
cd src
python -m common.evaluate --model original --out ../results/baseline
python -m common.evaluate --model modified --out ../results/modified_rope
cd ..
```

## 6. Generate comparison plots and tables

```bash
python scripts/evaluate_all.py
```

Produces:
- `results/comparison/combined_loss_curves.png` — train+val loss, both variants, one plot
- `results/comparison/combined_accuracy_curves.png`
- `results/comparison/top1_test_accuracy_comparison.json`
- `results/comparison/generalization_gap_comparison.csv`
- `results/comparison/comparison_table.md`

---

## Repository structure

```
src/
├── vit_original/vit.py            # unmodified reference ViT (learned 1D PE)
├── vit_modified/
│   ├── vit.py                     # RoPE variant (see CHANGES.md for diff)
│   └── positional_embeddings.py   # PE registry (learned_1d, rope_2d, extensible)
└── common/
    ├── config.py                  # shared hyperparameters (identical for both runs)
    ├── dataset.py                 # CIFAR-10 loading + augmentation
    ├── train.py                   # training loop (--model original|modified)
    ├── evaluate.py                # test-set Top-1 accuracy
    └── utils.py                   # seeding, device selection, checkpointing, logging

scripts/
├── download_data.py               # CIFAR-10 download
├── run_diagnostics.py             # 15-point pipeline sanity check
├── train_baseline.sh / train_modified.sh / run_all.sh
└── evaluate_all.py                # comparison plots + tables

results/
├── baseline/           {checkpoints, logs, visualizations, metrics}
├── modified_rope/      {checkpoints, logs, visualizations, metrics}
└── comparison/         combined plots, comparison_table.md

report/
├── justification.md    # required: why RoPE, what was expected (<=1 page)
└── discussion.md        # required: results vs. expectation (<=1 page)

docs/                    # process log, kept in markdown, chronological
```

## Deliverables checklist

- [x] Top-1 test accuracy for both variants → `results/comparison/top1_test_accuracy_comparison.json`
- [x] Train/val loss curves for both, one plot → `results/comparison/combined_loss_curves.png`
- [x] Written justification (≤1 page) → `report/justification.md`
- [x] Written discussion (≤1 page) → `report/discussion.md`
- [x] Full codebases with line numbers → `src/vit_original/vit.py`, `src/vit_modified/vit.py`
- [x] README of changes with line numbers → `CHANGES.md`
