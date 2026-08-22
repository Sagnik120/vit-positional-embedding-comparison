# 06. Results Log

Final numbers and where every deliverable artifact lives, once
`scripts/evaluate_all.py` has been run.

| Deliverable | File |
|---|---|
| Top-1 test accuracy, both variants | `results/comparison/top1_test_accuracy_comparison.json` |
| Combined train/val loss curves | `results/comparison/combined_loss_curves.png` |
| Combined train/val accuracy curves | `results/comparison/combined_accuracy_curves.png` |
| Comparison table | `results/comparison/comparison_table.md` |
| Generalization gap comparison | `results/comparison/generalization_gap_comparison.csv` |
| Justification (≤1 page) | `report/justification.md` |
| Discussion (≤1 page) | `report/discussion.md` |
| Original ViT full codebase | `src/vit_original/vit.py` |
| Modified ViT full codebase | `src/vit_modified/vit.py`, `src/vit_modified/positional_embeddings.py` |
| Line-numbered change log | `CHANGES.md` |

## Final numbers snapshot

| Metric | Original ViT | Modified ViT (RoPE) |
|---|---|---|
| Top-1 Test Accuracy | 83.73% | 86.77% |
| Best Val Accuracy | 0.8428 | 0.8706 |
| Best Epoch | 80 | 90 |
| Generalization Gap (train_acc - val_acc @ best epoch) | 0.0547 | 0.0828 |

Delta (Modified - Original) Top-1 Test Accuracy: **3.04 pp**
