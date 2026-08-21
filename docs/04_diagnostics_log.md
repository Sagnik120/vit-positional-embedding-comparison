# 04 — Diagnostics Log

**Date:** 2026-08-21
**Action:** Validated the pipeline after the human replaced `src/vit_original/vit.py`.

### Results

The diagnostic suite (`python tests/test_pipeline.py`) was run and all 15 checks passed:

```
============================================================
DIAGNOSTIC SUMMARY
============================================================
[PASS] 1. Environment / imports
[PASS] 2. Device availability
[PASS] 3. Config sanity
[PASS] 4. Dataset loader
[PASS] 5. Original ViT forward pass
[PASS] 6. Original ViT backward pass
[PASS] 7. Modified ViT (RoPE) forward pass
[PASS] 8. Modified ViT (RoPE) backward pass
[PASS] 9. Output shape equality (original vs modified)
[PASS] 10. Parameter count sanity (original vs modified)
[PASS] 11. RoPE norm-preservation numerical check
[PASS] 12a. One-batch overfit smoke test (original)
[PASS] 12b. One-batch overfit smoke test (modified)
[PASS] 13a. Full mini pipeline (original, 3 train + 3 val batches)
[PASS] 13b. Full mini pipeline (modified, 3 train + 3 val batches)

15/15 checks passed.
```

### Findings & Changes Made

*   **No code fixes were required to make the tests pass.** The class and constructor signatures in the fresh `vit.py` exactly matched the `build_model()` call in `src/common/train.py`.
*   **CRITICAL ISSUE FOUND & FIXED:** I discovered that `src/vit_modified/vit.py` was initially byte-identical to `src/vit_original/vit.py` (it appears the human accidentally overwrote BOTH files with the fresh upstream copy). Because `src/vit_modified/vit.py` was missing its RoPE implementation entirely, the tests passed only trivially.
*   **Resolution:** After receiving human sign-off, I re-applied the exact RoPE modifications defined in `CHANGES.md` back into `src/vit_modified/vit.py`. 
*   **Tests Re-run:** Re-ran `python tests/test_pipeline.py --skip-download` after fixing the file, and all 15 checks passed correctly (the `modified` ViT now successfully uses the RoPE positional embedding).
*   **Line Numbers Updated:** Regenerated the line-number diff table in `CHANGES.md` to reflect the new exact line numbers (the modified file is now 143 lines long).
