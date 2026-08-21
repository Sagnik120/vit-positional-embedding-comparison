# 02 — Agent Tasks (What To Actually Do, In Order)

> Read `01_problem_statement.md` FIRST and treat it as the hard boundary
> for everything below. If any task here seems to require going outside
> that boundary, stop and flag it instead of proceeding.

This file covers three jobs, in order:
1. Validate the existing codebase (diagnostics) after the `vit.py` file
   replacement — fix only what's broken, nothing else.
2. Produce a Google Colab notebook (`.ipynb`) that runs the entire
   pipeline end-to-end and downloads every result back to the local
   machine.
3. Commit and push the work in many small, well-spaced, professional
   commits across three platforms.

---

## Job 1 — Validate the codebase after the `vit.py` replacement

### Context
The human replaced `src/vit_original/vit.py` with a fresh copy pulled
directly from the `lucidrains/vit-pytorch` GitHub repo. No commands have
been run since. Your job is to confirm the whole pipeline still works
end-to-end with this replacement, without changing anything beyond what's
strictly necessary to make it work.

### Constraints on this job specifically
- **Do not modify `src/vit_modified/vit.py`'s positional embedding logic.**
  RoPE stays exactly as implemented.
- **Do not change `src/common/config.py` hyperparameters** unless a genuine
  dimensionality mismatch requires it (e.g. if the new `vit.py` has a
  different required `dim_head` divisibility constraint) — and if you do,
  log the exact reason in `docs/04_diagnostics_log.md`.
- **Do not restructure the folder layout.** The following structure must
  still exist, unchanged, after this job:
  ```
  src/vit_original/vit.py
  src/vit_modified/vit.py
  src/vit_modified/positional_embeddings.py
  src/common/{config,dataset,train,evaluate,utils}.py
  scripts/{download_data.py, train_baseline.sh, train_modified.sh, run_all.sh, evaluate_all.py, run_diagnostics.py}
  tests/test_pipeline.py
  results/{baseline,modified_rope,comparison}/...
  report/{justification.md, discussion.md}
  docs/*.md
  CHANGES.md, README.md, LICENSE, requirements.txt
  ```
- All diagnostic logic lives in **one file**: `tests/test_pipeline.py`.
  `scripts/run_diagnostics.py` is only a thin wrapper that calls it — keep
  it that way. Do not create a second, competing diagnostics file.

### Steps

1. **Inspect the replaced file.**
   ```bash
   cat -n src/vit_original/vit.py
   ```
   Confirm it defines a class `ViT` with constructor arguments:
   `image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool,
   channels, dim_head, dropout, emb_dropout`. This is the exact signature
   `src/common/train.py::build_model()` calls with. If the replaced file
   uses different argument names or a different class name, this is the
   first thing that will break — check it before running anything.

2. **Run the full diagnostic suite.**
   ```bash
   source .venv/bin/activate   # create venv first per README.md if not done
   pip install -r requirements.txt
   python tests/test_pipeline.py
   ```
   This runs all checks: imports, device selection, config sanity, dataset
   loader shapes, forward+backward pass for BOTH variants, output-shape
   equality between variants, parameter-count parity, RoPE norm-preservation
   math, one-batch overfit smoke tests, and a full mini train+val loop for
   both variants. All checks must print PASS.

3. **If any check FAILS**, diagnose using the traceback printed for that
   specific check, fix the **minimum necessary code** to resolve it, and
   re-run. Common failure modes to expect specifically from a `vit.py`
   swap:
   - **Constructor signature mismatch** → adjust the keyword arguments in
     `src/common/train.py::build_model()` to match the new file's actual
     parameter names (only if genuinely different — do not "guess" new
     names).
   - **Different variable/attribute names inside `ViT`** (e.g. if the new
     copy names the patch-embedding module differently) → this would only
     matter if `vit_modified/vit.py` is supposed to structurally mirror
     `vit_original/vit.py` line-for-line; if the replacement changes the
     internal structure enough that the RoPE modifications in
     `vit_modified/vit.py` no longer make sense as a diff against it, flag
     this explicitly rather than silently rewriting `vit_modified/vit.py`
     to match — that decision needs human sign-off, since it changes what
     `CHANGES.md` documents.
   - **Different output shape** (e.g. a different pooling default) →
     verify against `ModelConfig` in `src/common/config.py`; do not change
     `num_classes` or expected output shape assumptions elsewhere without
     understanding why they differ.
   - **`dim_head` divisibility assertion failing** in
     `RotaryPositionalEmbedding2D.__init__` (requires `dim_head % 4 == 0`)
     — this constraint is intentional (needed for 2D-axial RoPE indexing)
     and must NOT be loosened; instead adjust `dim_head` in
     `ModelConfig` if truly necessary, applied to both variants equally.

4. **Regenerate `CHANGES.md` if the replaced file's line numbers shifted.**
   ```bash
   diff -u src/vit_original/vit.py src/vit_modified/vit.py
   ```
   Update the line-number table in `CHANGES.md` to match the new line
   numbers exactly. Do not leave stale line-number references.

5. **Log the outcome** in `docs/04_diagnostics_log.md` — paste the full
   PASS/FAIL summary block, note what (if anything) was fixed and why,
   with a timestamp.

6. **Only once all checks PASS**, the pipeline is considered validated and
   ready for the actual training job (Job 2, below).

---

## Job 2 — Google Colab notebook (`.ipynb`) for full-pipeline training

### Goal
One self-contained Colab notebook that, run top to bottom with zero manual
intervention beyond mounting Google Drive, does the ENTIRE pipeline and
ends with every result downloadable back to the local machine:

```
clone repo → install deps → download CIFAR-10 → run diagnostics →
train original ViT → train modified (RoPE) ViT → evaluate both on test
set → generate comparison plots/tables → zip all results → download zip
```

### Required notebook structure (cell-by-cell)

1. **Markdown header cell** — title, link back to the GitHub repo, note
   that this notebook trains both ViT variants for the assignment
   described in `instruction/01_problem_statement.md`.

2. **GPU check cell**
   ```python
   import torch
   print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
   ```
   Remind the user (in a markdown cell above it) to set
   **Runtime → Change runtime type → T4 GPU** before running.

3. **Clone the repo cell**
   ```python
   !git clone https://github.com/Sagnik120/vit-positional-embedding-comparison.git
   %cd vit-positional-embedding-comparison
   ```

4. **Install dependencies cell**
   ```python
   !pip install -q -r requirements.txt
   ```

5. **Download CIFAR-10 cell**
   ```python
   !python scripts/download_data.py
   ```

6. **Run diagnostics cell** — must run and print PASS for all checks
   before proceeding; if it fails, the notebook should stop (raise) rather
   than silently continue to training.
   ```python
   !python tests/test_pipeline.py
   ```

7. **Train original ViT cell**
   ```python
   !bash scripts/train_baseline.sh
   ```

8. **Train modified (RoPE) ViT cell**
   ```python
   !bash scripts/train_modified.sh
   ```

9. **Evaluate both on test set cell**
   ```python
   %cd src
   !python -m common.evaluate --model original --out ../results/baseline
   !python -m common.evaluate --model modified --out ../results/modified_rope
   %cd ..
   ```

10. **Generate comparison plots/tables cell**
    ```python
    !python scripts/evaluate_all.py
    ```

11. **Display results inline cell** — show `results/comparison/comparison_table.md`
    contents and display both PNG plots inline with `IPython.display.Image`
    so results are visible without downloading anything yet.

12. **Zip everything cell**
    ```python
    !zip -r vit_pe_comparison_results.zip results/ report/ docs/ CHANGES.md
    ```

13. **Download zip cell**
    ```python
    from google.colab import files
    files.download('vit_pe_comparison_results.zip')
    ```
    This must trigger an actual browser download of: both models'
    checkpoints, both models' training logs (CSV), both models'
    metrics (JSON), the combined loss/accuracy curve PNGs, the
    comparison table, and the report markdown files — i.e. everything
    that needs to end up back in the local codebase folder structure,
    in one click.

14. **(Optional but requested) Download individual checkpoint files
    separately cell** — in case the zip is large, also offer:
    ```python
    files.download('results/baseline/checkpoints/best.pt')
    files.download('results/modified_rope/checkpoints/best.pt')
    ```

### Constraints on the notebook
- Every cell must be idempotent/safe to re-run (e.g. dataset download
  skips if already present — already true of `scripts/download_data.py`).
- Do not change any hyperparameter in the notebook independently of
  `src/common/config.py` — the notebook must call the exact same scripts
  the local pipeline uses, not reimplement training inline. This guarantees
  the Colab-trained models are produced by the identical, already-reviewed
  pipeline, not a parallel one that could drift from it.
- If GPU memory or time constraints require adjusting `epochs` or
  `batch_size`, that override must go through the `--epochs` CLI flag
  already supported by `train.py`, applied identically to both training
  cells — never a different value for each variant.

---

## Job 3 — Commit and push cadence (GitHub / Kaggle / Hugging Face)

### Accounts
- GitHub: `Sagnik120`
- Kaggle: `chandrasagnik027`
- Hugging Face: `Sagnik120`

### Requirement
More than **50 total commits** across the life of this project, pushed to
GitHub, each commit being a small, meaningful, professionally-scoped unit
of work — not one giant commit, and not 50 trivial whitespace commits
either. Each commit message should describe one real, reviewable change.

### Cadence
- Space commits out over time — roughly **one commit every 30–60 minutes**
  of active work, not all 50 in a burst.
- **Do not proactively run and push automatically in a loop.** Only commit
  and push when the human explicitly says to (e.g. "push now", "commit
  this"). Do not spend extra tool calls/tokens pushing speculatively.
- When told to push, make **one well-scoped commit** for the work done
  since the last commit — do not batch multiple unrelated changes into
  one commit "to save time," and do not split one small change into
  multiple pointless commits either.

### Example commit message granularity (for reference, not a literal script to run)
```
feat: replace src/vit_original/vit.py with fresh upstream copy
fix: adjust build_model() kwargs to match replaced vit.py signature
docs: regenerate CHANGES.md line numbers after vit.py replacement
test: consolidate diagnostics into tests/test_pipeline.py
fix: resolve dim_head assertion failure after vit.py replacement
docs: log diagnostic run results in docs/04_diagnostics_log.md
feat: add Colab notebook for full pipeline training
docs: add step-by-step Colab usage instructions to README
chore: add results download cell to Colab notebook
feat(train): log epoch timing to train_log.csv
results: add baseline ViT training run logs
results: add modified RoPE ViT training run logs
feat: generate combined loss/accuracy comparison plots
docs: fill in report/discussion.md with actual trained results
chore: bump requirements.txt pinned versions after Colab run
```

### Kaggle / Hugging Face
- Kaggle (`chandrasagnik027`): if a Kaggle notebook/dataset version of this
  pipeline is created, mirror the same commit-cadence discipline for
  notebook versions (Kaggle's "Save Version" acts like a commit).
- Hugging Face (`Sagnik120`): if trained checkpoints are pushed to a HF
  model repo for backup/sharing, this should happen as a small number of
  clearly-labeled commits (e.g. "add original ViT checkpoint",
  "add modified RoPE ViT checkpoint"), not bundled into the GitHub commit
  cadence above.
- Neither Kaggle nor Hugging Face pushes should happen automatically —
  only when explicitly requested, same rule as GitHub.
