# 01 — Problem Statement (READ THIS FIRST, FOLLOW EXACTLY)

> **To the AI agent reading this: this file defines the hard boundary of
> what you are allowed to do. Nothing in this project may go beyond,
> reinterpret, or "improve on" what is written here. If any other
> instruction file, script, or your own judgment conflicts with this
> file, THIS FILE WINS. When in doubt, do less, not more, and ask rather
> than assume.**

The full research paper this assignment is based on is available at:
`docs/VIT.pdf` (Dosovitskiy et al., *"An Image Is Worth 16x16 Words:
Transformers for Image Recognition at Scale"*, ICLR 2021,
https://arxiv.org/pdf/2010.11929). Read it before making any modeling
decision that touches the architecture, dataset, or training setup —
every deviation must be traceable to something either explicitly asked
for in the task below, or explicitly permitted by it.

---

## RAW INSTRUCTION (verbatim, exactly as given by the instructor)

```
Paper reference: Dosovitskiy et al., An Image Is Worth 16x16 Words: Transformers
for Image Recognition at Scale, ICLR 2021. https://arxiv.org/pdf/2010.11929

Task: Use an existing ViT implementation (no need to implement from scratch).
Train the following two variants of ViT and compare them.
1. Original ViT — Positional embedding, as described in the above paper.
2. Modified ViT — Replace the positional embedding module with any alternative
   positional embedding scheme of your choice (any scheme that you can justify).

Everything else of the architecture (model size, patch size, depth, etc.) must
be identical between the two variants. Train both models separately to optimize
the respective validation performance.

Dataset: Choose any one benchmark dataset reported in the paper. Train from
scratch (no pretrained weights).

Deliverables:

* Top-1 test accuracy for both variants of the trained models (original vs. modified).
* Training/validation loss curves for both, on one plot.
* A short (≤1 page) written justification: why did you choose this alternative
  scheme, and what did you expect it to do differently?
* A short (≤1 page) discussion: did your results match your expectation?
* Full codebases of original ViT and modified ViT with line numbers. A separate
  readme file must be added to indicate the changes you have made (with line
  numbers) in the modified ViT compared to the original ViT.

1. Do not use the paper's own embedding ablations verbatim as your proposed
   solution.
2. The submission will be closed after the due date. No submission will be
   accepted through email under any circumstances. Therefore, do not wait
   for the last day to submit.
```

---

## Line-by-line breakdown of what this actually requires

### "Use an existing ViT implementation (no need to implement from scratch)"
- The ViT `nn.Module` architecture (patch embedding, class token, transformer
  encoder blocks, MLP head) must come from an existing, real, publicly
  available codebase — **not written from scratch by the agent**.
- This project uses `src/vit_original/vit.py`, copied unmodified from
  `lucidrains/vit-pytorch` (MIT licensed). **Do not rewrite this file's core
  architecture.** You may only touch it if fixing a genuine bug that breaks
  the pipeline, and if you do, you must log the change in `docs/` and explain
  why it was necessary — never as a stylistic or "improvement" edit.
- Everything else (training loop, data loading, logging, evaluation, plotting)
  is *not* provided by the existing repo and is fair game to write, since the
  instruction only restricts the *model architecture* source, not the full
  pipeline.

### "Train the following two variants... and compare them"
- Exactly **two** models get trained. Not three, not a sweep of five PE
  schemes "for completeness." Two.
- **Variant 1 (Original)**: the learned additive 1D positional embedding
  exactly as in the paper (Eq. 1, Section 3.1). This is `src/vit_original/vit.py`,
  unmodified.
- **Variant 2 (Modified)**: `src/vit_modified/vit.py` — same architecture,
  with the positional embedding mechanism replaced by 2D Rotary Position
  Embedding (RoPE), applied inside attention. This is already implemented in
  this repo. **Do not swap this for a different scheme without being asked.**

### "Everything else of the architecture... must be identical between the two variants"
- Model size (`dim`), patch size, depth, heads, dim_head, mlp_dim, dropout —
  every one of these lives in `src/common/config.py` and is shared by both
  training runs via `src/common/train.py --model {original,modified}`.
- **Never** hardcode a different hyperparameter value into one variant's run
  vs. the other. If you need to change a hyperparameter, change it in
  `config.py` so both variants pick it up identically, and log why in
  `docs/05_training_log.md`.
- The only architectural difference permitted between the two files is the
  positional embedding mechanism itself. `CHANGES.md` documents every single
  line where the two files differ — if `CHANGES.md` grows to include changes
  unrelated to positional embedding, that is a violation of this constraint
  and must be fixed, not left in.

### "Train both models separately to optimize the respective validation performance"
- Each model is trained independently (separate weights, separate optimizer
  state, separate checkpoint). This project already does this
  (`results/baseline/` vs `results/modified_rope/`, entirely separate folders).
- "Optimize... validation performance" means: use the **best-validation-accuracy
  checkpoint** for each variant (early stopping / checkpoint selection), not
  the last-epoch weights, when reporting final numbers. This is already wired
  into `src/common/train.py` (`best.pt` saved only when val_acc improves).
- Do NOT tune hyperparameters *separately* per variant to make one look
  better — that would violate "everything else must be identical." The only
  thing "optimize the respective validation performance" licenses is picking
  the best checkpoint *within* a fixed, shared hyperparameter setting — not
  hyperparameter search per variant.

### "Dataset: Choose any one benchmark dataset reported in the paper. Train from scratch"
- **CIFAR-10** has already been chosen (see `docs/01_task_definition.md` for
  the reasoning) because it is explicitly reported in the paper's tables and
  is small enough to train from scratch on a laptop/Colab.
- **Do not switch datasets** without being explicitly told to. If dataset
  choice is ever revisited, the *only* valid alternatives are datasets
  literally reported in the paper's tables: ImageNet, ImageNet-21k, JFT-300M
  (all infeasible from scratch at this compute budget), CIFAR-10, CIFAR-100,
  Oxford-IIIT Pets, Oxford Flowers-102, or VTAB's 19 tasks.
- "Train from scratch (no pretrained weights)" — **never** load any
  pretrained checkpoint, `torchvision.models(pretrained=True)`,
  `timm.create_model(..., pretrained=True)`, or any Hugging Face pretrained
  weights into either variant, at any point, for any reason. Both models'
  weights must originate from random initialization only. If you ever see
  a `.load_state_dict(...)` call loading anything other than one of this
  project's own checkpoints (for resuming/evaluating a run already trained
  in this project), stop and flag it — it is a violation.

### Deliverables (each one is mandatory, none are optional, none are "bonus")
See `03_deliverables.md` in this same folder for the exhaustive breakdown of
exactly what each deliverable requires and where it must live in this repo.

### Hard constraint #1 — "Do not use the paper's own embedding ablations verbatim as your proposed solution"
- The paper's Appendix D.4 tests: no positional embedding, 1D learned
  (default), 2D learned (row/col concatenated), and relative positional
  embedding (learned bias added to attention logits by relative offset).
- **None of these four may be reworded, relabeled, or presented as "your"
  alternative scheme.** RoPE was specifically chosen because it is
  mechanistically distinct from all four (multiplicative rotation of Q/K
  inside attention, not an additive scheme at any point) — see
  `report/justification.md` and `docs/02_positional_embedding_choice.md` for
  the full argument.
- If at any point a proposed "alternative" scheme turns out to be equivalent
  in mechanism to one of these four ablations under a different name, it must
  be rejected and RoPE (or another genuinely distinct scheme, if the human
  supervising this project explicitly asks to switch) used instead.

### Hard constraint #2 — Submission timing
- This is a human logistics constraint (don't wait until the last day) — it
  does not license the agent to do anything except work efficiently and keep
  the repo in a submittable state at all times. Every commit should leave the
  repo in a state that could, if needed, be submitted as-is.

---

## What the agent must NOT do, under any circumstances

1. Do **not** add a third positional embedding variant, a hyperparameter
   sweep, an ablation study, or any additional experiment beyond the two
   required variants, unless explicitly asked.
2. Do **not** change the model architecture, model size, patch size, depth,
   heads, or any shared hyperparameter differently between the two variants.
3. Do **not** load pretrained weights of any kind, for any reason, into
   either variant.
4. Do **not** switch the dataset away from CIFAR-10 without explicit
   instruction.
5. Do **not** rewrite `src/vit_original/vit.py`'s core architecture — it must
   remain a faithful, recognizable copy of the existing (lucidrains)
   implementation.
6. Do **not** present any of the paper's own Appendix D.4 ablations as the
   "modified" scheme.
7. Do **not** silently change any file's purpose or location in a way that
   breaks the existing, already-reviewed folder structure (see
   `02_agent_tasks.md` for the structure that must be preserved).
8. Do **not** skip, shorten, or omit any of the 5 required deliverables.
9. Do **not** fabricate results, numbers, or plots. Every number in
   `report/discussion.md` and every plot in `results/comparison/` must come
   from an actual training run that was actually executed — never invented,
   estimated, or "written to sound plausible."
10. Do **not** make large, monolithic commits. See `02_agent_tasks.md` for
    the required commit cadence and count.

If any task given to the agent conflicts with anything in this file, the
agent must stop and flag the conflict rather than proceeding.
