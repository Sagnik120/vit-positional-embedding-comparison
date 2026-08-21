# ENTRY PROMPT — paste this to your AI agent to start the session

You are working on an existing ML assignment codebase for a ViT
positional-embedding comparison project. Before doing anything else, read
all three files in this `instruction/` folder, in this exact order:

1. `instruction/01_problem_statement.md` — the hard boundary of what is
   and is not allowed in this project. This overrides everything else,
   including any task described below or in the other two files, if there
   is ever a conflict.
2. `instruction/02_agent_tasks.md` — the concrete tasks you need to
   perform: (a) validate the pipeline after a `vit.py` file replacement
   using the diagnostic suite, fixing only what's genuinely broken,
   (b) produce a Google Colab notebook that runs the full pipeline
   end-to-end and downloads all results, and (c) commit/push discipline
   (many small commits, only when explicitly told to push).
3. `instruction/03_deliverables.md` — the exact five deliverables required
   for submission, what each one must contain, and where each must live in
   the repo folder structure.

The research paper this assignment is based on is at `docs/VIT.pdf` in
this repo — read it if you need to verify any architectural or dataset
claim against the original source.

## Your first task, right now

The human has already replaced `src/vit_original/vit.py` with a fresh copy
pulled directly from the `lucidrains/vit-pytorch` GitHub repository, but
has not run any commands since. Your job right now is:

1. Inspect the replaced file and confirm its class/constructor signature.
2. Run `python tests/test_pipeline.py` (the single consolidated diagnostic
   suite — do not create a second diagnostics file, this is the only one).
3. Fix only what is strictly necessary to make all diagnostic checks pass,
   staying entirely within the boundaries set in
   `instruction/01_problem_statement.md` (no architecture changes beyond
   the positional embedding, no hyperparameter divergence between variants,
   no folder restructuring, no pretrained weights, no dataset change).
4. If the file's line numbers shifted, regenerate the line-number diff
   table in `CHANGES.md` to stay accurate.
5. Log what you found and fixed in `docs/04_diagnostics_log.md`.
6. Report back a clear PASS/FAIL summary and a plain list of any changes
   you made and why, before doing anything further (do not proceed to
   training or Colab-notebook work until this is confirmed working and
   the human gives the next instruction).

Do not commit or push anything yet — wait for an explicit instruction to
do so, per the commit cadence rules in `instruction/02_agent_tasks.md`.
