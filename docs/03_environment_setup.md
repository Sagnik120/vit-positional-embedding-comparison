# 03. Environment Setup

## Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Deactivate with `deactivate` when done. Re-activate with
`source .venv/bin/activate` in future sessions.

## MacBook M5 (Apple Silicon) — MPS backend notes

PyTorch supports Apple's Metal Performance Shaders (MPS) as a GPU backend
on Apple Silicon. `src/common/utils.py::get_device("auto")` automatically
selects `torch.device("mps")` when available, falling back to CPU.

Things to watch for:
- First run of any new op on MPS can be noticeably slower (kernel
  compilation); subsequent epochs are much faster.
- If you hit an "operator not implemented for MPS backend" error for some
  PyTorch op, set the environment variable
  `PYTORCH_ENABLE_MPS_FALLBACK=1` before running, which silently falls
  back to CPU for that one op:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
- 24GB unified memory is generous for this model size (~3M params,
  batch size 128, 32×32 images) — no memory pressure expected. If you do
  see memory warnings, reduce `batch_size` in `src/common/config.py`.
- Confirm MPS is actually being used by checking the printed
  `[info] using device: mps` line at the start of `train.py`.

## Verifying the environment

```bash
python -c "import torch; print(torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

Then run the full diagnostic suite before any real training:

```bash
python scripts/run_diagnostics.py
```
