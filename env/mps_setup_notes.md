# MacBook M5 (MPS backend) setup notes

See also `docs/03_environment_setup.md` for the full narrative version.

Quick reference:

```bash
# Confirm MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"

# If some op isn't yet implemented for MPS, fall back to CPU for that op only:
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

`src/common/utils.py::get_device("auto")` picks MPS automatically when
available. Verify by checking the `[info] using device: mps` line printed
at the start of every `train.py` run.
