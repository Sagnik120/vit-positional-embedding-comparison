"""
scripts/run_diagnostics.py

Thin backwards-compatible wrapper. The full diagnostic suite now lives in
`tests/test_pipeline.py` (single canonical location for all pipeline
diagnostic checks). This wrapper exists only so existing commands and docs
(`python scripts/run_diagnostics.py`, `bash scripts/run_all.sh`) keep working
unchanged.

Equivalent to running:
    python tests/test_pipeline.py
"""

import os
import runpy
import sys

if __name__ == "__main__":
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    test_file = os.path.join(repo_root, "tests", "test_pipeline.py")
    sys.argv[0] = test_file
    runpy.run_path(test_file, run_name="__main__")
