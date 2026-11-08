"""Every script in examples/ runs (in a scratch directory) and writes its file."""

import glob
import os
import subprocess
import sys

import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples")
SCRIPTS = sorted(glob.glob(os.path.join(EXAMPLES, "*.py")))


def test_examples_exist():
    assert len(SCRIPTS) >= 8


@pytest.mark.parametrize("script", SCRIPTS, ids=[os.path.basename(s) for s in SCRIPTS])
def test_example_runs(tmp_path, script):
    # save_exo.py without arguments runs its offline example only (the others
    # download pyvista's letter_a mesh).
    result = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", script],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    written = [f for f in os.listdir(tmp_path) if f.endswith(".exo")]
    assert written, f"{os.path.basename(script)} wrote no .exo file"
