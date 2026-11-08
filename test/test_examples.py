"""Every script in examples/ runs (in a scratch directory) and writes its file."""

import glob
import os
import subprocess
import sys
from importlib.metadata import entry_points

import pytest
import pyvista as pv

EXAMPLES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples")
SCRIPTS = sorted(glob.glob(os.path.join(EXAMPLES, "*.py")))


#: Examples that call ``mesh.save("x.exo")`` (exovista's pyvista.writers plugin).
NEEDS_SAVE_PLUGIN = {"example_pyvista_save.py"}


def _has_save_plugin():
    """exovista's pyvista.writers entry points are installed."""
    if not hasattr(pv, "registered_writers"):  # PyVista < 0.48 (unsupported)
        return False
    return ".exo" in {ep.name for ep in entry_points(group="pyvista.writers")}


def test_examples_exist():
    assert len(SCRIPTS) >= 10


@pytest.mark.parametrize("script", SCRIPTS, ids=[os.path.basename(s) for s in SCRIPTS])
def test_example_runs(tmp_path, script):
    if os.path.basename(script) in NEEDS_SAVE_PLUGIN and not _has_save_plugin():
        pytest.skip("mesh.save('x.exo') needs exovista installed (pip install -e .)")
    # save_exo.py without arguments runs its offline example only (the others
    # download pyvista's letter_a mesh).
    result = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", script],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    written = [f for f in os.listdir(tmp_path) if f.endswith((".exo", ".e"))]
    assert written, f"{os.path.basename(script)} wrote no .exo file"
