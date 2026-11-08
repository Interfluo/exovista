"""Every script in examples/ runs (in a scratch directory) and writes its file."""

import glob
import os
import subprocess
import sys
from importlib.metadata import entry_points

import pytest
import pyvista as pv

from _pvcompat import requires_pyvista

EXAMPLES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples")
SCRIPTS = sorted(glob.glob(os.path.join(EXAMPLES, "*.py")))


#: Examples that call ``mesh.save("x.exo")`` (exovista's pyvista.writers plugin).
NEEDS_SAVE_PLUGIN = {"example_pyvista_save.py"}

#: The examples are written for current PyVista: ``extract_surface(algorithm=
#: None)`` needs PyVista 0.47, and writing back ``pv.ExodusIIReader`` output
#: with its side sets and node sets 0.49 (exovista itself needs neither).
NEEDS_PYVISTA = {
    "example_pyvista_save.py": ("0.48", "mesh.save, extract_surface(algorithm=None)"),
    "example_read_back.py": ("0.49", "pv.ExodusIIReader side sets with node sets"),
    "example_read_exo.py": ("0.47", "extract_surface(algorithm=None)"),
    "example_side_sets.py": ("0.47", "extract_surface(algorithm=None)"),
    "save_exo.py": ("0.47", "extract_surface(algorithm=None)"),
}


def _has_save_plugin():
    """exovista's pyvista.writers entry points are installed."""
    if not hasattr(pv, "registered_writers"):  # PyVista < 0.48 (unsupported)
        return False
    return ".exo" in {ep.name for ep in entry_points(group="pyvista.writers")}


def test_examples_exist():
    assert len(SCRIPTS) >= 10


@pytest.mark.parametrize("script", SCRIPTS, ids=[os.path.basename(s) for s in SCRIPTS])
def test_example_runs(tmp_path, script):
    name = os.path.basename(script)
    if name in NEEDS_PYVISTA:
        requires_pyvista(*NEEDS_PYVISTA[name])
    if name in NEEDS_SAVE_PLUGIN and not _has_save_plugin():
        pytest.skip("mesh.save('x.exo') needs exovista installed (pip install -e .)")
    # save_exo.py without arguments runs its offline example only (the others
    # download pyvista's letter_a mesh). Warnings are errors, except the
    # deprecation notices of other packages (pyparsing's PyparsingDeprecation-
    # Warning, raised when older matplotlib is imported, is also a UserWarning).
    result = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-W", "ignore::DeprecationWarning",
         script],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    written = [f for f in os.listdir(tmp_path) if f.endswith((".exo", ".e"))]
    assert written, f"{os.path.basename(script)} wrote no .exo file"
