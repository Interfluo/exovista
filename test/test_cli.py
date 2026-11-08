"""Tests for the ``exovista-read`` command-line interface (exovista.exoread)."""
import io
import os
import subprocess
import sys
import sysconfig
from importlib import metadata

import numpy as np
import pytest
import pyvista as pv

import exovista
from exovista.exoread import main


def _write_sample(path):
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    grid.cell_data["region"] = np.ones(grid.n_cells, dtype=int)
    grid.point_data["temp"] = grid.points[:, 0]
    exovista.write_exo(path, grid, region_key="region")
    return grid


def _console_script():
    """Path of the installed ``exovista-read`` script, or skip."""
    try:
        dist = metadata.distribution("exovista")
    except metadata.PackageNotFoundError:
        pytest.skip("exovista is not installed (running from a source tree)")
    if not any(ep.name == "exovista-read" for ep in dist.entry_points):
        pytest.fail("the installed exovista has no exovista-read console script")
    name = "exovista-read.exe" if sys.platform == "win32" else "exovista-read"
    script = os.path.join(sysconfig.get_path("scripts"), name)
    if not os.path.exists(script):
        pytest.skip(f"console script not found at {script}")
    return script


def test_exoread_describe(tmp_path):
    path = os.path.join(tmp_path, "cli.exo")
    grid = _write_sample(path)

    buf = io.StringIO()
    main([path], file=buf)
    out = buf.getvalue()

    assert "Title:" in out
    assert f"Num nodes   : {grid.n_points}" in out
    assert f"Num elements: {grid.n_cells}" in out


def test_exoread_print_node_variable(tmp_path):
    path = os.path.join(tmp_path, "cli_vars.exo")
    _write_sample(path)

    buf = io.StringIO()
    # -n selects a node variable; -i -1 picks the last time step.
    main(["-n", "temp", "-i", "-1", path], file=buf)
    out = buf.getvalue()

    assert "temp" in out
    # The output should contain at least one numeric value line.
    assert any(ch.isdigit() for ch in out)


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_reports_exovista_version(capsys, flag):
    with pytest.raises(SystemExit) as exc:
        main([flag])
    assert exc.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"exovista-read (exovista {exovista.__version__})"


def test_help_uses_new_program_name(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("usage: exovista-read")


def test_missing_file_argument_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["-n", "temp"])
    assert exc.value.code == 2
    assert "exovista-read" in capsys.readouterr().err


def test_python_dash_m(tmp_path):
    """``python -m exovista`` is exovista-read (for when scripts are not on PATH)."""
    path = os.path.join(tmp_path, "m.exo")
    grid = _write_sample(path)
    result = subprocess.run(
        [sys.executable, "-m", "exovista", path],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Warning" not in result.stderr  # e.g. runpy's "found in sys.modules"
    assert f"Num elements: {grid.n_cells}" in result.stdout

    version = subprocess.run(
        [sys.executable, "-m", "exovista", "--version"],
        capture_output=True, text=True, timeout=300,
    )
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == f"exovista-read (exovista {exovista.__version__})"


def test_console_script_entry_point():
    try:
        dist = metadata.distribution("exovista")
    except metadata.PackageNotFoundError:
        pytest.skip("exovista is not installed (running from a source tree)")
    scripts = {
        ep.name: ep.value for ep in dist.entry_points if ep.group == "console_scripts"
    }
    # 1.x installed "exoread.py", which collides with upstream exodusii's script.
    assert scripts == {"exovista-read": "exovista.exoread:main"}


def test_console_script_runs(tmp_path):
    script = _console_script()
    version = subprocess.run(
        [script, "--version"], capture_output=True, text=True, timeout=300
    )
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == f"exovista-read (exovista {exovista.__version__})"

    path = os.path.join(tmp_path, "script.exo")
    _write_sample(path)
    result = subprocess.run(
        [script, "-n", "temp", path], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert "temp" in result.stdout
