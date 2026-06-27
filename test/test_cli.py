"""Smoke tests for the exoread command-line interface (exovista.exoread)."""
import io
import logging
import os

import numpy as np
import pytest

import exovista

pv = pytest.importorskip("pyvista")

logging.disable(logging.CRITICAL)


def _write_sample(path):
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    grid.cell_data["region"] = np.ones(grid.n_cells, dtype=int)
    grid.point_data["temp"] = grid.points[:, 0]
    exovista.write_exo(path, grid, region_key="region")
    return grid


def test_exoread_describe(tmp_path):
    path = os.path.join(tmp_path, "cli.exo")
    grid = _write_sample(path)

    buf = io.StringIO()
    exovista.exoread([path], file=buf)
    out = buf.getvalue()

    assert "Title:" in out
    assert f"Num nodes   : {grid.n_points}" in out
    assert f"Num elements: {grid.n_cells}" in out


def test_exoread_print_node_variable(tmp_path):
    path = os.path.join(tmp_path, "cli_vars.exo")
    _write_sample(path)

    buf = io.StringIO()
    # -n selects a node variable; -i -1 picks the last time step.
    exovista.exoread(["-n", "temp", "-i", "-1", path], file=buf)
    out = buf.getvalue()

    assert "temp" in out
    # The output should contain at least one numeric value line.
    assert any(ch.isdigit() for ch in out)
