"""Regression tests for assorted 1.x correctness fixes (see audit commit)."""
import os

import numpy as np
import pytest

import exovista.legacy

pv = pytest.importorskip("pyvista")


def _hex_volume():
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    grid.cell_data["region"] = np.ones(grid.n_cells, dtype=int)
    return grid


def test_write_exo_does_not_mutate_input(tmp_path):
    vol = _hex_volume()
    point_keys_before = set(vol.point_data.keys())
    cell_keys_before = set(vol.cell_data.keys())

    exovista.write_exo(os.path.join(tmp_path, "x.exo"), vol, region_key="region")

    assert set(vol.point_data.keys()) == point_keys_before
    assert set(vol.cell_data.keys()) == cell_keys_before
    # Helper arrays must not have leaked onto the caller's mesh...
    assert "orig_pts" not in vol.point_data
    assert "orig_cell_ids" not in vol.cell_data
    # ... nor into the file (1.x wrote an 'orig_pts' node variable).
    with exovista.legacy.File(os.path.join(tmp_path, "x.exo")) as exof:
        for names in (exof.get_node_variable_names(),
                      exof.get_element_variable_names()):
            assert names is None or len(names) == 0


def test_write_exo_does_not_add_default_region(tmp_path):
    vol = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    assert "region" not in vol.cell_data
    exovista.write_exo(os.path.join(tmp_path, "y.exo"), vol)
    # Without a region array the mesh is one region; 1.x injected a default
    # region array, which must never reach the user's mesh.
    assert "region" not in vol.cell_data


def test_invalid_block_id_raises_valueerror(tmp_path):
    vol = _hex_volume()
    path = os.path.join(tmp_path, "z.exo")
    exovista.write_exo(path, vol, region_key="region")

    with exovista.legacy.File(path) as exof:
        valid = exof.get_element_block_ids().tolist()
        bad_id = max(valid) + 1000
        with pytest.raises(ValueError):
            exof.get_element_block(bad_id)


def test_import_leaves_warning_filters_alone():
    """1.x ran warnings.filterwarnings("ignore") at import (nc.py)."""
    import subprocess
    import sys

    code = (
        "import warnings, exovista\n"
        "assert not any(f[0] == 'ignore' and f[2] is Warning and f[1] is None "
        "for f in warnings.filters), warnings.filters\n"
        "warnings.warn('still visible', UserWarning)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "UserWarning: still visible" in result.stderr
