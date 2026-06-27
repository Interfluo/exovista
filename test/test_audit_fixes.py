"""Regression tests for assorted correctness fixes (see audit commit)."""
import logging
import os

import numpy as np
import pytest

import exovista

pv = pytest.importorskip("pyvista")

logging.disable(logging.CRITICAL)


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
    # Helper arrays must not have leaked onto the caller's mesh.
    assert "orig_pts" not in vol.point_data
    assert "orig_cell_ids" not in vol.cell_data


def test_write_exo_does_not_add_default_region(tmp_path):
    vol = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    assert "region" not in vol.cell_data
    exovista.write_exo(os.path.join(tmp_path, "y.exo"), vol)
    # process_meshes injects a default region when missing; it must stay on the
    # internal copy, not the user's mesh.
    assert "region" not in vol.cell_data


def test_invalid_block_id_raises_valueerror(tmp_path):
    vol = _hex_volume()
    path = os.path.join(tmp_path, "z.exo")
    exovista.write_exo(path, vol, region_key="region")

    with exovista.File(path) as exof:
        valid = exof.get_element_block_ids().tolist()
        bad_id = max(valid) + 1000
        with pytest.raises(ValueError):
            exof.get_element_block(bad_id)
