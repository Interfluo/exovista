"""Tests for the ExodusII -> PyVista reader (exovista.read_exo)."""
import logging
import os

import numpy as np
import pytest

import exovista

pv = pytest.importorskip("pyvista")

logging.disable(logging.CRITICAL)


def _hex_volume():
    """A 2x2x2 hex grid split into two regions along x."""
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    centers = grid.cell_centers().points
    grid.cell_data["region"] = (centers[:, 0] > 0.5).astype(int)
    return grid


def _connectivity_set(grid):
    """Return cell connectivity as a set of frozensets of point ids."""
    result = []
    for i in range(grid.n_cells):
        result.append(frozenset(grid.get_cell(i).point_ids))
    return result


class TestReadExo:
    def test_roundtrip_points_and_cells(self, tmp_path):
        vol = _hex_volume()
        path = os.path.join(tmp_path, "rt.exo")
        # Tag each cell with its original index via an element field so the
        # block-reordered read can be mapped back to the input ordering.
        orig = np.arange(vol.n_cells, dtype=float)
        exovista.write_exo(
            path, vol, region_key="region",
            element_fields={"orig": orig},
        )

        grid = exovista.read_exo(path)

        assert grid.n_points == vol.n_points
        assert np.allclose(grid.points, vol.points)
        assert grid.n_cells == vol.n_cells

        # The element field is a permutation of the original cell indices.
        read_orig = grid.cell_data["orig"].astype(int)
        assert sorted(read_orig.tolist()) == list(range(vol.n_cells))

        # Connectivity must match cell-for-cell once unpermuted.
        in_conn = _connectivity_set(vol)
        out_conn = _connectivity_set(grid)
        for read_cell, original_index in enumerate(read_orig):
            assert out_conn[read_cell] == in_conn[original_index]

    def test_block_ids_recorded(self, tmp_path):
        vol = _hex_volume()
        path = os.path.join(tmp_path, "blocks.exo")
        exovista.write_exo(path, vol, region_key="region")

        grid = exovista.read_exo(path)
        assert "exo_block_id" in grid.cell_data
        assert "exo_block_name" in grid.cell_data
        # Two regions -> two element blocks.
        assert set(np.unique(grid.cell_data["exo_block_id"])) == {1, 2}

    def test_node_and_element_field_roundtrip(self, tmp_path):
        vol = _hex_volume()
        n = vol.n_cells
        times = np.linspace(0.0, 1.0, 4)
        node_f = np.array([t * vol.points[:, 1] for t in times])
        elem_f = np.outer(times, np.arange(n, dtype=float))
        path = os.path.join(tmp_path, "fields.exo")
        exovista.write_exo(
            path, vol, region_key="region", times=times,
            node_fields={"nf": node_f},
            element_fields={"ef": elem_f, "orig": np.tile(np.arange(n), (4, 1))},
        )

        # Last step (default).
        grid = exovista.read_exo(path)
        assert np.allclose(grid.field_data["times"], times)
        assert np.allclose(grid.point_data["nf"], times[-1] * vol.points[:, 1])

        read_orig = grid.cell_data["orig"].astype(int)
        # ef at last step equals times[-1] * original_cell_index
        expected = times[-1] * read_orig
        assert np.allclose(grid.cell_data["ef"], expected)

        # First step selects different values.
        grid0 = exovista.read_exo(path, time_step=1)
        assert np.allclose(grid0.point_data["nf"], times[0] * vol.points[:, 1])

    def test_2d_mesh_embedded_in_3d(self, tmp_path):
        # A single quad in the z=0 plane.
        quad = pv.UnstructuredGrid(
            np.array([4, 0, 1, 2, 3]),
            np.array([pv.CellType.QUAD], dtype=np.uint8),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        quad.cell_data["region"] = np.array([1])
        path = os.path.join(tmp_path, "quad.exo")
        exovista.write_exo(path, quad, region_key="region")

        grid = exovista.read_exo(path)
        assert grid.n_cells == 1
        assert grid.get_cell(0).type == pv.CellType.QUAD
        assert np.allclose(grid.points[:, 2], 0.0)

    def test_static_mesh_has_empty_times(self, tmp_path):
        vol = _hex_volume()
        path = os.path.join(tmp_path, "static.exo")
        exovista.write_exo(path, vol, region_key="region", save_node_arrays=False)
        grid = exovista.read_exo(path)
        assert grid.field_data["times"].size == 0
        assert grid.n_cells == vol.n_cells
