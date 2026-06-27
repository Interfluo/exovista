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

    def test_side_set_reconstruction_3d(self, tmp_path):
        from scipy.spatial import cKDTree

        vol = pv.ImageData(dimensions=(4, 4, 4)).cast_to_unstructured_grid()
        vol.cell_data["region"] = np.ones(vol.n_cells, dtype=int)
        surf = vol.extract_surface()
        surf.cell_data["region"] = (
            surf.cell_centers().points[:, 2] > 1.5
        ).astype(int)
        path = os.path.join(tmp_path, "ss3d.exo")
        exovista.write_exo(path, vol, surf, region_key="region")

        grid, side_sets = exovista.read_exo(path, read_side_sets=True)
        assert side_sets.n_blocks == 2

        # Total reconstructed faces equals the input surface cell count, and the
        # face centroids coincide exactly with the original surface cells.
        centroids = []
        for name in side_sets.keys():
            poly = side_sets[name]
            assert "orig_elem_id" in poly.cell_data
            assert "face_id" in poly.cell_data
            assert poly.cell_data["orig_elem_id"].shape[0] == poly.n_cells
            centroids.append(poly.cell_centers().points)
        centroids = np.vstack(centroids)
        assert centroids.shape[0] == surf.n_cells

        surf_centroids = surf.cell_centers().points
        fwd, _ = cKDTree(surf_centroids).query(centroids)
        rev, _ = cKDTree(centroids).query(surf_centroids)
        assert fwd.max() < 1e-10
        assert rev.max() < 1e-10

    def test_side_set_reconstruction_2d_lines(self, tmp_path):
        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        ], dtype=float)
        volume = pv.UnstructuredGrid(
            np.array([4, 0, 1, 2, 3]),
            np.array([pv.CellType.QUAD], dtype=np.uint8),
            points,
        )
        volume.cell_data["region"] = np.array([1])
        surface = pv.PolyData()
        surface.points = points
        surface.lines = np.array([2, 1, 2])  # edge between nodes 1 and 2
        surface.cell_data["region"] = np.ones(surface.n_cells, dtype=int)

        path = os.path.join(tmp_path, "ss2d.exo")
        exovista.write_exo(path, volume, surface=surface, region_key="region")

        _, side_sets = exovista.read_exo(path, read_side_sets=True)
        assert side_sets.n_blocks == 1
        poly = side_sets[side_sets.keys()[0]]
        assert poly.n_cells == 1
        # The edge is a 2-node line cell.
        assert poly.get_cell(0).type == pv.CellType.LINE
        assert set(poly.get_cell(0).point_ids) == {1, 2}

    def test_read_side_sets_default_returns_grid_only(self, tmp_path):
        vol = _hex_volume()
        path = os.path.join(tmp_path, "noss.exo")
        exovista.write_exo(path, vol, region_key="region")
        result = exovista.read_exo(path)
        assert isinstance(result, pv.UnstructuredGrid)
