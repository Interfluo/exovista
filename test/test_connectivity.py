"""Raw connectivity written by exovista.write_exo, inspected with netCDF4.

VTK's axis-aligned cells (VOXEL, PIXEL) and the quadratic cells whose node
order differs between VTK and Exodus must be permuted on write.
"""

import netCDF4
import numpy as np
import pyvista as pv

from _pvcompat import example_cell
from exovista.write_pyvista import write_exo


def _connect1(path):
    with netCDF4.Dataset(path) as nc:
        return (
            np.asarray(nc.variables["connect1"][:]),
            nc.variables["connect1"].getncattr("elem_type"),
            nc.dimensions["num_dim"].size,
        )


def test_voxel_connectivity(tmp_path):
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
         [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=float,
    )
    grid = pv.UnstructuredGrid(
        np.array([8, 0, 1, 2, 3, 4, 5, 6, 7]), [pv.CellType.VOXEL], points
    )
    write_exo(tmp_path / "voxel.exo", grid)
    conn, elem_type, num_dim = _connect1(tmp_path / "voxel.exo")
    # Exodus HEX8 order is counter-clockwise: 0, 1, 3, 2, 4, 5, 7, 6 (1-based +1)
    np.testing.assert_array_equal(conn, [[1, 2, 4, 3, 5, 6, 8, 7]])
    assert (elem_type, num_dim) == ("HEX8", 3)


def test_pixel_connectivity(tmp_path):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    grid = pv.UnstructuredGrid(np.array([4, 0, 1, 2, 3]), [pv.CellType.PIXEL], points)
    write_exo(tmp_path / "pixel.exo", grid)
    conn, elem_type, num_dim = _connect1(tmp_path / "pixel.exo")
    np.testing.assert_array_equal(conn, [[1, 2, 4, 3]])
    assert (elem_type, num_dim) == ("QUAD4", 2)


def test_hex20_connectivity(tmp_path):
    # Exodus lists the 4 vertical mid-edge nodes before the 4 top ones
    grid = example_cell("QuadraticHexahedron")
    write_exo(tmp_path / "hex20.exo", grid)
    conn, elem_type, _ = _connect1(tmp_path / "hex20.exo")
    vtk_order = np.arange(20)
    expected = np.concatenate([vtk_order[:12], vtk_order[16:20], vtk_order[12:16]])
    np.testing.assert_array_equal(conn, [expected + 1])
    assert elem_type == "HEX20"


def test_element_id_map_and_block_order(tmp_path):
    grid = pv.ImageData(dimensions=(5, 2, 2)).cast_to_unstructured_grid()
    grid.cell_data["region"] = [2, 1, 2, 1]
    write_exo(tmp_path / "order.exo", grid)
    with netCDF4.Dataset(tmp_path / "order.exo") as nc:
        np.testing.assert_array_equal(nc.variables["eb_prop1"][:], [1, 2])
        np.testing.assert_array_equal(nc.variables["elem_num_map"][:], [2, 4, 1, 3])
        assert nc.dimensions["num_elem"].size == 4
        assert nc.dimensions["num_el_in_blk1"].size == 2
