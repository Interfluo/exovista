"""Tests for exovista.write_exo: cells, blocks, regions, names, inputs, options.

Files are checked with VTK's own ExodusII reader (``pv.ExodusIIReader``, the
independent oracle) and with ``exovista._backend.open_reader`` / raw netCDF4.
Coordinates are multiples of 1/64 so they survive VTK's float32 points exactly.
"""

import inspect
import logging
import os

import netCDF4
import numpy as np
import pytest
import pyvista as pv
from pyvista.examples import cells as example_cells

from exovista import _celltypes as ct
from exovista._backend import open_reader
from exovista._celltypes import UnsupportedCellTypeError
from exovista.write_pyvista import write_exo

CT = pv.CellType

# VTK type -> (Exodus type in a num_dim=2 file, in a num_dim=3 file). Spelled
# out here, independently of exovista._celltypes. The linear tetrahedron, wedge
# and pyramid are written as TETRA / WEDGE / PYRAMID and triangular shells as
# TRI3 / TRI6 (which meshio can read; every Exodus reader takes the first three
# letters plus the node count, and a TRI in a 3D file is a triangular shell).
EXPECTED_EXO_TYPE = {
    CT.VERTEX: ("SPHERE", "SPHERE"),
    CT.LINE: ("BAR2", "BAR2"),
    CT.QUADRATIC_EDGE: ("BAR3", "BAR3"),
    CT.TRIANGLE: ("TRI3", "TRI3"),
    CT.QUADRATIC_TRIANGLE: ("TRI6", "TRI6"),
    CT.QUAD: ("QUAD4", "SHELL4"),
    CT.PIXEL: ("QUAD4", "SHELL4"),
    CT.QUADRATIC_QUAD: ("QUAD8", "SHELL8"),
    CT.BIQUADRATIC_QUAD: ("QUAD9", "QUAD9"),  # VTK misreads SHELL9
    CT.TETRA: (None, "TETRA"),
    CT.QUADRATIC_TETRA: (None, "TETRA10"),
    CT.HEXAHEDRON: (None, "HEX8"),
    CT.VOXEL: (None, "HEX8"),
    CT.QUADRATIC_HEXAHEDRON: (None, "HEX20"),
    CT.TRIQUADRATIC_HEXAHEDRON: (None, "HEX27"),
    CT.WEDGE: (None, "WEDGE"),
    CT.QUADRATIC_WEDGE: (None, "WEDGE15"),
    CT.PYRAMID: (None, "PYRAMID"),
    CT.QUADRATIC_PYRAMID: (None, "PYRAMID13"),
}
_EXAMPLE = {
    CT.VERTEX: "Vertex",
    CT.LINE: "Line",
    CT.QUADRATIC_EDGE: "QuadraticEdge",
    CT.TRIANGLE: "Triangle",
    CT.QUADRATIC_TRIANGLE: "QuadraticTriangle",
    CT.QUAD: "Quadrilateral",
    CT.PIXEL: "Pixel",
    CT.QUADRATIC_QUAD: "QuadraticQuadrilateral",
    CT.BIQUADRATIC_QUAD: "BiQuadraticQuadrilateral",
    CT.TETRA: "Tetrahedron",
    CT.QUADRATIC_TETRA: "QuadraticTetrahedron",
    CT.HEXAHEDRON: "Hexahedron",
    CT.VOXEL: "Voxel",
    CT.QUADRATIC_HEXAHEDRON: "QuadraticHexahedron",
    CT.TRIQUADRATIC_HEXAHEDRON: "TriQuadraticHexahedron",
    CT.WEDGE: "Wedge",
    CT.QUADRATIC_WEDGE: "QuadraticWedge",
    CT.PYRAMID: "Pyramid",
    CT.QUADRATIC_PYRAMID: "QuadraticPyramid",
}
_SOLID = {t for t, (planar, _) in EXPECTED_EXO_TYPE.items() if planar is None}
# Axis-aligned VTK cells read back as their native cell with these corners.
_NATIVE = {
    CT.VOXEL: (CT.HEXAHEDRON, [0, 1, 3, 2, 4, 5, 7, 6]),
    CT.PIXEL: (CT.QUAD, [0, 1, 3, 2]),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def vtk_read(path):
    """Read ``path`` with VTK's reader: all arrays, sets and global element ids."""
    reader = pv.ExodusIIReader(str(path))
    reader.enable_all_point_arrays()
    reader.enable_all_cell_arrays()
    reader.side_sets.enable_all()
    reader.node_sets.enable_all()
    reader.reader.GenerateGlobalElementIdArrayOn()
    return reader.read()


def extract_surface(mesh, **kwargs):
    """``mesh.extract_surface`` without pyvista's FutureWarning on ``algorithm``."""
    if "algorithm" in inspect.signature(mesh.extract_surface).parameters:
        kwargs.setdefault("algorithm", "dataset_surface")
    return mesh.extract_surface(**kwargs)


def file_blocks(path):
    with open_reader(path) as reader:
        return [(b.id, b.name, b.elem_type, b.n_elem) for b in reader.blocks()]


def cell_points(grid, i):
    return np.asarray(grid.points)[grid.get_cell(i).point_ids]


def assert_cells_match(path, grid):
    """Every VTK-read element is the input cell its global element id names."""
    seen = []
    for block in vtk_read(path)["Element Blocks"]:
        ids = np.asarray(block.cell_data["GlobalElementId"]) - 1
        for j, orig in enumerate(ids):
            in_type = grid.celltypes[orig]
            expected_type, order = _NATIVE.get(in_type, (in_type, None))
            expected = cell_points(grid, orig)
            if order is not None:
                expected = expected[order]
            assert block.celltypes[j] == expected_type
            np.testing.assert_array_equal(cell_points(block, j), expected)
        seen.extend(ids.tolist())
    assert sorted(seen) == list(range(grid.n_cells))


def make_grid(cells, points):
    """UnstructuredGrid from {cell type: connectivity lists}."""
    return pv.UnstructuredGrid(
        {t: np.asarray(c, dtype=np.int64) for t, c in cells.items()},
        np.asarray(points, dtype=float),
    )


def hex_grid(nx=2, ny=2, nz=2, origin=(0.0, 0.0, 0.0)):
    """Structured hexahedra (VTK_HEXAHEDRON) with unit spacing."""
    x = origin[0] + np.arange(nx + 1, dtype=float)
    y = origin[1] + np.arange(ny + 1, dtype=float)
    z = origin[2] + np.arange(nz + 1, dtype=float)
    grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing="ij"))
    return grid.cast_to_unstructured_grid()


def quad_grid(nx=3, ny=2, z=0.0):
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, 1), origin=(0, 0, z))
    grid = grid.cast_to_unstructured_grid()  # pixels
    return pv.UnstructuredGrid(
        {CT.QUAD: grid.cell_connectivity.reshape(-1, 4)[:, [0, 1, 3, 2]]},
        np.asarray(grid.points),
    )


def mixed_grid():
    """Hex, tet, wedge and pyramid cells (in that order), unit sized."""
    points = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1], [0.5, 0.5, -1],
        ],
        dtype=float,
    )
    cells = {
        CT.HEXAHEDRON: [[0, 1, 2, 3, 4, 5, 6, 7]],
        CT.TETRA: [[1, 8, 9, 10]],
        CT.WEDGE: [[1, 9, 2, 5, 11, 6]],
        CT.PYRAMID: [[0, 3, 2, 1, 12]],
    }
    grid = make_grid(cells, points)
    assert np.all(grid.compute_cell_sizes()["Volume"] > 0)
    return grid


def _reference_points(vtk_type, num_dim):
    cell = getattr(example_cells, _EXAMPLE[vtk_type])()
    pts = np.round(np.asarray(cell.points, dtype=float) * 64) / 64
    n = len(pts)
    assert len(np.unique(pts, axis=0)) == n
    if num_dim == 2:
        pts[:, 2] = 0.0
    elif vtk_type not in _SOLID:
        pts[:, 2] += (np.arange(n) % 3) / 8.0  # genuinely 3D
    return pts


def two_cell_grid(vtk_type, num_dim):
    """Two cells of ``vtk_type`` with shuffled point numbering."""
    pts = _reference_points(vtk_type, num_dim)
    n = len(pts)
    shift = [4.0, 0.0, 0.25 if num_dim == 3 and vtk_type not in _SOLID else 0.0]
    points = np.vstack([pts, pts + shift])
    order = np.random.default_rng(int(vtk_type)).permutation(2 * n)
    conn = np.argsort(order)[np.arange(2 * n).reshape(2, n)]
    return pv.UnstructuredGrid({vtk_type: conn}, points[order])


_CELL_CASES = [
    (t, d) for t in EXPECTED_EXO_TYPE for d in (2, 3)
    if EXPECTED_EXO_TYPE[t][d - 2] is not None
]


# --------------------------------------------------------------------------- #
# cell types
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "vtk_type,num_dim", _CELL_CASES,
    ids=[f"{t.name}-{d}d" for t, d in _CELL_CASES],
)
def test_cell_type_written_and_read_by_vtk(tmp_path, vtk_type, num_dim):
    grid = two_cell_grid(vtk_type, num_dim)
    path = tmp_path / "cells.exo"
    write_exo(path, grid)  # num_dim auto-detected from the geometry

    with open_reader(path) as reader:
        assert reader.num_dim == num_dim
        (block,) = reader.blocks()
        assert reader.coords().shape == (grid.n_points, num_dim)
    assert block.elem_type == EXPECTED_EXO_TYPE[vtk_type][num_dim - 2]
    assert block.n_elem == 2

    assert_cells_match(path, grid)
    vtk_grid = vtk_read(path)["Element Blocks"][0]
    if vtk_type in _SOLID or num_dim == 2 and vtk_type not in (
        CT.VERTEX, CT.LINE, CT.QUADRATIC_EDGE
    ):
        size = "Volume" if vtk_type in _SOLID else "Area"
        expected = grid.compute_cell_sizes()[size]
        assert np.all(expected > 0)
        np.testing.assert_allclose(vtk_grid.compute_cell_sizes()[size], expected)


def test_voxels_and_hexahedra_share_a_block(tmp_path):
    voxel = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    hexa = hex_grid(1, 1, 1, origin=(1.0, 0.0, 0.0))
    grid = pv.merge([hexa, voxel, hexa.translate((1, 0, 0))], merge_points=False)
    assert list(grid.celltypes) == [CT.HEXAHEDRON, CT.VOXEL, CT.HEXAHEDRON]
    write_exo(tmp_path / "hv.exo", grid)
    assert file_blocks(tmp_path / "hv.exo") == [(1, "block_1", "HEX8", 3)]
    assert_cells_match(tmp_path / "hv.exo", grid)

    pixels = pv.ImageData(dimensions=(3, 2, 1)).cast_to_unstructured_grid()
    quads = quad_grid(2, 1).translate((0, 1, 0))
    grid2d = pv.merge([pixels, quads], merge_points=True)
    write_exo(tmp_path / "pq.exo", grid2d)
    assert file_blocks(tmp_path / "pq.exo") == [(1, "block_1", "QUAD4", 4)]
    assert_cells_match(tmp_path / "pq.exo", grid2d)


@pytest.mark.parametrize(
    "example",
    [
        "Polygon", "Polyhedron", "PolyLine", "PolyVertex", "TriangleStrip",
        "LagrangeHexahedron", "BiQuadraticTriangle", "QuadraticLinearWedge",
        "TriQuadraticPyramid", "BiQuadraticQuadraticHexahedron",
        "QuadraticPolygon", "ConvexPointSet", "HexagonalPrism", "CubicLine",
        "BezierTriangle", "QuadraticLinearQuadrilateral",
    ],
)
def test_unsupported_cells_raise_and_leave_no_file(tmp_path, example):
    bad = getattr(example_cells, example)()
    grid = pv.merge(
        [hex_grid(1, 1, 1, origin=(5, 5, 5)), bad, bad.translate((9, 0, 0))]
    )
    type_name = pv.CellType(bad.celltypes[0]).name

    path = tmp_path / "bad.exo"
    with pytest.raises(ValueError, match=rf"{type_name}: 2") as err:
        write_exo(path, grid)
    assert isinstance(err.value, UnsupportedCellTypeError)
    assert not path.exists()
    assert os.listdir(tmp_path) == []

    # an existing file is left untouched
    path.write_bytes(b"precious")
    with pytest.raises(ValueError):
        write_exo(path, grid)
    assert path.read_bytes() == b"precious"
    assert os.listdir(tmp_path) == ["bad.exo"]


def test_unsupported_error_suggests_fixes(tmp_path):
    with pytest.raises(ValueError, match=r"triangulate\(\).*extract_cells_by_type"):
        write_exo(tmp_path / "p.exo", example_cells.Polygon())


# --------------------------------------------------------------------------- #
# num_dim
# --------------------------------------------------------------------------- #
def _num_dim_and_types(path):
    with open_reader(path) as reader:
        return reader.num_dim, [b.elem_type for b in reader.blocks()]


def test_num_dim_auto_flat_mesh_is_2d(tmp_path):
    write_exo(tmp_path / "a.exo", quad_grid())
    assert _num_dim_and_types(tmp_path / "a.exo") == (2, ["QUAD4"])
    # a constant z other than 0 is dropped
    write_exo(tmp_path / "b.exo", quad_grid(z=5.0))
    assert _num_dim_and_types(tmp_path / "b.exo") == (2, ["QUAD4"])
    grid = vtk_read(tmp_path / "b.exo")["Element Blocks"][0]
    np.testing.assert_array_equal(np.asarray(grid.points)[:, 2], 0.0)
    # tiny z noise relative to the bounding box still counts as flat
    noisy = quad_grid()
    noisy.points[:, 2] = 1e-14 * np.random.default_rng(0).random(noisy.n_points)
    write_exo(tmp_path / "c.exo", noisy)
    assert _num_dim_and_types(tmp_path / "c.exo") == (2, ["QUAD4"])


def test_num_dim_auto_3d(tmp_path):
    bent = quad_grid()
    bent.points[:, 2] = bent.points[:, 0] ** 2 / 8
    write_exo(tmp_path / "s.exo", bent)
    assert _num_dim_and_types(tmp_path / "s.exo") == (3, ["SHELL4"])
    write_exo(tmp_path / "h.exo", hex_grid())
    assert _num_dim_and_types(tmp_path / "h.exo") == (3, ["HEX8"])
    # a flat mesh with 3D cells (never happens geometrically, but a 3D cell
    # type alone decides): a degenerate hex layer
    flat_hex = hex_grid(1, 1, 1)
    flat_hex.points[:, 2] = 0.0
    write_exo(tmp_path / "fh.exo", flat_hex)
    assert _num_dim_and_types(tmp_path / "fh.exo")[0] == 3


def test_num_dim_forced(tmp_path):
    write_exo(tmp_path / "f3.exo", quad_grid(), num_dim=3)
    assert _num_dim_and_types(tmp_path / "f3.exo") == (3, ["SHELL4"])
    assert_cells_match(tmp_path / "f3.exo", quad_grid())

    write_exo(tmp_path / "f2.exo", quad_grid(), num_dim=2)
    assert _num_dim_and_types(tmp_path / "f2.exo") == (2, ["QUAD4"])

    with pytest.raises(ValueError, match="3D cells"):
        write_exo(tmp_path / "x.exo", hex_grid(), num_dim=2)
    bent = quad_grid()
    bent.points[:, 2] = bent.points[:, 0]
    with pytest.raises(ValueError, match="z coordinate"):
        write_exo(tmp_path / "x.exo", bent, num_dim=2)
    for bad in (1, 4, True, "3", 2.5):
        with pytest.raises(ValueError, match="num_dim"):
            write_exo(tmp_path / "x.exo", quad_grid(), num_dim=bad)
    assert not (tmp_path / "x.exo").exists()


def test_lines_and_vertices_follow_geometry(tmp_path):
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0]], dtype=float)
    flat = make_grid({CT.LINE: [[0, 1], [1, 2]], CT.VERTEX: [[3]]}, points)
    write_exo(tmp_path / "l2.exo", flat)
    assert _num_dim_and_types(tmp_path / "l2.exo") == (2, ["BAR2", "SPHERE"])
    assert_cells_match(tmp_path / "l2.exo", flat)

    lifted = flat.copy()
    lifted.points[:, 2] = [0, 0.5, 1, 1.5]
    write_exo(tmp_path / "l3.exo", lifted)
    assert _num_dim_and_types(tmp_path / "l3.exo") == (3, ["BAR2", "SPHERE"])
    assert_cells_match(tmp_path / "l3.exo", lifted)


def test_2d_cells_in_3d_are_shells(tmp_path):
    cells = [two_cell_grid(t, 3) for t in (
        CT.TRIANGLE, CT.QUADRATIC_TRIANGLE, CT.QUAD, CT.QUADRATIC_QUAD,
        CT.BIQUADRATIC_QUAD, CT.PIXEL,
    )]
    grid = pv.merge(
        [c.translate((0, 3 * i, 0)) for i, c in enumerate(cells)], merge_points=False
    )
    grid.cell_data["region"] = np.repeat(np.arange(1, 7), 2)
    write_exo(tmp_path / "shells.exo", grid)
    num_dim, types = _num_dim_and_types(tmp_path / "shells.exo")
    assert num_dim == 3
    # TRI3 / TRI6 in a 3D file are triangular shells (Exodus library, VTK)
    assert types == ["TRI3", "TRI6", "SHELL4", "SHELL8", "QUAD9", "SHELL4"]
    with open_reader(tmp_path / "shells.exo") as r:
        resolved = [ct.resolve_exo_type(b.elem_type, b.n_nodes_per_elem, r.num_dim)
                    for b in r.blocks()]
    assert [t.name for t in resolved[:2]] == ["TRISHELL3", "TRISHELL6"]
    assert_cells_match(tmp_path / "shells.exo", grid)


def test_shells_and_solids_together(tmp_path):
    solid = hex_grid(1, 1, 1)
    skin = extract_surface(solid, pass_pointid=False, pass_cellid=False)
    grid = pv.merge([solid, skin.cast_to_unstructured_grid()], merge_points=True)
    write_exo(tmp_path / "ss.exo", grid)
    # one region with two element types: sequential ids, default names
    assert file_blocks(tmp_path / "ss.exo") == [
        (1, "block_1", "HEX8", 1), (2, "block_2", "SHELL4", 6)
    ]
    assert_cells_match(tmp_path / "ss.exo", grid)


# --------------------------------------------------------------------------- #
# input types
# --------------------------------------------------------------------------- #
def test_multiblock_leaves_become_regions(tmp_path):
    left = pv.ImageData(dimensions=(3, 3, 3))
    right = hex_grid(2, 2, 2, origin=(2.0, 0.0, 0.0))
    mb = pv.MultiBlock({"left": left, "right": right})
    path = tmp_path / "mb.exo"
    write_exo(path, mb)
    assert file_blocks(path) == [(1, "left", "HEX8", 8), (2, "right", "HEX8", 8)]
    with open_reader(path) as reader:
        assert reader.num_nodes == 27 + 27 - 9  # shared face merged
        np.testing.assert_array_equal(reader.elem_id_map(), np.arange(1, 17))
    assert vtk_read(path)["Element Blocks"].keys() == ["left", "right"]
    combined = mb.combine(merge_points=True)
    assert_cells_match(path, combined)


def test_multiblock_nested_empty_and_duplicate_names(tmp_path):
    a = hex_grid(1, 1, 1)
    b = hex_grid(1, 1, 1, origin=(3, 0, 0))
    c = pv.ImageData(dimensions=(2, 2, 2), origin=(6, 0, 0))
    mb = pv.MultiBlock(
        {
            "x": a,
            "nothing": None,
            "sub": pv.MultiBlock({"x": b, "empty": pv.UnstructuredGrid()}),
            "c": c,
        }
    )
    write_exo(tmp_path / "n.exo", mb)
    assert file_blocks(tmp_path / "n.exo") == [
        (1, "x", "HEX8", 1), (2, "x_2", "HEX8", 1), (3, "c", "HEX8", 1)
    ]


def test_multiblock_region_arrays_ignored_and_types_split(tmp_path):
    hexes = hex_grid(2, 1, 1)
    hexes.cell_data["region"] = [7, 8]  # each leaf is one region regardless
    mixed = mixed_grid().translate((10, 0, 0))
    mb = pv.MultiBlock({"hexes": hexes, "mixed": mixed})
    with pytest.warns(UserWarning, match="ignored the 'region' cell array"):
        write_exo(tmp_path / "m.exo", mb, block_names={1: "H"})
    assert file_blocks(tmp_path / "m.exo") == [
        (1, "H", "HEX8", 2),
        (2, "mixed_hex8", "HEX8", 1),
        (3, "mixed_pyramid5", "PYRAMID", 1),
        (4, "mixed_tetra4", "TETRA", 1),
        (5, "mixed_wedge6", "WEDGE", 1),
    ]


def test_multiblock_single_leaf_and_all_empty(tmp_path):
    write_exo(tmp_path / "one.exo", pv.MultiBlock({"only": hex_grid()}))
    assert file_blocks(tmp_path / "one.exo") == [(1, "only", "HEX8", 8)]
    for empty in (pv.MultiBlock(), pv.MultiBlock({"e": pv.UnstructuredGrid()})):
        with pytest.raises(ValueError, match="empty"):
            write_exo(tmp_path / "e.exo", empty)
    assert not (tmp_path / "e.exo").exists()


@pytest.mark.parametrize(
    "make,n_cells,types",
    [
        (lambda: pv.ImageData(dimensions=(3, 4, 2)), 6, ["HEX8"]),
        (lambda: pv.ImageData(dimensions=(3, 4, 1)), 6, ["QUAD4"]),
        (lambda: pv.RectilinearGrid([0, 1, 3], [0, 2], [0, 1]), 2, ["HEX8"]),
        (
            lambda: pv.StructuredGrid(
                *np.meshgrid([0.0, 1, 2], [0.0, 1], [0.0, 1], indexing="ij")
            ),
            2,
            ["HEX8"],
        ),
        (lambda: pv.Plane(i_resolution=2, j_resolution=3), 6, ["QUAD4"]),
        (lambda: pv.Sphere(theta_resolution=6, phi_resolution=6), 48, ["TRI3"]),
    ],
    ids=["ImageData3D", "ImageData2D", "RectilinearGrid", "StructuredGrid",
         "PlanePolyData", "SpherePolyData"],
)
def test_non_unstructured_inputs(tmp_path, make, n_cells, types):
    mesh = make()
    path = tmp_path / "in.exo"
    write_exo(path, mesh)
    assert [t for _, _, t, _ in file_blocks(path)] == types
    blocks = vtk_read(path)["Element Blocks"]
    assert sum(b.n_cells for b in blocks) == n_cells
    assert_cells_match(path, mesh.cast_to_unstructured_grid())


def test_vtk_object_input(tmp_path):
    vtk_grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    write_exo(tmp_path / "v.exo", vtk_grid.copy())  # plain pyvista
    raw = pv.UnstructuredGrid(vtk_grid)  # still pyvista; now a bare VTK object:
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

    bare = vtkUnstructuredGrid()
    bare.DeepCopy(raw)
    write_exo(tmp_path / "w.exo", bare)
    assert file_blocks(tmp_path / "w.exo") == [(1, "block_1", "HEX8", 1)]


@pytest.mark.parametrize("bad", [np.zeros((4, 3)), "mesh.vtk", 42, None])
def test_bad_volume_type(tmp_path, bad):
    with pytest.raises(TypeError, match="volume"):
        write_exo(tmp_path / "x.exo", bad)


@pytest.mark.parametrize(
    "empty",
    [
        lambda: pv.UnstructuredGrid(),
        lambda: _points_without_cells(pv.PolyData()),
        lambda: pv.PolyData(),
        lambda: _points_without_cells(pv.UnstructuredGrid()),
    ],
    ids=["UnstructuredGrid", "PointsWithoutCells", "PolyData", "UGPointsOnly"],
)
def test_empty_mesh_raises(tmp_path, empty):
    with pytest.raises(ValueError, match="empty mesh"):
        write_exo(tmp_path / "e.exo", empty())
    assert not (tmp_path / "e.exo").exists()


def _points_without_cells(mesh):
    mesh.points = np.random.default_rng(0).random((5, 3))
    return mesh


def test_point_cloud_is_written_as_spheres(tmp_path):
    cloud = pv.PolyData(np.arange(15, dtype=float).reshape(5, 3) / 4)
    write_exo(tmp_path / "pc.exo", cloud)
    assert file_blocks(tmp_path / "pc.exo") == [(1, "block_1", "SPHERE", 5)]
    assert_cells_match(tmp_path / "pc.exo", cloud.cast_to_unstructured_grid())


def test_inputs_are_never_mutated(tmp_path):
    volume = mixed_grid()
    volume.cell_data["region"] = [2, 1, 2, 1]
    volume.point_data["T"] = np.arange(volume.n_points, dtype=float)
    volume.point_data["U"] = np.asarray(volume.points) * 2
    volume.cell_data["S"] = ["a", "b", "c", "d"]
    volume.field_data["meta"] = [1, 2, 3]
    surface = extract_surface(volume)
    surface.cell_data["region"] = np.arange(surface.n_cells) % 2 + 1

    vol_before, surf_before = volume.copy(deep=True), surface.copy(deep=True)
    keys = (
        list(volume.point_data), list(volume.cell_data), list(volume.field_data),
        list(surface.point_data), list(surface.cell_data),
    )
    with pytest.warns(UserWarning, match="non-numeric"):
        write_exo(
            tmp_path / "m.exo", volume, surface,
            node_sets={"a": [0, 1]},
            element_fields={"E": np.ones((1, volume.n_cells))},
        )
    assert volume == vol_before
    assert surface == surf_before
    assert keys == (
        list(volume.point_data), list(volume.cell_data), list(volume.field_data),
        list(surface.point_data), list(surface.cell_data),
    )

    leaf = hex_grid(1, 1, 1)
    leaf_before = leaf.copy(deep=True)
    mb = pv.MultiBlock({"a": leaf, "b": hex_grid(1, 1, 1, origin=(1, 0, 0))})
    write_exo(tmp_path / "mb.exo", mb)
    assert leaf == leaf_before
    assert list(leaf.cell_data) == list(leaf_before.cell_data)
    assert mb.keys() == ["a", "b"] and mb["a"] is leaf


# --------------------------------------------------------------------------- #
# regions, block ids and names
# --------------------------------------------------------------------------- #
def interleaved_hexes(regions):
    grid = hex_grid(len(regions), 1, 1)
    grid.cell_data["region"] = regions
    return grid


def test_region_values_are_block_ids(tmp_path):
    grid = interleaved_hexes([30, 7, 30, 7, 12])
    write_exo(tmp_path / "r.exo", grid)
    assert file_blocks(tmp_path / "r.exo") == [
        (7, "block_7", "HEX8", 2), (12, "block_12", "HEX8", 1),
        (30, "block_30", "HEX8", 2),
    ]
    mb = vtk_read(tmp_path / "r.exo")["Element Blocks"]
    assert mb.keys() == ["block_7", "block_12", "block_30"]
    assert [int(b.cell_data["ObjectId"][0]) for b in mb] == [7, 12, 30]
    assert_cells_match(tmp_path / "r.exo", grid)


def test_region_key_and_missing_region(tmp_path):
    grid = interleaved_hexes([1, 2, 1])
    grid.cell_data["mat"] = [5, 5, 6]
    write_exo(tmp_path / "k.exo", grid, region_key="mat")
    assert [b[0] for b in file_blocks(tmp_path / "k.exo")] == [5, 6]
    write_exo(tmp_path / "none.exo", grid, region_key="nope")
    assert file_blocks(tmp_path / "none.exo") == [(1, "block_1", "HEX8", 3)]
    write_exo(tmp_path / "none2.exo", grid, region_key=None)
    assert file_blocks(tmp_path / "none2.exo") == [(1, "block_1", "HEX8", 3)]


@pytest.mark.parametrize(
    "regions,expected",
    [
        ([0, 1, 0], [(1, "block_1", 2), (2, "block_2", 1)]),  # 0 -> sequential
        ([-5, 3, 3], [(1, "block_1", 1), (2, "block_2", 2)]),
        ([1.0, 4.0, 1.0], [(1, "block_1", 2), (4, "block_4", 1)]),  # integral
        ([0.5, 1.5, 0.5], [(1, "block_1", 2), (2, "block_2", 1)]),
        ([True, False, True], [(1, "block_1", 1), (2, "block_2", 2)]),
        ([2**31, 1, 1], [(1, "block_1", 2), (2, "block_2", 1)]),  # > int32
    ],
)
def test_region_id_rule_falls_back_to_sequential(tmp_path, regions, expected):
    write_exo(tmp_path / "s.exo", interleaved_hexes(regions))
    assert [(i, n, k) for i, n, _, k in file_blocks(tmp_path / "s.exo")] == expected


def test_string_regions(tmp_path):
    grid = interleaved_hexes(["steel", "copper", "steel", "air"])
    write_exo(tmp_path / "s.exo", grid)
    assert file_blocks(tmp_path / "s.exo") == [
        (1, "air", "HEX8", 1), (2, "copper", "HEX8", 1), (3, "steel", "HEX8", 2)
    ]
    assert vtk_read(tmp_path / "s.exo")["Element Blocks"].keys() == [
        "air", "copper", "steel"
    ]
    assert_cells_match(tmp_path / "s.exo", grid)
    # dict keys are the string region values
    write_exo(tmp_path / "d.exo", grid, block_names={"air": "Air"})
    assert [b[1] for b in file_blocks(tmp_path / "d.exo")] == ["Air", "copper", "steel"]


def test_region_spanning_types_gets_sequential_ids_and_suffixes(tmp_path):
    grid = mixed_grid()
    grid.cell_data["region"] = [5, 5, 9, 9]  # hex+tet in 5, wedge+pyramid in 9
    write_exo(tmp_path / "m.exo", grid, block_names={5: "front"})
    assert file_blocks(tmp_path / "m.exo") == [
        (1, "front_hex8", "HEX8", 1),
        (2, "front_tetra4", "TETRA", 1),
        (3, "block_3", "PYRAMID", 1),
        (4, "block_4", "WEDGE", 1),
    ]
    assert_cells_match(tmp_path / "m.exo", grid)


def test_block_names_land_on_their_blocks(tmp_path):
    """1.x named blocks in (cell type, region) order: names hit wrong blocks."""
    tris = pv.Plane(i_resolution=2, j_resolution=2).triangulate()
    quads = pv.Plane(i_resolution=2, j_resolution=2, center=(2, 0, 0))
    grid = pv.merge([quads, tris], merge_points=False)
    grid.cell_data["region"] = [0] * quads.n_cells + [1] * tris.n_cells
    write_exo(tmp_path / "n.exo", grid, block_names=["quads", "tris"])
    assert file_blocks(tmp_path / "n.exo") == [
        (1, "quads", "QUAD4", 4), (2, "tris", "TRI3", 8)
    ]
    mb = vtk_read(tmp_path / "n.exo")["Element Blocks"]
    assert set(mb["quads"].celltypes) == {CT.QUAD}
    assert set(mb["tris"].celltypes) == {CT.TRIANGLE}


def test_block_names_list_dict_and_str(tmp_path):
    grid = interleaved_hexes([2, 1, 2])
    write_exo(tmp_path / "l.exo", grid, block_names=["one", "two"])
    assert [b[:2] for b in file_blocks(tmp_path / "l.exo")] == [(1, "one"), (2, "two")]
    write_exo(tmp_path / "d.exo", grid, block_names={np.int64(2): "two"})
    assert [b[:2] for b in file_blocks(tmp_path / "d.exo")] == [
        (1, "block_1"), (2, "two")
    ]
    write_exo(tmp_path / "s.exo", hex_grid(), block_names="body")
    assert file_blocks(tmp_path / "s.exo") == [(1, "body", "HEX8", 8)]
    long_name = "a_rather_long_block_name_" * 4  # > 32 chars survives
    write_exo(tmp_path / "long.exo", hex_grid(), block_names=[long_name])
    assert file_blocks(tmp_path / "long.exo")[0][1] == long_name
    assert vtk_read(tmp_path / "long.exo")["Element Blocks"].keys() == [long_name]


@pytest.mark.parametrize(
    "names,error,match",
    [
        (["only_one"], ValueError, "1 name"),
        (["a", "b", "c"], ValueError, "3 name"),
        ({3: "x"}, ValueError, "not a region value"),
        (["same", "same"], ValueError, "unique"),
        ({1: "block_2"}, ValueError, "unique"),  # clashes with a default
        (["x" * 257, "y"], ValueError, "256"),
        ("single", TypeError, "single string"),
        (5, TypeError, "list, a dict or a str"),
    ],
)
def test_block_name_errors(tmp_path, names, error, match):
    with pytest.raises(error, match=match):
        write_exo(tmp_path / "x.exo", interleaved_hexes([1, 2]), block_names=names)
    assert not (tmp_path / "x.exo").exists()


def test_element_id_map_is_original_cell_index(tmp_path):
    regions = [3, 1, 2, 1, 3, 2, 1]
    grid = interleaved_hexes(regions)
    path = tmp_path / "ids.exo"
    write_exo(path, grid)
    expected = np.argsort(regions, kind="stable") + 1
    with netCDF4.Dataset(path) as nc:
        np.testing.assert_array_equal(nc.variables["elem_num_map"][:], expected)
    ids = np.concatenate(
        [b.cell_data["GlobalElementId"] for b in vtk_read(path)["Element Blocks"]]
    )
    np.testing.assert_array_equal(ids, expected)
    assert_cells_match(path, grid)


# --------------------------------------------------------------------------- #
# node sets
# --------------------------------------------------------------------------- #
def _node_sets(path):
    with open_reader(path) as reader:
        return [(s.id, s.name, s.nodes.tolist()) for s in reader.node_sets()]


def test_node_sets(tmp_path):
    grid = hex_grid(2, 1, 1)
    pts = np.asarray(grid.points)
    left = np.flatnonzero(pts[:, 0] == 0)
    top = pts[:, 2] == 1  # boolean mask
    path = tmp_path / "ns.exo"
    write_exo(path, grid, node_sets={"left": left, "top": top, "one": [5]})
    assert _node_sets(path) == [
        (1, "left", left.tolist()), (2, "top", np.flatnonzero(top).tolist()),
        (3, "one", [5]),
    ]
    sets = vtk_read(path)["Node Sets"]
    assert sets.keys() == ["left", "top", "one"]
    for name, nodes in [("left", left), ("top", np.flatnonzero(top))]:
        got = np.asarray(sets[name].points)
        assert sorted(map(tuple, got)) == sorted(map(tuple, pts[nodes]))

    write_exo(path, grid, node_sets={10: [0, 1], 4: [2]})
    assert _node_sets(path) == [(10, "node_set_10", [0, 1]), (4, "node_set_4", [2])]
    write_exo(path, grid, node_sets=[[0], np.array([3, 1])])
    assert _node_sets(path) == [(1, "node_set_1", [0]), (2, "node_set_2", [3, 1])]


@pytest.mark.parametrize(
    "node_sets,error,match",
    [
        ({"a": [0, 12]}, ValueError, "0..11"),
        ({"a": [-1]}, ValueError, "0..11"),
        ({"a": [1, 2, 1]}, ValueError, "repeated"),
        ({"a": np.ones(5, bool)}, ValueError, "boolean mask"),
        ({"a": [0.5]}, TypeError, "integers"),
        ("abc", TypeError, "node_sets"),
    ],
)
def test_node_set_errors(tmp_path, node_sets, error, match):
    with pytest.raises(error, match=match):
        write_exo(tmp_path / "x.exo", hex_grid(2, 1, 1), node_sets=node_sets)
    assert not (tmp_path / "x.exo").exists()


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
def test_precision(tmp_path):
    grid = hex_grid(2, 1, 1, origin=(1e6 + 0.123456789, 0, 0))
    grid.point_data["T"] = 300.000123456789 + np.arange(grid.n_points)
    grid.point_data["I"] = 16777217 + np.arange(grid.n_points)  # > 2**24
    write_exo(tmp_path / "d.exo", grid, times=[1000.000000001])
    with pytest.warns(UserWarning, match=r"'I' holds integers up to 16777228"):
        write_exo(tmp_path / "s.exo", grid, precision="single")
    with netCDF4.Dataset(tmp_path / "d.exo") as nc:
        assert nc.variables["coordx"].dtype == np.float64
        assert nc.variables["vals_nod_var1"].dtype == np.float64
        assert nc.variables["time_whole"].dtype == np.float64
    with netCDF4.Dataset(tmp_path / "s.exo") as nc:
        assert nc.variables["coordx"].dtype == np.float32
        assert nc.variables["vals_nod_var1"].dtype == np.float32
    with open_reader(tmp_path / "d.exo") as reader:
        np.testing.assert_array_equal(reader.coords(), np.asarray(grid.points))
        np.testing.assert_array_equal(reader.node_var("T", 0), grid.point_data["T"])
        np.testing.assert_array_equal(reader.node_var("I", 0), grid.point_data["I"])
        assert reader.times()[0] == 1000.000000001
    for bad in ("half", 2, None):
        with pytest.raises(ValueError, match="precision"):
            write_exo(tmp_path / "x.exo", grid, precision=bad)


def test_compression_levels(tmp_path):
    grid = hex_grid(4, 4, 4)
    grid.point_data["T"] = np.zeros(grid.n_points)
    for level in (0, 1, 9):
        write_exo(tmp_path / f"c{level}.exo", grid, compression_level=level)
        with netCDF4.Dataset(tmp_path / f"c{level}.exo") as nc:
            filters = nc.variables["connect1"].filters()
            assert filters["zlib"] is (level > 0)
            if level:
                assert filters["complevel"] == level
    write_exo(tmp_path / "default.exo", grid)
    with netCDF4.Dataset(tmp_path / "default.exo") as nc:
        assert nc.variables["connect1"].filters()["zlib"]
    for bad, error in ((10, ValueError), (-1, ValueError), (1.5, TypeError),
                       (True, TypeError)):
        with pytest.raises(error, match="compression_level"):
            write_exo(tmp_path / "x.exo", grid, compression_level=bad)


def test_title(tmp_path):
    write_exo(tmp_path / "t.exo", hex_grid(), title="my model")
    write_exo(tmp_path / "d.exo", hex_grid())
    with netCDF4.Dataset(tmp_path / "t.exo") as nc:
        assert nc.title == "my model"
    with netCDF4.Dataset(tmp_path / "d.exo") as nc:
        assert nc.title == "exovista"


def test_no_root_logger_use(tmp_path, caplog):
    """1.x logged through the root logger (and so configured it)."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    grid = hex_grid()  # no region array: 1.x warned through the root logger
    with caplog.at_level(logging.DEBUG):
        write_exo(tmp_path / "l.exo", grid, extract_surface(grid))
    assert caplog.records  # the module logger is live (not a vacuous pass)
    assert all(r.name.startswith("exovista") for r in caplog.records)
    assert all(r.levelno < logging.WARNING for r in caplog.records)
    assert root.handlers == handlers


def test_overwrite_and_pathlike(tmp_path):
    path = tmp_path / "o.exo"
    write_exo(path, hex_grid(1, 1, 1))
    write_exo(str(path), hex_grid(2, 1, 1))
    assert file_blocks(path) == [(1, "block_1", "HEX8", 2)]
    assert sorted(os.listdir(tmp_path)) == ["o.exo"]  # no temp files


# --------------------------------------------------------------------------- #
# PyVista's built-in meshes (1.x crashed on most of them)
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "make,num_dim,types",
    [
        (lambda: pv.Sphere().delaunay_3d(), 3, ["TETRA"]),
        (lambda: pv.Cube().delaunay_3d(), 3, ["TETRA"]),
        (lambda: pv.Plane(), 2, ["QUAD4"]),
        (lambda: pv.examples.load_hexbeam(), 3, ["HEX8"]),
        (lambda: pv.ImageData(dimensions=(5, 4, 3)), 3, ["HEX8"]),
    ],
    ids=["Sphere.delaunay_3d", "Cube.delaunay_3d", "Plane", "hexbeam", "ImageData"],
)
def test_builtin_meshes(tmp_path, make, num_dim, types):
    mesh = make()
    path = tmp_path / "builtin.exo"
    write_exo(path, mesh)
    assert _num_dim_and_types(path) == (num_dim, types)
    blocks = vtk_read(path)["Element Blocks"]
    assert sum(b.n_cells for b in blocks) == mesh.n_cells
    for name, values in mesh.point_data.items():
        if np.asarray(values).ndim == 2 and values.shape[1] in (2, 3):
            name += "_"  # VTK regroups name_X, name_Y[, name_Z]
        assert name in blocks[0].point_data
    for name in mesh.cell_data:
        assert name in blocks[0].cell_data
    assert_cells_match(path, mesh.cast_to_unstructured_grid())
