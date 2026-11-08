"""read_exo builds the right VTK cells for every supported Exodus element type.

Files are written with ``exovista._backend.write_model``; VTK's own reader
(``pv.ExodusIIReader``) is the independent oracle. Coordinates are multiples
of 1/64 so they survive VTK's float32 points exactly.
"""

import warnings

import numpy as np
import pytest
import pyvista as pv
from pyvista.examples import cells as example_cells

from exovista import _celltypes as ct
from exovista import read_pyvista as rp
from exovista._backend import write_model
from exovista._model import Block, ExodusModel, SideSet

VTK = ct.VTK

# VTK type -> pyvista example cell with a valid geometry (VTK node order)
_EXAMPLES = {
    VTK.VERTEX: "Vertex",
    VTK.LINE: "Line",
    VTK.QUADRATIC_EDGE: "QuadraticEdge",
    VTK.TRIANGLE: "Triangle",
    VTK.QUADRATIC_TRIANGLE: "QuadraticTriangle",
    VTK.QUAD: "Quadrilateral",
    VTK.QUADRATIC_QUAD: "QuadraticQuadrilateral",
    VTK.BIQUADRATIC_QUAD: "BiQuadraticQuadrilateral",
    VTK.TETRA: "Tetrahedron",
    VTK.QUADRATIC_TETRA: "QuadraticTetrahedron",
    VTK.HEXAHEDRON: "Hexahedron",
    VTK.QUADRATIC_HEXAHEDRON: "QuadraticHexahedron",
    VTK.TRIQUADRATIC_HEXAHEDRON: "TriQuadraticHexahedron",
    VTK.WEDGE: "Wedge",
    VTK.QUADRATIC_WEDGE: "QuadraticWedge",
    VTK.PYRAMID: "Pyramid",
    VTK.QUADRATIC_PYRAMID: "QuadraticPyramid",
}

# (Exodus element type as stored, num_dim of the file)
_CASES = [
    ("SPHERE", 3),
    ("SPHERE", 2),
    ("BAR2", 3),
    ("BAR2", 2),
    ("BAR3", 3),
    ("BEAM2", 3),
    ("BEAM3", 2),
    ("TRI3", 2),
    ("TRI6", 2),
    ("QUAD4", 2),
    ("QUAD8", 2),
    ("QUAD9", 2),
    ("QUAD9", 3),
    ("TRISHELL3", 3),
    ("TRISHELL6", 3),
    ("TRI3", 3),  # a TRI in a 3D file is a TRISHELL
    ("SHELL4", 3),
    ("SHELL8", 3),
    ("TETRA4", 3),
    ("TETRA10", 3),
    ("HEX8", 3),
    ("HEX20", 3),
    ("HEX27", 3),
    ("WEDGE6", 3),
    ("WEDGE15", 3),
    ("PYRAMID5", 3),
    ("PYRAMID13", 3),
    # names other tools write; the node count decides the order
    ("hex", 3),
    ("HEXAHEDRON", 3),
    ("tetra", 3),
    ("TET", 3),
    ("wedge", 3),
    ("pyramid", 3),
    ("quad", 2),
    ("TRIANGLE", 2),
    ("SHELL", 3),
    ("TRUSS2", 3),
    ("EDGE2", 3),
    ("BEAM", 3),
    ("CIRCLE", 2),
]
# aliases resolved with a quadratic node count
_QUADRATIC_ALIASES = [("HEX", 20), ("HEXAHEDRON", 27), ("TETRA", 10),
                      ("WEDGE", 15), ("PYRA", 13), ("TRIANGLE", 6), ("QUAD", 8),
                      ("SHELL", 8), ("TRUSS", 3)]


def _reference_points(vtk_type, num_dim):
    """Points of one valid cell (VTK order) on a 1/64 grid."""
    cell = getattr(example_cells, _EXAMPLES[vtk_type])()
    pts = np.round(np.asarray(cell.points, dtype=float) * 64) / 64
    assert len(np.unique(pts, axis=0)) == len(pts)
    if num_dim == 2:
        pts[:, 2] = 0.0
    return pts


def _two_cells(info, num_dim):
    """Two cells of ``info``'s type with shuffled node numbering.

    Returns (points, vtk_conn): ``points[vtk_conn[i]]`` are the points of cell
    ``i`` in VTK order.
    """
    pts = _reference_points(info.vtk_type, num_dim)
    if num_dim == 3 and info.dim == 2:
        pts[:, 2] = 0.25  # a planar shell off the z = 0 plane
    n = len(pts)
    points = np.vstack([pts, pts + [4.0, 0.0, 0.0]])
    order = np.random.default_rng(n).permutation(2 * n)
    inverse = np.argsort(order)
    conn = inverse[np.arange(2 * n).reshape(2, n)]
    return points[order], conn


def _write_case(path, exo_name, n_nodes, num_dim):
    info = ct.resolve_exo_type(exo_name, n_nodes, num_dim)
    points, vtk_conn = _two_cells(info, num_dim)
    exo_conn = vtk_conn[:, list(info.vtk_to_exo)]
    model = ExodusModel(
        "cells", num_dim, points[:, :num_dim],
        [Block(3, "blk", exo_name, exo_conn)],
    )
    write_model(path, model)
    return info, points, vtk_conn


def _cell_points(mesh, i):
    return np.asarray(mesh.points)[mesh.get_cell(i).point_ids]


def _check_cells(path, info, points, vtk_conn, num_dim, oracle=True):
    grid = rp.read_exo(path)
    assert grid.n_cells == 2
    assert grid.points.dtype == np.float64
    assert list(grid.celltypes) == [info.vtk_type] * 2
    expected_points = points.copy()
    if num_dim == 2:
        expected_points[:, 2] = 0.0
    np.testing.assert_array_equal(grid.points, expected_points)
    for i in range(2):
        assert list(grid.get_cell(i).point_ids) == list(vtk_conn[i])
    report = grid.validate_mesh(
        ["wrong_number_of_points", "invalid_point_references", "negative_size",
         "inverted_faces", "non_finite_points"]
    )
    assert report.is_valid, report.message
    if info.dim > 0:
        size = {1: "Length", 2: "Area", 3: "Volume"}[info.dim]
        sizes = grid.compute_cell_sizes(
            length=size == "Length", area=size == "Area", volume=size == "Volume",
            vertex_count=False,
        )[size]
        source = pv.UnstructuredGrid({info.vtk_type: vtk_conn}, expected_points)
        expected = source.compute_cell_sizes(
            length=size == "Length", area=size == "Area", volume=size == "Volume",
            vertex_count=False,
        )[size]
        assert np.all(expected > 0)
        np.testing.assert_allclose(sizes, expected)
    if oracle:
        vtk_grid = pv.ExodusIIReader(str(path)).read()["Element Blocks"][0]
        assert list(vtk_grid.celltypes) == list(grid.celltypes)
        for i in range(2):
            np.testing.assert_array_equal(
                _cell_points(vtk_grid, i), _cell_points(grid, i)
            )
    return grid


@pytest.mark.parametrize("exo_name,num_dim", _CASES,
                         ids=[f"{n}-{d}d" for n, d in _CASES])
def test_element_type_matches_vtk_reader(tmp_path, exo_name, num_dim):
    path = tmp_path / "cells.exo"
    info, points, vtk_conn = _write_case(path, exo_name, None, num_dim)
    grid = _check_cells(path, info, points, vtk_conn, num_dim)
    assert list(grid.cell_data["exo_block_id"]) == [3, 3]
    assert list(grid.cell_data["exo_block_name"]) == ["blk", "blk"]
    assert list(grid.cell_data["exo_elem_id"]) == [1, 2]
    np.testing.assert_array_equal(grid.field_data["num_dim"], [num_dim])


@pytest.mark.parametrize("exo_name,n_nodes", _QUADRATIC_ALIASES)
def test_generic_names_use_node_count(tmp_path, exo_name, n_nodes):
    num_dim = 2 if exo_name in ("TRIANGLE", "QUAD") else 3
    path = tmp_path / "alias.exo"
    info, points, vtk_conn = _write_case(path, exo_name, n_nodes, num_dim)
    assert info.n_nodes == n_nodes
    _check_cells(path, info, points, vtk_conn, num_dim)


def test_shell9_reads_as_biquadratic_quad(tmp_path):
    """SHELL9 (written by other tools) keeps all 9 nodes. VTK's reader (<= 9.7)
    reads SHELL9 blocks with an 8-node stride, so it is no oracle here."""
    path = tmp_path / "shell9.exo"
    info, points, vtk_conn = _write_case(path, "SHELL9", None, 3)
    assert info.vtk_type == VTK.BIQUADRATIC_QUAD
    _check_cells(path, info, points, vtk_conn, 3, oracle=False)


def test_voxel_and_pixel_inputs(tmp_path):
    """Hexes/quads written from VTK voxels/pixels read back as native cells."""
    vol = pv.ImageData(dimensions=(3, 2, 2), spacing=(0.5, 0.25, 0.125))
    vol = vol.cast_to_unstructured_grid()
    assert set(vol.celltypes) == {VTK.VOXEL}
    conn = vol.cells.reshape(-1, 9)[:, 1:]
    exo = ct.to_exodus_conn(VTK.VOXEL, conn)
    write_model(tmp_path / "vox.exo",
                ExodusModel("vox", 3, vol.points, [Block(1, "v", "HEX8", exo)]))
    grid = rp.read_exo(tmp_path / "vox.exo")
    assert set(grid.celltypes) == {VTK.HEXAHEDRON}
    assert grid.validate_mesh().is_valid
    np.testing.assert_allclose(grid.compute_cell_sizes()["Volume"], 0.5 * 0.25 * 0.125)


def _mixed_model():
    """One file with many element types; element ids map a shuffled order."""
    blocks, all_points, expected = [], [], []
    types = ["HEX8", "TETRA10", "WEDGE6", "PYRAMID13", "HEX27", "TETRA4",
             "WEDGE15", "HEX20", "PYRAMID5"]
    offset = 0
    for k, name in enumerate(types):
        info = ct.resolve_exo_type(name)
        pts, vtk_conn = _two_cells(info, 3)
        pts = pts + [0.0, 2.0 * k, 0.0]
        vtk_conn = vtk_conn + offset
        blocks.append(Block(10 * (k + 1), name.lower(), name,
                            vtk_conn[:, list(info.vtk_to_exo)]))
        all_points.append(pts)
        expected.extend((info.vtk_type, row) for row in vtk_conn)
        offset += len(pts)
    points = np.vstack(all_points)
    num_elem = len(expected)
    elem_ids = np.random.default_rng(1).permutation(num_elem) + 1
    model = ExodusModel("mixed", 3, points, blocks, elem_id_map=elem_ids)
    return model, points, expected, elem_ids


@pytest.mark.parametrize("original_order", [True, False])
def test_mixed_types_and_original_order(tmp_path, original_order):
    model, points, expected, elem_ids = _mixed_model()
    path = tmp_path / "mixed.exo"
    write_model(path, model)
    grid = rp.read_exo(path, original_order=original_order)
    assert grid.n_cells == len(expected)
    order = np.argsort(elem_ids) if original_order else np.arange(len(expected))
    np.testing.assert_array_equal(grid.cell_data["exo_elem_id"], elem_ids[order])
    if original_order:
        np.testing.assert_array_equal(
            grid.cell_data["exo_elem_id"], np.arange(1, len(expected) + 1)
        )
    block_of = np.repeat([b.id for b in model.blocks], 2)
    np.testing.assert_array_equal(grid.cell_data["exo_block_id"], block_of[order])
    names = np.repeat([b.name for b in model.blocks], 2)
    assert list(grid.cell_data["exo_block_name"]) == list(names[order])
    for cell, file_index in enumerate(order):
        vtk_type, row = expected[file_index]
        assert grid.celltypes[cell] == vtk_type
        assert list(grid.get_cell(cell).point_ids) == list(row)
    assert grid.validate_mesh().is_valid

    # VTK's reader agrees cell by cell (it reports the id map as GlobalElementId)
    reader = pv.ExodusIIReader(str(path))
    reader.reader.GenerateGlobalElementIdArrayOn()
    blocks = reader.read()["Element Blocks"]
    by_id = {}
    for block in blocks:
        for i, gid in enumerate(block.cell_data["GlobalElementId"]):
            by_id[int(gid)] = (block.celltypes[i], _cell_points(block, i))
    for cell in range(grid.n_cells):
        vtk_type, pts = by_id[int(grid.cell_data["exo_elem_id"][cell])]
        assert grid.celltypes[cell] == vtk_type
        np.testing.assert_array_equal(_cell_points(grid, cell), pts)


def test_unsupported_blocks_warn_and_are_skipped(tmp_path):
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0]],
        dtype=float,
    )
    model = ExodusModel(
        "unsupported", 3, points,
        [
            Block(1, "tet", "TETRA4", [[0, 1, 2, 4]]),
            Block(2, "poly", "NSIDED", [[0, 1, 2, 3]]),
            Block(3, "hex7", "HEX", [[0, 1, 2, 3, 4, 5, 0]]),  # bad node count
            Block(4, "null", "NULL", np.zeros((0, 0), int)),
            Block(5, "empty", "HEX8", np.zeros((0, 8), int)),
            Block(6, "tet2", "TETRA4", [[1, 5, 2, 4]]),
        ],
        side_sets=[SideSet(9, "ss", [0, 1, 2, 3], [1, 1, 1, 2])],
        elem_id_map=[4, 3, 2, 1],
        times=[0.0],
        elem_vars={"e": {1: [[1.0]], 2: [[2.0]], 3: [[3.0]], 6: [[6.0]]}},
    )
    path = tmp_path / "unsupported.exo"
    write_model(path, model)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        grid, side_sets = rp.read_exo(path, read_side_sets=True)
    assert all(w.category is UserWarning for w in record)
    assert all(w.filename == __file__ for w in record)  # points at the caller
    messages = [str(w.message) for w in record]
    assert len(messages) == 2
    block_msg = [m for m in messages if "unsupported element type" in m]
    assert len(block_msg) == 1
    assert "2 ('NSIDED'" in block_msg[0] and "3 ('HEX'" in block_msg[0]
    assert "4 (" not in block_msg[0] and "5 (" not in block_msg[0]  # empty blocks
    (ss_msg,) = [m for m in messages if "Side set 9" in m]
    assert "dropping 2 of 4 entries" in ss_msg

    # the map is a permutation: kept cells sorted by element id
    assert grid.n_cells == 2
    assert list(grid.cell_data["exo_block_id"]) == [6, 1]
    assert list(grid.cell_data["exo_elem_id"]) == [1, 4]
    assert list(grid.celltypes) == [VTK.TETRA, VTK.TETRA]
    np.testing.assert_array_equal(grid.cell_data["e"], [6.0, 1.0])
    ss = side_sets["ss"]
    assert list(ss.cell_data["exo_elem_id"]) == [1, 4]  # file order indices
    assert list(ss.cell_data["cell_id"]) == [1, 0]

    with pytest.warns(UserWarning, match="unsupported element type"):
        fields = rp.read_element_fields(path)
    np.testing.assert_array_equal(fields["e"], [[6.0, 1.0]])

    with pytest.warns(UserWarning, match="unsupported element type"):
        grid = rp.read_exo(path, original_order=False)
    assert list(grid.cell_data["exo_block_id"]) == [1, 6]


def test_null_blocks_are_skipped_silently(datafile):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grid = rp.read_exo(datafile("mkmesh.par.2.0"))  # block 20 is NULL
    assert set(grid.cell_data["exo_block_id"]) == {10}
