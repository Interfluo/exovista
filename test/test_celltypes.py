"""Oracle tests for exovista._celltypes.

Every VTK <-> Exodus node permutation and every Exodus side table is checked
against VTK's own ExodusII reader (``pv.ExodusIIReader``), which applies its
own Exodus -> VTK permutation and builds side-set geometry with the SEACAS
Exodus library's side tables. Files are written through
``exovista._backend.write_model``.

Coordinates are multiples of 1/64 so they survive VTK's float32 points
exactly and comparisons can be exact.
"""

import numpy as np
import pytest
import pyvista as pv
from pyvista.examples import cells as example_cells

from exovista import _celltypes as ct
from exovista._backend import open_reader, write_model
from exovista._model import Block, ExodusModel, SideSet

VTK = ct.VTK

# VTK's axis-aligned cells use a different corner order than the native cell
# they are written as (VTK_PIXEL -> QUAD4/SHELL4, VTK_VOXEL -> HEX8): the
# native connectivity is ``conn[:, order]``. Spelled out independently of
# ``_celltypes`` so the permutation under test is never its own oracle.
_AXIS_ALIGNED_TO_NATIVE = {
    VTK.PIXEL: [0, 1, 3, 2],
    VTK.VOXEL: [0, 1, 3, 2, 4, 5, 7, 6],
}

# VTK type -> pyvista example-cell constructor providing a valid geometry
_EXAMPLES = {
    VTK.VERTEX: "Vertex",
    VTK.LINE: "Line",
    VTK.QUADRATIC_EDGE: "QuadraticEdge",
    VTK.TRIANGLE: "Triangle",
    VTK.QUADRATIC_TRIANGLE: "QuadraticTriangle",
    VTK.QUAD: "Quadrilateral",
    VTK.PIXEL: "Pixel",
    VTK.QUADRATIC_QUAD: "QuadraticQuadrilateral",
    VTK.BIQUADRATIC_QUAD: "BiQuadraticQuadrilateral",
    VTK.TETRA: "Tetrahedron",
    VTK.QUADRATIC_TETRA: "QuadraticTetrahedron",
    VTK.HEXAHEDRON: "Hexahedron",
    VTK.VOXEL: "Voxel",
    VTK.QUADRATIC_HEXAHEDRON: "QuadraticHexahedron",
    VTK.TRIQUADRATIC_HEXAHEDRON: "TriQuadraticHexahedron",
    VTK.WEDGE: "Wedge",
    VTK.QUADRATIC_WEDGE: "QuadraticWedge",
    VTK.PYRAMID: "Pyramid",
    VTK.QUADRATIC_PYRAMID: "QuadraticPyramid",
}

# (vtk type, num_dim) combinations exovista writes
_WRITE_CASES = [
    (vtk_type, num_dim)
    for vtk_type in sorted(ct.SUPPORTED_VTK_TYPES)
    for num_dim in (2, 3)
    if not (num_dim == 2 and ct.EXO_TYPES[ct.exo_type_for(vtk_type, 3)].dim == 3)
]


def _reference_points(vtk_type, num_dim):
    """Points of one valid cell of ``vtk_type`` (VTK order), on a 1/64 grid."""
    cell = getattr(example_cells, _EXAMPLES[vtk_type])()
    pts = np.round(np.asarray(cell.points, dtype=float) * 64) / 64
    n = pts.shape[0]
    assert len(np.unique(pts, axis=0)) == n
    assert list(cell.cell_connectivity) == list(range(n))
    if num_dim == 2:
        pts[:, 2] = 0.0
    elif ct.EXO_TYPES[ct.exo_type_for(vtk_type, 3)].dim < 3:
        # make 2D / 1D cells genuinely three dimensional
        pts[:, 2] += (np.arange(n) % 3) / 8.0
    return pts


def _two_cell_mesh(vtk_type, num_dim):
    """Two cells of ``vtk_type`` with shuffled node numbering."""
    pts = _reference_points(vtk_type, num_dim)
    n = len(pts)
    points = np.vstack([pts, pts + [4.0, 0.0, 0.0]])
    order = np.random.default_rng(vtk_type).permutation(2 * n)
    inverse = np.argsort(order)
    conn = inverse[np.arange(2 * n).reshape(2, n)]
    return points[order], conn


def _oracle_grid(path):
    reader = pv.ExodusIIReader(str(path))
    return reader.read()["Element Blocks"][0]


def _cell_points(grid, i):
    return np.asarray(grid.points)[grid.get_cell(i).point_ids]


@pytest.mark.parametrize(
    "vtk_type,num_dim",
    _WRITE_CASES,
    ids=[f"{ct.vtk_type_name(t)}-{d}d" for t, d in _WRITE_CASES],
)
def test_permutation_matches_vtk_reader(tmp_path, vtk_type, num_dim):
    points, conn = _two_cell_mesh(vtk_type, num_dim)
    info = ct.exo_info_for(vtk_type, num_dim)
    model = ExodusModel(
        "celltype",
        num_dim,
        points[:, :num_dim],
        [Block(7, "blk", info.name, ct.to_exodus_conn(vtk_type, conn))],
    )
    path = tmp_path / "cells.exo"
    write_model(path, model)

    # VOXEL/PIXEL are written as HEX8/QUAD4 (SHELL4) and read back in their
    # native VTK order. The expected order is spelled out here, independently
    # of the permutation under test.
    native = conn[:, _AXIS_ALIGNED_TO_NATIVE[vtk_type]] if vtk_type in (
        _AXIS_ALIGNED_TO_NATIVE
    ) else conn

    grid = _oracle_grid(path)
    assert grid.n_cells == 2
    assert list(grid.celltypes) == [info.vtk_type] * 2
    for i in range(2):
        np.testing.assert_array_equal(_cell_points(grid, i), points[native[i]])
    # Geometric check: what VTK reads back has the size of the input cells.
    size = "Volume" if info.dim == 3 else "Area" if info.dim == 2 else None
    if size is not None and (info.dim == 3 or num_dim == 2):
        source = pv.UnstructuredGrid({vtk_type: conn}, points)
        expected = source.compute_cell_sizes()[size]
        assert np.all(expected > 0)
        np.testing.assert_allclose(grid.compute_cell_sizes()[size], expected)

    # and back through exovista's own reader
    with open_reader(path) as reader:
        (block,) = reader.blocks()
    assert block.elem_type == info.name
    assert ct.resolve_exo_type(block.elem_type, block.n_nodes_per_elem) == info
    np.testing.assert_array_equal(
        ct.to_vtk_conn(block.elem_type, block.conn, num_dim), native
    )


@pytest.mark.parametrize(
    "exo_name,vtk_type,num_dim",
    [
        ("BEAM2", VTK.LINE, 3),
        ("BEAM3", VTK.QUADRATIC_EDGE, 3),
        ("TRUSS2", VTK.LINE, 2),
        ("TRUSS3", VTK.QUADRATIC_EDGE, 3),
        ("BAR", VTK.LINE, 3),
        ("CIRCLE", VTK.VERTEX, 2),
        ("TRI", VTK.TRIANGLE, 3),
        ("TRIANGLE", VTK.QUADRATIC_TRIANGLE, 2),
        ("SHELL", VTK.QUADRATIC_QUAD, 3),
        ("QUAD", VTK.BIQUADRATIC_QUAD, 3),
        ("TET", VTK.QUADRATIC_TETRA, 3),
        ("HEX", VTK.QUADRATIC_HEXAHEDRON, 3),
        ("HEXAHEDRON", VTK.TRIQUADRATIC_HEXAHEDRON, 3),
        ("WEDGE", VTK.QUADRATIC_WEDGE, 3),
        ("PYRA", VTK.QUADRATIC_PYRAMID, 3),
        # libMesh / MOOSE 1D elements, and names matched by 3 letters
        ("EDGE2", VTK.LINE, 3),
        ("EDGE3", VTK.QUADRATIC_EDGE, 2),
        ("STRAIGHT2", VTK.LINE, 3),
        ("TRIA3", VTK.TRIANGLE, 2),
        ("TRIA6", VTK.QUADRATIC_TRIANGLE, 3),
        ("HEXA8", VTK.HEXAHEDRON, 3),
        ("HEXA", VTK.QUADRATIC_HEXAHEDRON, 3),
        ("QUADRANGLE", VTK.QUAD, 2),
        ("TETRAHEDRAL", VTK.TETRA, 3),
        ("PYRAMIDAL", VTK.PYRAMID, 3),
        ("SPHERICAL", VTK.VERTEX, 3),
        ("TRISHELL3", VTK.TRIANGLE, 2),
    ],
)
def test_read_aliases_match_vtk_reader(tmp_path, exo_name, vtk_type, num_dim):
    """Names other tools write resolve to what VTK's reader produces."""
    points, conn = _two_cell_mesh(vtk_type, num_dim)
    n_nodes = conn.shape[1]
    info = ct.resolve_exo_type(exo_name, n_nodes, num_dim)
    exo_conn = conn[:, list(info.vtk_to_exo)]
    model = ExodusModel(
        "alias", num_dim, points[:, :num_dim], [Block(1, "b", exo_name, exo_conn)]
    )
    write_model(tmp_path / "alias.exo", model)
    grid = _oracle_grid(tmp_path / "alias.exo")
    got_type, perm = ct.vtk_type_for(exo_name, n_nodes, num_dim)
    assert got_type == vtk_type
    assert list(grid.celltypes) == [vtk_type] * 2
    for i in range(2):
        np.testing.assert_array_equal(
            _cell_points(grid, i), points[exo_conn[i][perm]]
        )


def test_shell9_is_read_as_biquadratic_quad(tmp_path):
    """SHELL9 (written by other tools) reads back with all 9 nodes."""
    points, conn = _two_cell_mesh(VTK.BIQUADRATIC_QUAD, 3)
    model = ExodusModel("s9", 3, points, [Block(1, "s", "SHELL9", conn)])
    write_model(tmp_path / "s9.exo", model)
    with open_reader(tmp_path / "s9.exo") as reader:
        (block,) = reader.blocks()
    assert block.elem_type == "SHELL9"
    vtk_type, perm = ct.vtk_type_for(block.elem_type, block.n_nodes_per_elem)
    assert vtk_type == VTK.BIQUADRATIC_QUAD
    np.testing.assert_array_equal(block.conn[:, perm], conn)


def _side_cases():
    cases = []
    for info in ct.EXO_TYPES.values():
        if not info.sides:
            continue
        num_dim = 2 if info.family in ("TRI", "QUAD") else 3
        cases.append((info.name, num_dim))
    # Like the Exodus library, a TRI element in a 3D file has shell sides and
    # a TRISHELL element in a 2D file has the three edges of a TRI.
    cases += [("TRI3", 3), ("TRI6", 3), ("TRISHELL3", 2), ("TRISHELL6", 2)]
    # libMesh / MOOSE name for BAR elements
    cases += [("EDGE2", 3), ("EDGE3", 2)]
    return cases


@pytest.mark.parametrize("exo_name,num_dim", _side_cases())
def test_side_table_matches_vtk_reader(tmp_path, exo_name, num_dim):
    info = ct.resolve_exo_type(exo_name, num_dim=num_dim)
    pts = _reference_points(info.vtk_type, num_dim)
    econn = np.arange(info.n_nodes)[None, :][:, list(info.vtk_to_exo)]
    side_sets = [
        SideSet(k, f"side_{k}", [0], [k]) for k in range(1, info.n_sides + 1)
    ]
    model = ExodusModel(
        "sides",
        num_dim,
        pts[:, :num_dim],
        [Block(1, "blk", exo_name, econn)],
        side_sets=side_sets,
    )
    path = tmp_path / "sides.exo"
    write_model(path, model)

    reader = pv.ExodusIIReader(str(path))
    reader.side_sets.enable_all()
    sets = reader.read()["Side Sets"]
    corners = pts[econn[0, : info.n_corners]]
    for k, side in enumerate(ct.side_corner_nodes(exo_name, num_dim=num_dim), 1):
        mesh = sets[f"side_{k}"]
        assert mesh.n_cells == 1
        got = _cell_points(mesh, 0)
        # keep the element's corner nodes, in VTK's (= Exodus') order
        is_corner = (got[:, None, :] == corners[None, :, :]).all(-1).any(-1)
        np.testing.assert_array_equal(
            got[is_corner], pts[econn[0, list(side)]], err_msg=f"side {k}"
        )


# --------------------------------------------------------------------------- #
# Pure table / helper checks
# --------------------------------------------------------------------------- #
def test_tables_are_consistent():
    for info in ct.EXO_TYPES.values():
        assert sorted(info.exo_to_vtk) == list(range(info.n_nodes))
        assert [info.exo_to_vtk[i] for i in info.vtk_to_exo] == list(
            range(info.n_nodes)
        )
        assert info.n_corners <= info.n_nodes
        for side in info.sides:
            assert all(0 <= node < info.n_corners for node in side)
            assert len(set(side)) == len(side)


@pytest.mark.parametrize("vtk_type", sorted(ct.SUPPORTED_VTK_TYPES))
def test_conn_roundtrip(vtk_type):
    info = ct.exo_info_for(vtk_type, 3)
    conn = np.random.default_rng(0).permutation(3 * info.n_nodes)
    conn = conn.reshape(3, info.n_nodes)
    exo = ct.to_exodus_conn(vtk_type, conn)
    back = ct.to_vtk_conn(info.name, exo)
    if vtk_type in _AXIS_ALIGNED_TO_NATIVE:
        np.testing.assert_array_equal(
            back, conn[:, _AXIS_ALIGNED_TO_NATIVE[vtk_type]]
        )
    else:
        np.testing.assert_array_equal(back, conn)
    native = ct.vtk_type_for(info.name)
    assert native[0] == info.vtk_type


def test_exo_type_for():
    assert ct.exo_type_for(VTK.TRIANGLE, 2) == "TRI3"
    assert ct.exo_type_for(VTK.TRIANGLE, 3) == "TRISHELL3"
    assert ct.exo_type_for(VTK.QUADRATIC_TRIANGLE, 3) == "TRISHELL6"
    assert ct.exo_type_for(VTK.QUAD, 2) == "QUAD4"
    assert ct.exo_type_for(VTK.PIXEL, 3) == "SHELL4"
    assert ct.exo_type_for(VTK.BIQUADRATIC_QUAD, 2) == "QUAD9"
    # SHELL9 is mis-read by VTK's reader, see _celltypes
    assert ct.exo_type_for(VTK.BIQUADRATIC_QUAD, 3) == "QUAD9"
    assert ct.exo_type_for(VTK.VOXEL, 3) == "HEX8"
    assert ct.exo_type_for(VTK.TRIQUADRATIC_HEXAHEDRON, 3) == "HEX27"
    assert ct.exo_type_for(VTK.LINE, 2) == "BAR2"
    assert ct.exo_type_for(VTK.QUADRATIC_EDGE, 3) == "BAR3"
    assert ct.exo_type_for(VTK.VERTEX, 3) == "SPHERE"
    assert ct.exo_type_for(pv.CellType.QUADRATIC_PYRAMID, 3) == "PYRAMID13"


def test_written_names():
    # meshio (<= 5.3) only reads the short names of these types (and knows no
    # TRISHELL; a TRI in a 3D file is a triangular shell for every Exodus
    # reader)
    short = {"TETRA4": "TETRA", "WEDGE6": "WEDGE", "PYRAMID5": "PYRAMID",
             "TRISHELL3": "TRI3", "TRISHELL6": "TRI6"}
    for name, info in ct.EXO_TYPES.items():
        assert info.written_name == short.get(name, name)
        # every reader resolves the written name (plus node count) to the type
        num_dim = {"TRI": 2, "TRISHELL": 3}.get(info.family)
        assert ct.resolve_exo_type(info.written_name, info.n_nodes, num_dim) is info
    for name, written in short.items():
        # without a node count the short name is still the linear element
        num_dim = 3 if name.startswith("TRISHELL") else None
        assert ct.resolve_exo_type(written, num_dim=num_dim) is ct.EXO_TYPES[name]


@pytest.mark.parametrize(
    "vtk_type",
    [
        pv.CellType.POLYGON,
        pv.CellType.POLYHEDRON,
        pv.CellType.POLY_LINE,
        pv.CellType.POLY_VERTEX,
        pv.CellType.TRIANGLE_STRIP,
        pv.CellType.LAGRANGE_HEXAHEDRON,
        pv.CellType.TRIQUADRATIC_PYRAMID,
        pv.CellType.BIQUADRATIC_TRIANGLE,
        pv.CellType.QUADRATIC_LINEAR_WEDGE,
    ],
)
def test_unsupported_vtk_types(vtk_type):
    with pytest.raises(ct.UnsupportedCellTypeError, match=vtk_type.name):
        ct.exo_type_for(vtk_type, 3)
    assert vtk_type not in ct.SUPPORTED_VTK_TYPES
    assert issubclass(ct.UnsupportedCellTypeError, ValueError)


def test_3d_cells_rejected_in_2d_files():
    with pytest.raises(ct.UnsupportedCellTypeError, match="num_dim=2"):
        ct.exo_type_for(VTK.TETRA, 2)
    with pytest.raises(ValueError):
        ct.exo_type_for(VTK.TETRA, 4)


@pytest.mark.parametrize(
    "name,n_nodes,num_dim,expected",
    [
        ("HEX", None, None, "HEX8"),
        ("hex8", None, None, "HEX8"),
        (" Hexahedron ", 20, None, "HEX20"),
        ("HEX", 27, 3, "HEX27"),
        ("HEX8", 20, None, "HEX20"),  # node count wins, as in Exodus
        ("TET", None, None, "TETRA4"),
        ("TETRA", 10, None, "TETRA10"),
        ("tetrahedron", 4, None, "TETRA4"),
        ("TETRA10", None, None, "TETRA10"),
        ("WEDGE", 15, None, "WEDGE15"),
        ("PYRA", None, None, "PYRAMID5"),
        ("PYRAMID13", None, None, "PYRAMID13"),
        ("TRI", None, 2, "TRI3"),
        ("TRIANGLE", 6, 2, "TRI6"),
        ("TRI3", 3, 3, "TRISHELL3"),
        ("TRISHELL", 6, None, "TRISHELL6"),
        ("QUAD", None, None, "QUAD4"),
        ("QUADRILATERAL", 8, None, "QUAD8"),
        ("QUAD9", None, 3, "QUAD9"),
        ("SHELL", None, None, "SHELL4"),
        ("SHELL9", None, None, "SHELL9"),
        ("BAR", None, None, "BAR2"),
        ("TRUSS3", None, None, "BAR3"),
        ("BEAM", 3, None, "BEAM3"),
        ("BEAM2", None, None, "BEAM2"),
        ("SPHERE", 1, None, "SPHERE"),
        ("CIRCLE", None, None, "SPHERE"),
        # first three letters decide, as in the Exodus library and VTK
        ("EDGE2", None, None, "BAR2"),
        ("edge", 3, 3, "BAR3"),
        ("STRAIGHT2", None, None, "BAR2"),
        ("TRIA3", None, 2, "TRI3"),
        ("HEXA", 20, None, "HEX20"),
        ("SHE", 8, None, "SHELL8"),
        ("PYRAMIDAL", 13, None, "PYRAMID13"),
        # only the leading letters count: separators and suffixes are ignored
        ("HEX_8", None, None, "HEX8"),
        ("HEX8_ALT", 8, None, "HEX8"),
        ("HEX-8", 8, None, "HEX8"),
        ("HEX8S", 8, None, "HEX8"),
        ("Hex 8", 8, None, "HEX8"),
        ("HEX8_2", None, None, "HEX8"),
        ("TETRA_10", None, None, "TETRA10"),
        ("tri-6", None, 2, "TRI6"),
        # TRI <-> TRISHELL follows num_dim
        ("TRISHELL3", 3, 2, "TRI3"),
        ("TRISHELL", 6, 2, "TRI6"),
        ("TRISHELL3", None, 3, "TRISHELL3"),
        ("TRIA", 6, 3, "TRISHELL6"),
    ],
)
def test_resolve_aliases(name, n_nodes, num_dim, expected):
    assert ct.resolve_exo_type(name, n_nodes, num_dim).name == expected


@pytest.mark.parametrize(
    "name,n_nodes",
    [
        ("FOO", None),
        ("HEX", 7),
        ("TETRA", 11),
        ("NULL", 0),
        ("", None),
        ("TR", None),
        ("STR2", None),  # VTK only knows the full word STRAIGHT
        ("PENTA6", None),
        ("_HEX8", None),
        ("8HEX", None),
        ("NSIDED", None),
        ("NFACED", None),
    ],
)
def test_resolve_unknown(name, n_nodes):
    assert ct.resolve_exo_type(name, n_nodes) is None
    assert ct.vtk_type_for(name, n_nodes) is None
    with pytest.raises(ct.UnsupportedCellTypeError):
        ct.side_corner_nodes(name, n_nodes)


def test_helpers():
    assert ct.n_corners("HEX27") == 8
    assert ct.n_corners("TETRA", 10) == 4
    assert ct.side_corner_nodes("HEX8")[0] == (0, 1, 5, 4)
    assert ct.side_corner_nodes("TRI3", num_dim=3)[1] == (2, 1, 0)
    assert ct.side_corner_nodes("TRISHELL3", num_dim=2) == [(0, 1), (1, 2), (2, 0)]
    vtk_type, perm = ct.vtk_type_for("HEX20")
    assert vtk_type == VTK.QUADRATIC_HEXAHEDRON
    assert list(perm[12:]) == [16, 17, 18, 19, 12, 13, 14, 15]
    with pytest.raises(ValueError, match="shape"):
        ct.to_exodus_conn(VTK.HEXAHEDRON, np.zeros((2, 4), int))


def test_trishell_in_2d_file_has_tri_sides(tmp_path):
    """Like the Exodus library, a TRISHELL block in a 2D file is a TRI."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    blocks = [Block(1, "b", "TRISHELL3", [[0, 1, 2]])]
    ok = ExodusModel("ok", 2, coords, blocks, side_sets=[SideSet(1, "s", [0], [3])])
    ok.validate()
    bad = ExodusModel("bad", 2, coords, blocks, side_sets=[SideSet(1, "s", [0], [4])])
    with pytest.raises(ValueError, match="3 sides"):
        write_model(tmp_path / "bad.exo", bad)
    # in a 3D file the same element keeps its 2 faces + 3 edges
    coords3 = np.c_[coords, [0.0, 0.0, 0.5]]
    ExodusModel(
        "ok3", 3, coords3, blocks, side_sets=[SideSet(1, "s", [0], [5])]
    ).validate()
