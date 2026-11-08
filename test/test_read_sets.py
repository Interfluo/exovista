"""Side sets and node sets: exovista.read_side_sets / read_node_sets and
read_exo(read_side_sets=True).

VTK's reader (``pv.ExodusIIReader``) is the oracle for side geometry: it
builds side sets with the SEACAS Exodus library's side tables. Coordinates are
multiples of 1/64 so they survive VTK's float32 points exactly.
"""

import warnings

import netCDF4
import numpy as np
import pytest
import pyvista as pv
from pyvista.examples import cells as example_cells

from exovista import _celltypes as ct
from exovista import read_pyvista as rp
from exovista._backend import write_model
from exovista._model import Block, ExodusModel, NodeSet, SideSet

VTK = ct.VTK

_EXAMPLES = {
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


def _side_cases():
    cases = []
    for info in ct.EXO_TYPES.values():
        if not info.sides or info.name == "SHELL9":  # VTK misreads SHELL9
            continue
        cases.append((info.name, 2 if info.family in ("TRI", "QUAD") else 3))
    # a TRI element in a 3D file has shell sides; 2D bars have point sides
    cases += [("TRI3", 3), ("TRI6", 3), ("QUAD4", 3), ("BAR2", 2)]
    return cases


def _one_element_file(path, exo_name, num_dim, side_order=None):
    info = ct.resolve_exo_type(exo_name, num_dim=num_dim)
    cell = getattr(example_cells, _EXAMPLES[info.vtk_type])()
    pts = np.round(np.asarray(cell.points, dtype=float) * 64) / 64
    if num_dim == 2:
        pts[:, 2] = 0.0
    # shuffle the node numbering
    order = np.random.default_rng(info.n_nodes).permutation(len(pts))
    points = pts[order]
    vtk_conn = np.argsort(order)[None, :]
    exo_conn = vtk_conn[:, list(info.vtk_to_exo)]
    sides = list(range(1, info.n_sides + 1)) if side_order is None else side_order
    model = ExodusModel(
        "sides", num_dim, points[:, :num_dim],
        [Block(1, "blk", exo_name, exo_conn)],
        side_sets=[SideSet(k, f"side_{k}", [0], [k]) for k in sides]
        + [SideSet(100, "all", [0] * len(sides), sides)],
    )
    write_model(path, model)
    return info, points, exo_conn[0]


@pytest.mark.parametrize("exo_name,num_dim", _side_cases(),
                         ids=[f"{n}-{d}d" for n, d in _side_cases()])
def test_side_geometry_matches_vtk_reader(tmp_path, exo_name, num_dim):
    path = tmp_path / "sides.exo"
    info, points, exo_conn = _one_element_file(path, exo_name, num_dim)
    grid, side_sets = rp.read_exo(path, read_side_sets=True)
    assert side_sets.keys() == [f"side_{k}" for k in range(1, info.n_sides + 1)] + [
        "all"
    ]

    reader = pv.ExodusIIReader(str(path))
    reader.side_sets.enable_all()
    oracle = reader.read()["Side Sets"]
    corners = points[exo_conn[: info.n_corners]]
    for k, side in enumerate(info.sides, start=1):
        mesh = side_sets[f"side_{k}"]
        assert mesh.n_cells == 1
        assert mesh.n_points == grid.n_points
        assert mesh.points.dtype == np.float64
        kinds = (mesh.GetNumberOfVerts(), mesh.GetNumberOfLines(),
                 mesh.GetNumberOfPolys())
        assert kinds == {1: (1, 0, 0), 2: (0, 1, 0)}.get(len(side), (0, 0, 1))
        assert list(mesh.get_cell(0).point_ids) == list(exo_conn[list(side)])
        assert list(mesh.cell_data["exo_side"]) == [k]
        assert list(mesh.cell_data["exo_elem_id"]) == [1]
        assert list(mesh.cell_data["cell_id"]) == [0]
        np.testing.assert_array_equal(mesh.field_data["exo_side_set_id"], [k])
        # VTK's side: the same corner nodes in the same order
        vtk_mesh = oracle[f"side_{k}"]
        got = np.asarray(vtk_mesh.points)[vtk_mesh.get_cell(0).point_ids]
        is_corner = (got[:, None, :] == corners[None, :, :]).all(-1).any(-1)
        np.testing.assert_array_equal(
            got[is_corner], np.asarray(mesh.points)[mesh.get_cell(0).point_ids]
        )
    # all sides of the element in one set
    mesh = side_sets["all"]
    assert mesh.n_cells == info.n_sides
    assert sorted(mesh.cell_data["exo_side"]) == list(range(1, info.n_sides + 1))
    for i in range(mesh.n_cells):
        side = info.sides[mesh.cell_data["exo_side"][i] - 1]
        assert list(mesh.get_cell(i).point_ids) == list(exo_conn[list(side)])


def test_mixed_side_kinds_keep_cell_data_aligned(tmp_path):
    """Shell faces (polygons) and edges (lines) in one set: PolyData stores
    lines before polygons; the cell data follows the cells."""
    path = tmp_path / "shell.exo"
    order = [3, 1, 6, 2, 4]
    info, points, exo_conn = _one_element_file(path, "SHELL4", 3, side_order=order)
    mesh = rp.read_side_sets(path)["all"]
    assert list(mesh.cell_data["exo_side"]) == [3, 6, 4, 1, 2]
    assert (mesh.GetNumberOfLines(), mesh.GetNumberOfPolys()) == (3, 2)
    for i in range(mesh.n_cells):
        side = info.sides[mesh.cell_data["exo_side"][i] - 1]
        assert list(mesh.get_cell(i).point_ids) == list(exo_conn[list(side)])
    np.testing.assert_array_equal(mesh.cell_data["face_id"], mesh.cell_data["exo_side"])


def _hex_box_model(elem_id_map="permuted"):
    """4x3x2 hexes in two interleaved blocks with a boundary side set per face
    direction; element ids record the original (input) cell order."""
    img = pv.ImageData(dimensions=(5, 4, 3), spacing=(0.5, 0.25, 0.125))
    vox = img.cast_to_unstructured_grid()
    rows = vox.cells.reshape(-1, 9)[:, 1:][:, [0, 1, 3, 2, 4, 5, 7, 6]]
    n = len(rows)
    region = np.arange(n) % 2
    file_order = np.argsort(region, kind="stable")
    blocks = [
        Block(r + 1, f"r{r}", "HEX8", rows[file_order[region[file_order] == r]])
        for r in (0, 1)
    ]
    file_pos = np.empty(n, int)
    file_pos[file_order] = np.arange(n)
    centers = np.asarray(vox.points)[rows].mean(axis=1)
    bottom = np.flatnonzero(centers[:, 2] < 0.125)
    xmax = np.flatnonzero(centers[:, 0] > 1.5)
    side_sets = [
        SideSet(1, "bottom", file_pos[bottom], np.full(len(bottom), 5)),
        SideSet(2, "xmax", file_pos[xmax], np.full(len(xmax), 2)),
        SideSet(3, "empty", [], []),
    ]
    ids = {"permuted": file_order + 1, "none": None,
           "gaps": 10 * (file_order + 1)}[elem_id_map]
    model = ExodusModel("box", 3, img.points, blocks, side_sets=side_sets,
                        elem_id_map=ids)
    return model, rows, file_order


@pytest.mark.parametrize("elem_id_map", ["permuted", "none", "gaps"])
@pytest.mark.parametrize("original_order", [True, False])
def test_side_set_cell_ids(tmp_path, elem_id_map, original_order):
    model, rows, file_order = _hex_box_model(elem_id_map)
    path = tmp_path / "box.exo"
    write_model(path, model)
    grid, side_sets = rp.read_exo(path, read_side_sets=True,
                                  original_order=original_order)
    assert side_sets.keys() == ["bottom", "xmax", "empty"]
    assert rp.read_side_sets(path, original_order=original_order).keys() == (
        side_sets.keys()
    )
    raw = rp.read_exo(path, original_order=False)
    for ss, mesh in zip(model.side_sets, side_sets):
        assert mesh.n_points == grid.n_points
        np.testing.assert_array_equal(mesh.points, grid.points)
        np.testing.assert_array_equal(mesh.cell_data["exo_elem_id"], ss.elems + 1)
        np.testing.assert_array_equal(mesh.cell_data["exo_side"], ss.sides)
        np.testing.assert_array_equal(mesh.cell_data["orig_elem_id"], ss.elems + 1)
        np.testing.assert_array_equal(mesh.cell_data["face_id"], ss.sides)
        np.testing.assert_array_equal(mesh.field_data["exo_side_set_id"], [ss.id])
        cell_id = mesh.cell_data["cell_id"]
        assert cell_id.dtype.kind == "i"
        # exo_elem_id is the file-order index: the cell of the file-order grid
        np.testing.assert_array_equal(
            raw.cell_data["exo_elem_id"][ss.elems],
            grid.cell_data["exo_elem_id"][cell_id],
        )
        if original_order and elem_id_map == "permuted":
            np.testing.assert_array_equal(cell_id, file_order[ss.elems])
        elif not original_order or elem_id_map != "permuted":
            np.testing.assert_array_equal(cell_id, ss.elems)
        # every side's nodes belong to the cell it names
        for i in range(mesh.n_cells):
            nodes = set(mesh.get_cell(i).point_ids)
            assert len(nodes) == 4
            assert nodes <= set(grid.get_cell(cell_id[i]).point_ids)
    assert side_sets["empty"].n_cells == 0
    assert len(side_sets["empty"].cell_data["cell_id"]) == 0


def test_side_sets_match_vtk_geometry_on_a_box(tmp_path):
    model, _, _ = _hex_box_model()
    path = tmp_path / "box.exo"
    write_model(path, model)
    ours = rp.read_side_sets(path)
    reader = pv.ExodusIIReader(str(path))
    reader.side_sets.enable_all()
    oracle = reader.read()["Side Sets"]
    for name in ("bottom", "xmax"):
        a = ours[name].cell_centers().points
        b = oracle[name].cell_centers().points
        np.testing.assert_array_equal(a, b)  # same faces, in the same order
        np.testing.assert_allclose(_normals(ours[name]), _normals(oracle[name]))
    # outward normals
    np.testing.assert_allclose(_normals(ours["bottom"]), [[0, 0, -1]] * 12)
    np.testing.assert_allclose(_normals(ours["xmax"]), [[1, 0, 0]] * 6)


def _normals(mesh):
    """Unit normal of each (planar) face from its first three points."""
    out = []
    for i in range(mesh.n_cells):
        p = np.asarray(mesh.points)[mesh.get_cell(i).point_ids]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        out.append(n / np.linalg.norm(n))
    return np.array(out)


def test_2d_side_sets_are_lines(tmp_path):
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]], float)
    model = ExodusModel(
        "2d", 2, coords,
        [Block(1, "q", "QUAD4", [[0, 1, 2, 3]]), Block(2, "t", "TRI3", [[1, 4, 5]])],
        side_sets=[SideSet(7, "edges", [0, 1, 1, 0], [1, 1, 2, 4])],
    )
    write_model(tmp_path / "2d.exo", model)
    mesh = rp.read_side_sets(tmp_path / "2d.exo")["edges"]
    assert mesh.n_cells == mesh.GetNumberOfLines() == 4
    lines = [tuple(mesh.get_cell(i).point_ids) for i in range(4)]
    assert lines == [(0, 1), (1, 4), (4, 5), (3, 0)]
    np.testing.assert_array_equal(mesh.points[:, 2], 0.0)
    reader = pv.ExodusIIReader(str(tmp_path / "2d.exo"))
    reader.side_sets.enable_all()
    vtk_mesh = reader.read()["Side Sets"]["edges"]
    np.testing.assert_array_equal(vtk_mesh.cell_centers().points,
                                  mesh.cell_centers().points)


def test_side_set_keys_are_unique(tmp_path):
    coords = np.eye(4, 3)
    sets = [SideSet(5, "", [0], [1]), SideSet(6, "a", [0], [2]),
            SideSet(7, "a", [0], [3]), SideSet(8, "side_set_5", [0], [4]),
            SideSet(9, "", [], [])]
    write_model(tmp_path / "k.exo", ExodusModel(
        "k", 3, coords, [Block(1, "t", "TETRA4", [[0, 1, 2, 3]])], side_sets=sets))
    side_sets = rp.read_side_sets(tmp_path / "k.exo")
    assert side_sets.keys() == ["side_set_5", "a", "side_set_7", "side_set_8",
                                "side_set_9"]
    ids = [int(m.field_data["exo_side_set_id"][0]) for m in side_sets]
    assert ids == [5, 6, 7, 8, 9]
    assert [m.cell_data["exo_side"][0] for m in list(side_sets)[:4]] == [1, 2, 3, 4]


def test_unnamed_side_sets_of_data_file(datafile):
    side_sets = rp.read_side_sets(datafile("mkmesh.gen"))
    assert side_sets.keys() == ["side_set_200", "side_set_201"]
    assert [m.n_cells for m in side_sets] == [6, 6]
    assert all(m.GetNumberOfLines() == 6 for m in side_sets)  # 2D file: edges


def test_bad_side_entries_warn_and_are_dropped(tmp_path):
    coords = np.vstack([np.eye(4, 3), np.eye(4, 3) + 2])
    model = ExodusModel(
        "bad", 3, coords,
        [Block(1, "t", "TETRA4", [[0, 1, 2, 3], [4, 5, 6, 7]])],
        side_sets=[SideSet(3, "s", [0, 1, 0, 1], [1, 2, 3, 4])],
    )
    path = tmp_path / "bad.exo"
    write_model(path, model)
    with netCDF4.Dataset(path, "a") as ds:  # corrupt two entries
        ds["side_ss1"][1] = 9
        ds["elem_ss1"][2] = 7
    with pytest.warns(UserWarning, match=r"dropping 2 of 4 entries") as record:
        mesh = rp.read_side_sets(path)["s"]
    message = str(record[0].message)
    assert "side 9" in message and "outside 1..2" in message
    np.testing.assert_array_equal(mesh.cell_data["exo_elem_id"], [1, 2])
    np.testing.assert_array_equal(mesh.cell_data["exo_side"], [1, 4])
    np.testing.assert_array_equal(mesh.cell_data["cell_id"], [0, 1])


# --------------------------------------------------------------------------- #
# node sets
# --------------------------------------------------------------------------- #
def test_node_sets(tmp_path):
    coords = np.arange(30.0).reshape(10, 3) / 64
    sets = [NodeSet(4, "bottom", [3, 1, 2]), NodeSet(9, "", [0, 9]),
            NodeSet(5, "bottom", [7]), NodeSet(1, "empty", [])]
    blocks = [Block(1, "s", "SPHERE", np.arange(10)[:, None])]
    path = tmp_path / "ns.exo"
    write_model(path, ExodusModel("ns", 3, coords, blocks, node_sets=sets))
    node_sets = rp.read_node_sets(path)
    assert list(node_sets) == ["bottom", "node_set_9", "node_set_5", "empty"]
    np.testing.assert_array_equal(node_sets["bottom"], [3, 1, 2])
    np.testing.assert_array_equal(node_sets["node_set_9"], [0, 9])
    np.testing.assert_array_equal(node_sets["node_set_5"], [7])
    assert node_sets["empty"].size == 0
    assert all(v.dtype == np.int64 for v in node_sets.values())

    # VTK's reader: the same nodes (it numbers the sets' points itself)
    grid = rp.read_exo(path)
    reader = pv.ExodusIIReader(str(path))
    reader.node_sets.enable_all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oracle = reader.read()["Node Sets"]
    for key, name in [("bottom", "bottom"), ("node_set_9", "Unnamed set ID: 9")]:
        if name not in oracle.keys():
            continue
        np.testing.assert_array_equal(
            np.sort(oracle[name].points, axis=0),
            np.sort(grid.points[node_sets[key]], axis=0),
        )


def test_node_sets_of_data_files(datafile):
    sets = rp.read_node_sets(datafile("mkmesh.gen"))
    assert list(sets) == ["node_set_100", "node_set_101"]
    assert [len(v) for v in sets.values()] == [6, 6]
    sets = rp.read_node_sets(datafile("edges.base.exo"))
    assert len(sets) == 14
    assert all(k.startswith("node_set_") for k in sets)
