"""Side sets written by exovista.write_exo, checked with VTK's ExodusII reader.

VTK rebuilds every side from the (element, side) pairs in the file with its
own Exodus side tables, so comparing VTK's side cells with the surface cells
given to write_exo checks the whole matcher. VTK's ``SourceElementId`` is the
element's global id - 1, i.e. (through the element id map write_exo stores)
the ORIGINAL cell index of the owning element.
"""

import inspect
import os
import warnings

import numpy as np
import pytest
import pyvista as pv
from pyvista.examples import cells as example_cells

from exovista._backend import open_reader
from exovista.write_pyvista import write_exo

CT = pv.CellType
_N_CORNERS = {
    CT.LINE: 2, CT.QUADRATIC_EDGE: 2, CT.TRIANGLE: 3, CT.QUADRATIC_TRIANGLE: 3,
    CT.QUAD: 4, CT.PIXEL: 4, CT.QUADRATIC_QUAD: 4, CT.BIQUADRATIC_QUAD: 4,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def vtk_side_sets(path):
    reader = pv.ExodusIIReader(str(path))
    reader.side_sets.enable_all()
    reader.reader.GenerateGlobalElementIdArrayOn()
    return reader.read()["Side Sets"]


def file_side_sets(path):
    with open_reader(path) as reader:
        return [(s.id, s.name, len(s.elems)) for s in reader.side_sets()]


def extract_surface(mesh, **kwargs):
    """``extract_surface`` (owner ids kept) without pyvista's FutureWarning."""
    if "algorithm" in inspect.signature(mesh.extract_surface).parameters:
        kwargs.setdefault("algorithm", "dataset_surface")
    kwargs.setdefault("nonlinear_subdivision", 0)
    return mesh.extract_surface(pass_cellid=True, **kwargs)


def corner_sets(mesh):
    """Per cell: the sorted corner coordinates (orientation-free)."""
    out = []
    points = np.asarray(mesh.points)
    for i in range(mesh.n_cells):
        cell = mesh.get_cell(i)
        k = _N_CORNERS.get(cell.type, len(cell.point_ids))
        out.append(sorted(map(tuple, points[cell.point_ids[:k]])))
    return out


def assert_side_set(vtk_set, surface, owners):
    """VTK's side cells are the surface cells, in order, owned by ``owners``."""
    assert vtk_set.n_cells == surface.n_cells
    assert corner_sets(vtk_set) == corner_sets(surface)
    np.testing.assert_array_equal(vtk_set.cell_data["SourceElementId"], owners)


def make_grid(cells, points):
    return pv.UnstructuredGrid(
        {t: np.asarray(c, dtype=np.int64) for t, c in cells.items()},
        np.asarray(points, dtype=float),
    )


def hex_grid(nx=2, ny=2, nz=2, origin=(0.0, 0.0, 0.0)):
    axes = [o + np.arange(n + 1, dtype=float) for o, n in zip(origin, (nx, ny, nz))]
    grid = pv.StructuredGrid(*np.meshgrid(*axes, indexing="ij"))
    return grid.cast_to_unstructured_grid()


def _positive(cell_type, conn, points, flip):
    """Flip cells with negative volume (``flip`` reorders one cell)."""
    conn = np.asarray(conn)
    volume = make_grid({cell_type: conn}, points).compute_cell_sizes()["Volume"]
    conn[volume < 0] = conn[volume < 0][:, flip]
    grid = make_grid({cell_type: conn}, points)
    assert np.all(grid.compute_cell_sizes()["Volume"] > 0)
    return grid


def wedge_grid():
    plane = pv.Plane(i_resolution=2, j_resolution=2).triangulate()
    pts, tris = np.asarray(plane.points, dtype=float), plane.regular_faces
    points = np.vstack([pts, pts + [0.0, 0.0, 1.0]])
    conn = np.hstack([tris, tris + len(pts)])
    return _positive(CT.WEDGE, conn, points, [0, 2, 1, 3, 5, 4])


def pyramid_grid():
    """Six pyramids filling the cube [0, 2]^3, apexes at its centre."""
    cube = pv.Cube(center=(1, 1, 1), x_length=2, y_length=2, z_length=2)
    points = np.vstack([np.asarray(cube.points, dtype=float), [[1.0, 1.0, 1.0]]])
    conn = np.hstack([cube.regular_faces, np.full((6, 1), 8)])
    return _positive(CT.PYRAMID, conn, points, [0, 3, 2, 1, 4])


def mixed_grid():
    points = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1], [0.5, 0.5, -1],
        ],
        dtype=float,
    )
    return make_grid(
        {
            CT.HEXAHEDRON: [[0, 1, 2, 3, 4, 5, 6, 7]],
            CT.TETRA: [[1, 8, 9, 10]],
            CT.WEDGE: [[1, 9, 2, 5, 11, 6]],
            CT.PYRAMID: [[0, 3, 2, 1, 12]],
        },
        points,
    )


def polydata(points, cells, kind="faces"):
    """PolyData from a list of point-id lists (faces or lines)."""
    flat = np.concatenate([[len(c), *c] for c in cells]).astype(np.int64)
    mesh = pv.PolyData(np.asarray(points, dtype=float))
    setattr(mesh, kind, flat)
    return mesh


def faces_of(grid, cell_ids):
    """Surface of the faces (VTK's cell faces, possibly quadratic) of cells."""
    cells, types = [], []
    for i in cell_ids:
        for face in grid.get_cell(i).faces:
            cells.append(face.point_ids)
            types.append(face.type)
    flat = np.concatenate([[len(c), *c] for c in cells])
    return pv.UnstructuredGrid(flat, np.array(types, np.uint8), grid.points)


def original_ids(sub):
    """Original cell ids of the surface of an ``extract_cells`` subset."""
    surf = extract_surface(sub)
    return surf, np.asarray(sub.cell_data["vtkOriginalCellIds"])[
        surf.cell_data["vtkOriginalCellIds"]
    ]


# --------------------------------------------------------------------------- #
# boundary faces of every 3D element type
# --------------------------------------------------------------------------- #
_SOLIDS = {
    "hex": lambda: hex_grid(),
    "voxel": lambda: pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid(),
    "tet": lambda: pv.ImageData(dimensions=(3, 3, 3)).triangulate(),
    "wedge": wedge_grid,
    "pyramid": pyramid_grid,
    "mixed": mixed_grid,
    "tet10": lambda: example_cells.QuadraticTetrahedron(),
    "hex20": lambda: example_cells.QuadraticHexahedron(),
    "hex27": lambda: example_cells.TriQuadraticHexahedron(),
    "wedge15": lambda: example_cells.QuadraticWedge(),
    "pyramid13": lambda: example_cells.QuadraticPyramid(),
}


@pytest.mark.parametrize("name", list(_SOLIDS))
def test_boundary_faces(tmp_path, name):
    grid = _SOLIDS[name]()
    surface = extract_surface(grid)
    assert surface.n_cells > 0
    path = tmp_path / "bnd.exo"
    write_exo(path, grid, surface)
    assert file_side_sets(path) == [(1, "side_set_1", surface.n_cells)]
    (vtk_set,) = vtk_side_sets(path)
    # (VTK's extract_surface(nonlinear_subdivision=0) numbers the faces, not
    # the cells, in vtkOriginalCellIds of quadratic cells: one cell -> 0.)
    owners = surface["vtkOriginalCellIds"] if grid.n_cells > 1 else 0
    assert_side_set(vtk_set, surface, np.broadcast_to(owners, surface.n_cells))


@pytest.mark.parametrize("name", ["tet10", "hex20", "hex27", "wedge15", "pyramid13"])
def test_quadratic_surface_cells_match_by_corners(tmp_path, name):
    grid = _SOLIDS[name]()
    surface = faces_of(grid, [0])  # QUADRATIC_TRIANGLE / QUADRATIC_QUAD / ...
    assert set(surface.celltypes) & {
        CT.QUADRATIC_TRIANGLE, CT.QUADRATIC_QUAD, CT.BIQUADRATIC_QUAD
    }
    write_exo(tmp_path / "q.exo", grid, surface)
    (vtk_set,) = vtk_side_sets(tmp_path / "q.exo")
    assert_side_set(vtk_set, surface, np.zeros(surface.n_cells))
    with open_reader(tmp_path / "q.exo") as reader:
        (ss,) = reader.side_sets()
    assert sorted(ss.sides) == list(range(1, surface.n_cells + 1))  # every side


def test_inward_oriented_boundary_faces_still_match(tmp_path):
    grid = hex_grid()
    surface = extract_surface(grid).flip_faces()
    write_exo(tmp_path / "in.exo", grid, surface)
    (vtk_set,) = vtk_side_sets(tmp_path / "in.exo")
    assert_side_set(vtk_set, surface, surface.cell_data["vtkOriginalCellIds"])


def test_surface_cell_kinds(tmp_path):
    """PolyData quads, a 4-point POLYGON, an UnstructuredGrid of triangles."""
    grid = hex_grid(1, 1, 1)
    tets = pv.ImageData(dimensions=(2, 2, 2)).triangulate()
    pts = np.asarray(grid.points)
    bottom = np.flatnonzero(pts[:, 2] == 0)
    order = bottom[np.argsort(np.arctan2(pts[bottom, 1] - 0.5, pts[bottom, 0] - 0.5))]
    poly = make_grid({CT.POLYGON: [order]}, pts)
    assert poly.celltypes[0] == CT.POLYGON  # a genuine 4-point VTK_POLYGON
    write_exo(tmp_path / "poly.exo", grid, poly)
    assert file_side_sets(tmp_path / "poly.exo") == [(1, "side_set_1", 1)]

    tri_surface = extract_surface(tets).cast_to_unstructured_grid()
    write_exo(tmp_path / "ug.exo", tets, tri_surface)
    (vtk_set,) = vtk_side_sets(tmp_path / "ug.exo")
    assert_side_set(vtk_set, tri_surface, tri_surface.cell_data["vtkOriginalCellIds"])


# --------------------------------------------------------------------------- #
# internal interfaces: ownership by the surface normal
# --------------------------------------------------------------------------- #
def two_region_hexes(merge=True):
    left, right = hex_grid(2, 2, 2), hex_grid(2, 2, 2, origin=(2.0, 0.0, 0.0))
    grid = pv.merge([left, right], merge_points=merge)
    grid.cell_data["region"] = np.repeat([1, 2], 8)
    return grid


@pytest.mark.parametrize("merge", [True, False], ids=["shared", "duplicate-nodes"])
@pytest.mark.parametrize("region", [1, 2])
def test_region_surface_belongs_to_region(tmp_path, region, merge):
    grid = two_region_hexes(merge)
    sub = grid.extract_cells(np.flatnonzero(grid.cell_data["region"] == region))
    surface, owners = original_ids(sub)
    surface.cell_data.remove("region")  # one set
    path = tmp_path / "iface.exo"
    write_exo(path, grid, surface)
    (vtk_set,) = vtk_side_sets(path)
    assert_side_set(vtk_set, surface, owners)
    assert set(grid.cell_data["region"][owners]) == {region}
    # 4 of the faces lie on the interface x == 2
    on_interface = [all(p[0] == 2 for p in c) for c in corner_sets(surface)]
    assert sum(on_interface) == 4


def test_flipped_interface_faces_change_owner(tmp_path):
    grid = two_region_hexes()
    sub = grid.extract_cells(np.flatnonzero(grid.cell_data["region"] == 1))
    surface, owners = original_ids(sub)
    interface = np.array([all(p[0] == 2 for p in c) for c in corner_sets(surface)])
    faces = extract_surface(surface.extract_cells(np.flatnonzero(interface)))
    write_exo(tmp_path / "a.exo", grid, faces)
    (vtk_set,) = vtk_side_sets(tmp_path / "a.exo")
    assert set(grid.cell_data["region"][vtk_set.cell_data["SourceElementId"]]) == {1}

    write_exo(tmp_path / "b.exo", grid, faces.flip_faces())
    (vtk_set,) = vtk_side_sets(tmp_path / "b.exo")
    assert set(grid.cell_data["region"][vtk_set.cell_data["SourceElementId"]]) == {2}
    assert corner_sets(vtk_set) == corner_sets(faces)


def test_tie_goes_to_lower_block_id(tmp_path):
    """Two coincident elements: both have the face 'pointing out'."""
    one = hex_grid(1, 1, 1)
    grid = pv.merge([one, one], merge_points=True)
    grid.cell_data["region"] = [7, 3]  # the copy in region 3 comes second
    face = extract_surface(extract_surface(one).extract_cells([0]))
    write_exo(tmp_path / "tie.exo", grid, face)
    with open_reader(tmp_path / "tie.exo") as reader:
        blocks = reader.blocks()
        (ss,) = reader.side_sets()
    assert [b.id for b in blocks] == [3, 7]
    assert ss.elems.tolist() == [0]  # first element of block 3
    (vtk_set,) = vtk_side_sets(tmp_path / "tie.exo")
    assert vtk_set.cell_data["SourceElementId"].tolist() == [1]


# --------------------------------------------------------------------------- #
# ids, names, duplicates
# --------------------------------------------------------------------------- #
def boundary_with_regions(values):
    grid = hex_grid()
    surface = extract_surface(grid)
    normals = surface.compute_normals(cell_normals=True, point_normals=False)
    axis = np.argmax(np.abs(normals.cell_data["Normals"]), axis=1)
    surface.cell_data["region"] = np.asarray(values, dtype=object)[axis].astype(
        type(values[0])
    )
    return grid, surface


def test_side_set_ids_from_integer_regions(tmp_path):
    grid, surface = boundary_with_regions([9, 5, 9])  # x faces 9, y faces 5
    write_exo(tmp_path / "i.exo", grid, surface)
    assert file_side_sets(tmp_path / "i.exo") == [
        (5, "side_set_5", 8), (9, "side_set_9", 16)
    ]
    sets = vtk_side_sets(tmp_path / "i.exo")
    assert sets.keys() == ["side_set_5", "side_set_9"]
    for set_id, vtk_set in zip((5, 9), sets):
        members = np.flatnonzero(surface["region"] == set_id)
        part = surface.extract_cells(members)
        assert_side_set(vtk_set, part, surface["vtkOriginalCellIds"][members])


def test_side_set_names_and_string_regions(tmp_path):
    grid, surface = boundary_with_regions(["walls", "walls", "floor"])
    write_exo(tmp_path / "s.exo", grid, surface)
    assert file_side_sets(tmp_path / "s.exo") == [(1, "floor", 8), (2, "walls", 16)]
    write_exo(tmp_path / "d.exo", grid, surface, side_set_names={"floor": "bottom"})
    assert file_side_sets(tmp_path / "d.exo") == [(1, "bottom", 8), (2, "walls", 16)]
    assert vtk_side_sets(tmp_path / "d.exo").keys() == ["bottom", "walls"]

    grid, surface = boundary_with_regions([0, 1, 1])  # 0 -> sequential ids
    write_exo(tmp_path / "l.exo", grid, surface, side_set_names=["x", "yz"])
    assert file_side_sets(tmp_path / "l.exo") == [(1, "x", 8), (2, "yz", 16)]
    write_exo(tmp_path / "n.exo", grid, surface)
    assert file_side_sets(tmp_path / "n.exo") == [
        (1, "side_set_1", 8), (2, "side_set_2", 16)
    ]
    write_exo(tmp_path / "k.exo", grid, surface, region_key="other")  # absent
    assert file_side_sets(tmp_path / "k.exo") == [(1, "side_set_1", 24)]

    for names, match in ((["a"], "1 name"), (["a", "a"], "unique"),
                         ({5: "x"}, "not a region value")):
        with pytest.raises(ValueError, match=match):
            write_exo(tmp_path / "x.exo", grid, surface, side_set_names=names)
    assert not (tmp_path / "x.exo").exists()


def test_duplicate_faces_are_dropped_once(tmp_path):
    grid = hex_grid(1, 1, 1)
    surface = extract_surface(grid)
    doubled = pv.merge([surface, surface.flip_faces(), surface], merge_points=True)
    with pytest.warns(UserWarning, match="repeated") as record:
        write_exo(tmp_path / "dup.exo", grid, doubled)
    assert len([w for w in record if "repeated" in str(w.message)]) == 1
    (vtk_set,) = vtk_side_sets(tmp_path / "dup.exo")
    assert_side_set(vtk_set, surface, np.zeros(6))  # first occurrences, in order


def test_same_face_in_two_sets_is_kept(tmp_path):
    grid = hex_grid(1, 1, 1)
    surface = extract_surface(grid)
    both = pv.merge([surface, surface], merge_points=True)
    both.cell_data["region"] = np.repeat([1, 2], 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(tmp_path / "two.exo", grid, both)
    assert file_side_sets(tmp_path / "two.exo") == [
        (1, "side_set_1", 6), (2, "side_set_2", 6)
    ]


def test_empty_surface_writes_no_side_sets(tmp_path):
    with pytest.warns(UserWarning, match="no cells"):
        write_exo(tmp_path / "e.exo", hex_grid(), pv.PolyData())
    assert file_side_sets(tmp_path / "e.exo") == []


# --------------------------------------------------------------------------- #
# surfaces that do not match
# --------------------------------------------------------------------------- #
def test_far_surface_raises(tmp_path):
    with pytest.raises(ValueError, match=r"farther than side_set_tolerance.*largest"):
        far = pv.Sphere(radius=10, center=(50, 0, 0))
        write_exo(tmp_path / "f.exo", hex_grid(), far)
    assert os.listdir(tmp_path) == []


def test_faces_that_are_not_element_sides_raise(tmp_path):
    grid = hex_grid(1, 1, 1)
    triangles = extract_surface(grid).triangulate()  # corners on nodes, wrong faces
    with pytest.raises(ValueError, match="12 of 12 surface cell"):
        write_exo(tmp_path / "t.exo", grid, triangles)
    edges = extract_surface(grid).extract_all_edges()  # lines on a 3D mesh
    with pytest.raises(ValueError, match="faces of the 3D elements"):
        write_exo(tmp_path / "e.exo", grid, edges)
    assert os.listdir(tmp_path) == []


def test_shells_are_not_side_set_targets(tmp_path):
    shells = pv.Sphere(theta_resolution=8, phi_resolution=8)
    with pytest.raises(ValueError, match="shell, bar and sphere"):
        write_exo(tmp_path / "s.exo", shells, shells)


@pytest.mark.parametrize(
    "bad",
    [
        lambda g: pv.PolyData(np.asarray(g.points[:3])),  # vertices
        lambda g: g,  # 3D cells
        lambda g: extract_surface(g).triangulate().strip(),  # triangle strips
        lambda g: polydata(np.asarray(g.points), [[0, 1, 3, 2, 6]]),  # pentagon
    ],
    ids=["vertices", "solids", "strips", "pentagon"],
)
def test_unsupported_surface_cells(tmp_path, bad):
    grid = hex_grid(1, 1, 1)
    with pytest.raises(ValueError, match="surface cells must be faces"):
        write_exo(tmp_path / "x.exo", grid, bad(grid))


def test_side_set_tolerance(tmp_path):
    grid = hex_grid()
    surface = extract_surface(grid)
    shifted = surface.translate((1e-4, 0, 0))
    with pytest.raises(ValueError, match="farther than"):
        write_exo(tmp_path / "x.exo", grid, shifted)  # default 1e-6 * diagonal
    write_exo(tmp_path / "t.exo", grid, shifted, side_set_tolerance=1e-3)
    (vtk_set,) = vtk_side_sets(tmp_path / "t.exo")
    assert corner_sets(vtk_set) == corner_sets(surface)  # snapped to the nodes
    write_exo(tmp_path / "z.exo", grid, surface, side_set_tolerance=0)  # exact
    for bad in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="side_set_tolerance"):
            write_exo(tmp_path / "x.exo", grid, surface, side_set_tolerance=bad)
    assert not (tmp_path / "x.exo").exists()


# --------------------------------------------------------------------------- #
# performance sanity check (the full ~1M-cell budget is measured separately)
# --------------------------------------------------------------------------- #
def test_matching_scales(tmp_path):
    grid = pv.ImageData(dimensions=(41, 41, 41)).cast_to_unstructured_grid()
    surface = extract_surface(grid)
    import time

    start = time.perf_counter()
    write_exo(tmp_path / "big.exo", grid, surface)
    assert time.perf_counter() - start < 10
    with open_reader(tmp_path / "big.exo") as reader:
        (ss,) = reader.side_sets()
    assert len(ss.elems) == surface.n_cells == 6 * 40 * 40
