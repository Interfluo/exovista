"""2D meshes (num_dim=2) and their edge side sets, checked with VTK's reader.

Regression tests for exovista 1.x, which wrote 2D meshes with num_dim=3 (VTK
then read triangle edge sides as faces) and mixed up closed boundary loops,
where the surface has as many points as cells (n_points == n_cells), when it
attached helper arrays with ``surface[name] = ...``.
"""

import numpy as np
import pytest
import pyvista as pv

from _pvcompat import example_cell, exodus_reader
from exovista._backend import open_reader
from exovista.write_pyvista import write_exo

CT = pv.CellType


def vtk_read(path):
    reader = exodus_reader(str(path))
    reader.side_sets.enable_all()
    reader.reader.GenerateGlobalElementIdArrayOn()
    return reader.read()


def file_info(path):
    with open_reader(path) as reader:
        return (
            reader.num_dim,
            [b.elem_type for b in reader.blocks()],
            [(s.id, s.name, len(s.elems)) for s in reader.side_sets()],
        )


def lines(points, edges, region=None):
    """Line PolyData (``pv.PolyData(points)`` alone would add vertex cells)."""
    edges = np.asarray(edges, dtype=np.int64)
    flat = np.hstack([np.full((len(edges), 1), 2), edges]).ravel()
    mesh = pv.PolyData(np.asarray(points, dtype=float), lines=flat)
    if region is not None:
        mesh.cell_data["region"] = region
    return mesh


def boundary_edges(mesh):
    """Boundary edges of a 2D mesh as a line PolyData (unordered, unoriented)."""
    return mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False,
        non_manifold_edges=False,
    )


def edge_points(mesh):
    """Per line cell: its two end points (2, 3), in cell order."""
    pts = np.asarray(mesh.points)
    return np.array([pts[mesh.get_cell(i).point_ids[:2]] for i in range(mesh.n_cells)])


def assert_edges(vtk_set, surface, grid, quadratic=False):
    """VTK side cells are the surface edges and lie on their owning cells."""
    # VTK (<= 9.7) labels the 3-node edge sides of TRI6/QUAD8/QUAD9 elements
    # VTK_TRIANGLE (nodes: corner, corner, mid-edge) instead of a quadratic edge.
    expected = {CT.TRIANGLE} if quadratic else {CT.LINE}
    assert set(vtk_set.celltypes) == expected
    got = edge_points(vtk_set)
    want = edge_points(surface)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert np.allclose(g, w, atol=1e-6) or np.allclose(g, w[::-1], atol=1e-6)
    for g, owner in zip(got, vtk_set.cell_data["SourceElementId"]):
        cell_pts = np.asarray(grid.points)[grid.get_cell(owner).point_ids]
        for p in g:
            assert np.min(np.linalg.norm(cell_pts - p, axis=1)) < 1e-6


def quad_grid(nx, ny):
    image = pv.ImageData(dimensions=(nx + 1, ny + 1, 1))
    pixels = image.cast_to_unstructured_grid()
    conn = pixels.cell_connectivity.reshape(-1, 4)[:, [0, 1, 3, 2]]  # CCW quads
    return pv.UnstructuredGrid({CT.QUAD: conn}, np.asarray(image.points, float))


def annulus(n_theta=16, n_r=2, r0=1.0, r1=2.0):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    radii = np.linspace(r0, r1, n_r + 1)
    r, t = np.meshgrid(radii, theta, indexing="ij")
    points = np.c_[(r * np.cos(t)).ravel(), (r * np.sin(t)).ravel(), np.zeros(r.size)]
    i, j = np.meshgrid(np.arange(n_r), np.arange(n_theta), indexing="ij")
    a = (i * n_theta + j).ravel()
    b = (i * n_theta + (j + 1) % n_theta).ravel()
    conn = np.c_[a, a + n_theta, b + n_theta, b]  # CCW
    return pv.UnstructuredGrid({CT.QUAD: conn}, points)


# --------------------------------------------------------------------------- #
def test_quad_mesh_with_one_edge(tmp_path):
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    volume = pv.UnstructuredGrid({CT.QUAD: np.array([[0, 1, 2, 3]])}, points)
    volume.cell_data["region"] = [1]
    surface = lines(points, [[1, 2]], region=[1])
    path = tmp_path / "quad.exo"
    write_exo(path, volume, surface)
    assert file_info(path) == (2, ["QUAD4"], [(1, "side_set_1", 1)])
    with open_reader(path) as reader:
        (ss,) = reader.side_sets()
    assert (ss.elems.tolist(), ss.sides.tolist()) == ([0], [2])
    mb = vtk_read(path)
    assert mb["Element Blocks"][0].n_cells == 1
    assert_edges(mb["Side Sets"][0], surface, volume)


def test_closed_boundary_loop_split_into_regions(tmp_path):
    grid = quad_grid(4, 4)
    surface = boundary_edges(grid)
    assert surface.n_points == surface.n_cells == 16  # the 1.x trigger
    mid = np.asarray(surface.cell_centers().points)
    surface.cell_data["region"] = np.where((mid[:, 1] == 0) | (mid[:, 0] == 4), 1, 2)
    path = tmp_path / "loop.exo"
    write_exo(path, grid, surface, side_set_names=["bottom_right", "top_left"])
    assert file_info(path) == (
        2, ["QUAD4"], [(1, "bottom_right", 8), (2, "top_left", 8)]
    )
    sets = vtk_read(path)["Side Sets"]
    for name, region in (("bottom_right", 1), ("top_left", 2)):
        part = surface.extract_cells(np.flatnonzero(surface["region"] == region))
        assert_edges(sets[name], part, grid)
    pts = np.asarray(sets["bottom_right"].points)
    assert np.all((pts[:, 1] == 0) | (pts[:, 0] == 4))


def test_annulus_inner_and_outer_boundaries(tmp_path):
    grid = annulus()
    surface = boundary_edges(grid)
    assert surface.n_points == surface.n_cells == 32
    radius = np.linalg.norm(np.asarray(surface.cell_centers().points), axis=1)
    surface.cell_data["region"] = np.where(radius < 1.5, "inner", "outer")
    path = tmp_path / "annulus.exo"
    write_exo(path, grid, surface)
    assert file_info(path) == (2, ["QUAD4"], [(1, "inner", 16), (2, "outer", 16)])
    sets = vtk_read(path)["Side Sets"]
    for name, r in (("inner", 1.0), ("outer", 2.0)):
        pts = np.asarray(sets[name].points)
        np.testing.assert_allclose(np.linalg.norm(pts, axis=1), r, rtol=1e-6)
        part = surface.extract_cells(np.flatnonzero(surface["region"] == name))
        assert_edges(sets[name], part, grid)


def test_triangle_edges_read_as_lines(tmp_path):
    grid = pv.Plane(i_resolution=3, j_resolution=2).triangulate()
    surface = boundary_edges(grid)
    path = tmp_path / "tri.exo"
    write_exo(path, grid, surface)
    num_dim, types, sets = file_info(path)
    assert (num_dim, types) == (2, ["TRI3"])
    assert sets == [(1, "side_set_1", surface.n_cells)]
    (vtk_set,) = vtk_read(path)["Side Sets"]
    assert set(vtk_set.celltypes) == {CT.LINE}
    assert_edges(vtk_set, surface, grid.cast_to_unstructured_grid())


def test_pixels(tmp_path):
    image = pv.ImageData(dimensions=(5, 4, 1))
    surface = boundary_edges(image)
    write_exo(tmp_path / "px.exo", image, surface)
    assert file_info(tmp_path / "px.exo") == (2, ["QUAD4"], [(1, "side_set_1", 14)])
    (vtk_set,) = vtk_read(tmp_path / "px.exo")["Side Sets"]
    assert_edges(vtk_set, surface, image.cast_to_unstructured_grid())


@pytest.mark.parametrize("example", ["QuadraticTriangle", "QuadraticQuadrilateral",
                                     "BiQuadraticQuadrilateral"])
@pytest.mark.parametrize("quadratic_edges", [True, False])
def test_quadratic_elements(tmp_path, example, quadratic_edges):
    grid = example_cell(example)
    grid.points[:, 2] = 0.0
    cell = grid.get_cell(0)
    if quadratic_edges:
        edges = [e.point_ids for e in cell.edges]  # QUADRATIC_EDGE, 3 points
        surface = pv.UnstructuredGrid(
            np.hstack([[3, *e] for e in edges]),
            np.full(len(edges), CT.QUADRATIC_EDGE, np.uint8),
            grid.points,
        )
    else:
        surface = lines(grid.points, [e.point_ids[:2] for e in cell.edges])
    path = tmp_path / "q.exo"
    write_exo(path, grid, surface)
    num_dim, _, sets = file_info(path)
    assert num_dim == 2 and sets == [(1, "side_set_1", len(cell.edges))]
    with open_reader(path) as reader:
        (ss,) = reader.side_sets()
    assert ss.sides.tolist() == list(range(1, len(cell.edges) + 1))
    (vtk_set,) = vtk_read(path)["Side Sets"]
    assert_edges(vtk_set, surface, grid, quadratic=True)


def _oriented_boundary(grid, cells):
    """Boundary edges of ``cells`` oriented like the (CCW) cells."""
    conn = grid.cell_connectivity.reshape(-1, 4)[cells]
    edges = np.stack([conn, np.roll(conn, -1, axis=1)], axis=-1).reshape(-1, 2)
    key = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True,
                                   return_counts=True)
    return edges[counts[inverse.ravel()] == 1]


@pytest.mark.parametrize("region", [1, 2])
def test_interface_edges_owned_by_normal(tmp_path, region):
    grid = quad_grid(4, 2)
    centers = np.asarray(grid.cell_centers().points)
    grid.cell_data["region"] = np.where(centers[:, 0] < 2, 1, 2)
    cells = np.flatnonzero(grid.cell_data["region"] == region)
    edges = _oriented_boundary(grid, cells)
    surface = lines(grid.points, edges)
    write_exo(tmp_path / "a.exo", grid, surface)
    (vtk_set,) = vtk_read(tmp_path / "a.exo")["Side Sets"]
    owners = vtk_set.cell_data["SourceElementId"]
    assert set(grid.cell_data["region"][owners]) == {region}
    assert_edges(vtk_set, surface, grid)

    # reversed edges: the interface edges now belong to the other region
    reversed_surface = lines(grid.points, edges[:, ::-1])
    write_exo(tmp_path / "b.exo", grid, reversed_surface)
    (vtk_set,) = vtk_read(tmp_path / "b.exo")["Side Sets"]
    owners = vtk_set.cell_data["SourceElementId"]
    on_interface = np.all(np.asarray(grid.points)[edges][:, :, 0] == 2, axis=1)
    assert set(grid.cell_data["region"][owners[on_interface]]) == {3 - region}
    assert set(grid.cell_data["region"][owners[~on_interface]]) == {region}


def test_2d_mesh_forced_3d_has_shells_without_side_sets(tmp_path):
    grid = quad_grid(2, 2)
    write_exo(tmp_path / "s.exo", grid, num_dim=3)
    assert file_info(tmp_path / "s.exo") == (3, ["SHELL4"], [])
    with pytest.raises(ValueError, match="shell, bar and sphere"):
        write_exo(tmp_path / "x.exo", grid, boundary_edges(grid), num_dim=3)
    assert not (tmp_path / "x.exo").exists()


def test_faces_on_a_2d_mesh_raise(tmp_path):
    grid = quad_grid(2, 2)
    faces = grid.cast_to_unstructured_grid()  # the quads themselves
    with pytest.raises(ValueError, match="edges of the 2D elements"):
        write_exo(tmp_path / "x.exo", grid, faces)
