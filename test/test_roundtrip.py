"""End-to-end round trips: ``write_exo`` -> ``read_exo`` and the oracles.

Every mesh of the corpus below is written once with regions, block names,
static point and cell arrays (vectors and tensors included), node sets, time
histories (node, element and global) and, where its elements have sides,
side sets from a surface. The file is then checked three ways:

* ``read_exo`` & co. must return the input *exactly*: float64 points, cell
  types and connectivity in the input cell order (``original_order``), the
  arrays (components recombined), block ids / names from the regions, side
  sets (the surface faces, as sorted corner-node keys), node sets and every
  step of every history;
* VTK's ``pv.ExodusIIReader`` (the independent oracle) must read the same
  mesh, variables, times and sets (VTK stores points as float32, hence a
  tolerance on coordinates only);
* meshio must read the meshes whose element types it knows.

``write_exo(read_exo(file))`` round trips of the files in ``test/data``
close the loop the other way.
"""

import collections
import glob
import inspect
import os
import warnings
from dataclasses import dataclass, field

import meshio
import netCDF4
import numpy as np
import pytest
import pyvista as pv
from pyvista import examples
from scipy.spatial import cKDTree
from meshio.exodus._exodus import categorize as meshio_categorize
from vtkmodules.vtkIOExodus import vtkExodusIIReader

from exovista import (
    read_element_fields,
    read_exo,
    read_global_fields,
    read_node_fields,
    read_node_sets,
    read_side_sets,
    write_exo,
)
from exovista._backend import open_reader
from exovista._model import component_suffixes

CT = pv.CellType
DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

# VTK cells written with a different node order than they read back.
VOXEL_TO_HEX = [0, 1, 3, 2, 4, 5, 7, 6]
PIXEL_TO_QUAD = [0, 1, 3, 2]
# Corner nodes of side cells (VTK's side sets list mid-side nodes after them).
N_CORNERS = {
    CT.VERTEX: 1,
    CT.LINE: 2,
    CT.QUADRATIC_EDGE: 2,
    CT.TRIANGLE: 3,
    CT.QUADRATIC_TRIANGLE: 3,
    CT.QUAD: 4,
    CT.PIXEL: 4,
    CT.QUADRATIC_QUAD: 4,
    CT.BIQUADRATIC_QUAD: 4,
}
# meshio cell type of each VTK cell type exovista writes for these meshes.
MESHIO_TYPE = {
    CT.TETRA: "tetra",
    CT.HEXAHEDRON: "hexahedron",
    CT.WEDGE: "wedge",
    CT.PYRAMID: "pyramid",
    CT.QUAD: "quad",
    CT.TRIANGLE: "triangle",
    CT.QUADRATIC_TETRA: "tetra10",
    CT.QUADRATIC_HEXAHEDRON: "hexahedron20",
}
LINEAR = {CT.TETRA, CT.HEXAHEDRON, CT.WEDGE, CT.PYRAMID, CT.QUAD, CT.TRIANGLE}
# Bookkeeping arrays write_exo never writes.
NOT_WRITTEN = {"vtkOriginalPointIds", "vtkOriginalCellIds", "vtkGhostType"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def extract_surface(mesh, **kwargs):
    """``mesh.extract_surface`` without pyvista's FutureWarning on ``algorithm``."""
    if "algorithm" in inspect.signature(mesh.extract_surface).parameters:
        kwargs.setdefault("algorithm", "dataset_surface")
    else:
        kwargs.pop("algorithm", None)
    return mesh.extract_surface(**kwargs)


def boundary_edges(mesh):
    return mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False,
        non_manifold_edges=False,
    )


def offsets_of(grid):
    """Cell offsets (n_cells + 1) of an UnstructuredGrid, any pyvista version."""
    if hasattr(type(grid), "cell_offsets"):
        return np.asarray(grid.cell_offsets)
    return np.asarray(grid.offset)


def cell_ids(mesh):
    """Point ids of every cell, as a list of int arrays."""
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    offsets = offsets_of(mesh)
    conn = np.asarray(mesh.cell_connectivity)
    return [conn[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def map_points(points, targets):
    """Index of the ``targets`` point at every point of ``points`` (exact)."""
    dist, idx = cKDTree(np.asarray(targets, float)).query(np.asarray(points, float))
    assert np.all(dist <= 1e-9 * max(1.0, np.abs(targets).max())), dist.max()
    return idx


def side_keys(mesh, node_of_point=None):
    """Sorted corner-node key of every cell of a side mesh, in cell order."""
    ug = mesh if isinstance(mesh, pv.UnstructuredGrid) else (
        mesh.cast_to_unstructured_grid())
    keys = []
    for i, ids in enumerate(cell_ids(ug)):
        n = N_CORNERS.get(int(ug.celltypes[i]), len(ids))
        nodes = ids[:n] if node_of_point is None else node_of_point[ids[:n]]
        keys.append(tuple(sorted(int(v) for v in nodes)))
    return keys


def expected_cells(grid):
    """(cell types, offsets, connectivity) read_exo returns for ``grid``."""
    types = np.asarray(grid.celltypes).copy()
    offsets = offsets_of(grid)
    conn = np.asarray(grid.cell_connectivity).copy()
    for vtk_type, native, perm in (
        (CT.VOXEL, CT.HEXAHEDRON, VOXEL_TO_HEX),
        (CT.PIXEL, CT.QUAD, PIXEL_TO_QUAD),
    ):
        cells = np.flatnonzero(types == vtk_type)
        if cells.size:
            idx = offsets[cells][:, None] + np.arange(len(perm))
            conn[idx] = conn[idx][:, perm]
            types[cells] = native
    return types, offsets, conn


def cell_point_coords(grid, cells, conn_types=None):
    """``(len(cells), k, 3)`` coordinates of same-sized cells (VTK order)."""
    types, offsets, conn = conn_types or expected_cells(grid)
    k = int(offsets[cells[0] + 1] - offsets[cells[0]])
    idx = conn[offsets[cells][:, None] + np.arange(k)]
    return np.asarray(grid.points, dtype=float)[idx]


def as_columns(values):
    """``(n,)`` stays, ``(n, k)`` stays, anything else becomes ``(n, k)``."""
    arr = np.asarray(values, dtype=np.float64)
    return arr if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)


def vtk_array(data, name, n_comp):
    """Array ``name`` with ``n_comp`` components as VTK's reader stores it.

    VTK regroups ``name_X, name_Y[, name_Z]`` into ``name_``, reads the 9
    full-tensor components as three row vectors ``name_X/_Y/_Z`` and leaves
    the symmetric tensor and other counts as scalars.
    """
    if n_comp == 1:
        return np.asarray(data[name], dtype=np.float64)
    if n_comp in (2, 3):
        values = np.asarray(data[name + "_"], dtype=np.float64)
        if n_comp == 2 and values.shape[1] == 3:  # 2D files: z padded with 0
            assert not values[:, 2].any()
            values = values[:, :2]
        return values
    if n_comp == 9:
        return np.hstack([np.asarray(data[name + s]) for s in ("_X", "_Y", "_Z")])
    suffixes = (
        ("_XX", "_YY", "_ZZ", "_XY", "_YZ", "_XZ") if n_comp == 6
        else tuple(f"_{i}" for i in range(n_comp))
    )
    return np.column_stack([np.asarray(data[name + s]) for s in suffixes])


def vtk_read(path, time_index=None):
    reader = pv.ExodusIIReader(str(path))
    reader.enable_all_point_arrays()
    reader.enable_all_cell_arrays()
    reader.reader.SetAllArrayStatus(vtkExodusIIReader.GLOBAL, 1)
    reader.side_sets.enable_all()
    reader.node_sets.enable_all()
    reader.reader.GenerateGlobalElementIdArrayOn()
    reader.reader.GenerateGlobalNodeIdArrayOn()
    reader.reader.ApplyDisplacementsOff()  # "displacement" would move points
    if time_index is not None:
        reader.set_active_time_point(time_index)
    return reader, reader.read()


# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """A mesh, everything written with it, and what must be read back."""

    volume: object  # what write_exo gets
    expected: pv.UnstructuredGrid  # the mesh read_exo returns (input order)
    num_dim: int
    block_ids: np.ndarray  # per cell
    block_names: np.ndarray  # per cell
    write_kwargs: dict = field(default_factory=dict)
    side_sets: dict = field(default_factory=dict)  # name -> list of node keys
    owner_regions: dict = field(default_factory=dict)  # name -> region values
    regions: np.ndarray | None = None  # per cell (for owner_regions)
    node_sets: dict = field(default_factory=dict)
    times: np.ndarray = None
    node_fields: dict = field(default_factory=dict)
    element_fields: dict = field(default_factory=dict)
    global_fields: dict = field(default_factory=dict)
    meshio: bool = True
    vtk_atol: float = 0.0  # VTK coordinates: float32 rounding allowance

    @property
    def n_steps(self):
        return len(self.times)

    def point_arrays(self):
        """Static point arrays written (and read back), float64."""
        return {
            k: as_columns(v) for k, v in self.expected.point_data.items()
            if k not in NOT_WRITTEN and k not in self.node_fields
        }

    def cell_arrays(self):
        region_key = self.write_kwargs.get("region_key", "region")
        return {
            k: as_columns(v) for k, v in self.expected.cell_data.items()
            if k not in NOT_WRITTEN and k != region_key
            and k not in self.element_fields
        }

    def node_history(self):
        """Every node variable as ``(n_steps, n_points[, k])``."""
        out = {
            k: np.broadcast_to(v, (self.n_steps, *v.shape))
            for k, v in self.point_arrays().items()
        }
        out.update(self.node_fields)
        return out

    def element_history(self):
        out = {
            k: np.broadcast_to(v, (self.n_steps, *v.shape))
            for k, v in self.cell_arrays().items()
        }
        out.update(self.element_fields)
        return out

    def write(self, path):
        kwargs = dict(self.write_kwargs)
        if self.node_sets:
            kwargs["node_sets"] = self.node_sets
        write_exo(
            path, self.volume, times=self.times, node_fields=self.node_fields,
            element_fields=self.element_fields, global_fields=self.global_fields,
            **kwargs,
        )


def _histories(case, seed, n_steps=3, vector=3, elem_comp=6):
    """Random float64 histories (not representable in float32)."""
    rng = np.random.default_rng(seed)
    n_p, n_c = case.expected.n_points, case.expected.n_cells
    case.times = np.cumsum(rng.uniform(0.1, 1.0, n_steps))
    case.node_fields = {
        "temperature": rng.standard_normal((n_steps, n_p)),
        "displacement": rng.standard_normal((n_steps, n_p, vector)),
    }
    case.element_fields = {
        "pressure": rng.standard_normal((n_steps, n_c)),
        "stress": rng.standard_normal((n_steps, n_c, elem_comp)),
    }
    case.global_fields = {
        "energy": rng.standard_normal(n_steps),
        "momentum": rng.standard_normal(n_steps),
    }
    return case


def _regions_by(values, names):
    """Per-cell block ids / names for region ``values`` (ids = values)."""
    values = np.asarray(values)
    return values.copy(), np.array([names[v] for v in values.tolist()])


def _surface_keys_by_region(surface, volume_points, regions, names):
    node_of = map_points(surface.points, volume_points)
    keys = side_keys(surface, node_of)
    out = collections.defaultdict(list)
    for key, region in zip(keys, np.asarray(regions).tolist()):
        out[names[region]].append(key)
    return dict(out)


def sphere_tets():
    vol = pv.Sphere(theta_resolution=10, phi_resolution=10).delaunay_3d()
    vol.cell_data["region"] = 1 + (vol.cell_centers().points[:, 0] > 0)
    names = {1: "west", 2: "east"}
    surf = extract_surface(vol)
    surf.cell_data["region"] = 1 + (surf.cell_centers().points[:, 2] > 0)
    ids, bnames = _regions_by(vol.cell_data["region"], names)
    pts = np.asarray(vol.points)
    case = Case(
        vol, vol, 3, ids, bnames,
        write_kwargs=dict(surface=surf, block_names=["west", "east"],
                          side_set_names={1: "south", 2: "north"}),
        side_sets=_surface_keys_by_region(
            surf, pts, surf.cell_data["region"], {1: "south", 2: "north"}),
        node_sets={"pole": np.flatnonzero(pts[:, 2] == pts[:, 2].max()),
                   "equator": np.flatnonzero(np.abs(pts[:, 2]) < 0.05)},
    )
    return _histories(case, 1)


def cube_tets():
    vol = pv.Cube().delaunay_3d()
    vol.cell_data["velocity"] = np.arange(vol.n_cells * 3.0).reshape(-1, 3) / 7
    surf = extract_surface(vol)
    case = Case(
        vol, vol, 3, np.ones(vol.n_cells, int), np.full(vol.n_cells, "block_1"),
        write_kwargs=dict(surface=surf),
        side_sets={"side_set_1": side_keys(
            surf, map_points(surf.points, vol.points))},
    )
    return _histories(case, 2, elem_comp=9)


def plane_2d():
    plane = pv.Plane(i_resolution=4, j_resolution=3)
    ug = plane.cast_to_unstructured_grid()
    centers = ug.cell_centers().points
    region = np.where(centers[:, 0] < 0, 3, 7)
    plane.cell_data["region"] = region
    ug.cell_data["region"] = region
    edges = boundary_edges(plane)
    edge_region = np.where(np.isclose(edges.cell_centers().points[:, 0], -0.5), 1, 2)
    edges.cell_data["region"] = edge_region
    ids, names = _regions_by(region, {3: "block_3", 7: "block_7"})
    case = Case(
        plane, ug, 2, ids, names,
        write_kwargs=dict(surface=edges, side_set_names={1: "left", 2: "others"}),
        side_sets=_surface_keys_by_region(
            edges, ug.points, edge_region, {1: "left", 2: "others"}),
        node_sets={"corner": [0], "top": np.flatnonzero(
            ug.points[:, 1] == ug.points[:, 1].max())},
    )
    return _histories(case, 3, vector=2, elem_comp=4)


def hexbeam():
    vol = examples.load_hexbeam()
    z = vol.cell_centers().points[:, 2]
    vol.cell_data["region"] = np.where(z < 2.5, 10, 20)
    surf = extract_surface(vol)
    zc = surf.cell_centers().points[:, 2]
    surf.cell_data["region"] = np.where(
        zc == 0, "fixed", np.where(zc == 5, "loaded", "free"))
    names = {10: "root", 20: "tip"}
    ids, bnames = _regions_by(vol.cell_data["region"], names)
    case = Case(
        vol, vol, 3, ids, bnames,
        write_kwargs=dict(surface=surf, block_names=names),
        side_sets=_surface_keys_by_region(
            surf, vol.points, surf.cell_data["region"],
            {s: s for s in ("fixed", "free", "loaded")}),
        node_sets={"fixed_nodes": np.flatnonzero(vol.points[:, 2] == 0)},
    )
    return _histories(case, 4)


def voxels():
    image = pv.ImageData(dimensions=(4, 3, 3), spacing=(0.5, 0.25, 1.0),
                         origin=(1.0, 2.0, 3.0))
    image.point_data["phi"] = np.arange(image.n_points) * 0.125
    image.cell_data["flux"] = np.arange(image.n_cells * 3.0).reshape(-1, 3)
    ug = image.cast_to_unstructured_grid()
    surf = extract_surface(image)
    case = Case(
        image, ug, 3, np.ones(ug.n_cells, int), np.full(ug.n_cells, "block_1"),
        write_kwargs=dict(surface=surf),
        side_sets={"side_set_1": side_keys(surf, map_points(surf.points, ug.points))},
        node_sets={"node_set_1": np.arange(4)},
    )
    return _histories(case, 5)


def far_from_origin():
    """Coordinates, times and values that float32 cannot hold (1.x lost them)."""
    x = 1e6 + 0.123456789 + 1e-3 * np.arange(4)
    y = -2e5 - 0.987654321 + 2e-3 * np.arange(3)
    z = 0.5 + 1e-3 * np.arange(3)
    vol = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing="ij"))
    vol = vol.cast_to_unstructured_grid()
    vol.point_data["big"] = 16777217 + np.arange(vol.n_points)
    case = Case(
        vol, vol, 3, np.ones(vol.n_cells, int), np.full(vol.n_cells, "block_1"),
        vtk_atol=4 * np.finfo(np.float32).eps * 1e6,
    )
    _histories(case, 6, n_steps=2)
    case.times = np.array([1000.000000001, 1000.000000002])
    return case


def _lattice(nx, ny, nz, h=0.5):
    x, y, z = (h * np.arange(n + 1) for n in (nx, ny, nz))
    pts = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1).reshape(-1, 3)

    def pid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    def hexa(i, j, k):
        return [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k),
                pid(i, j + 1, k), pid(i, j, k + 1), pid(i + 1, j, k + 1),
                pid(i + 1, j + 1, k + 1), pid(i, j + 1, k + 1)]

    return pts, hexa


_KUHN = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6),
         (0, 5, 1, 6)]
_FLIP = {CT.TETRA: [0, 2, 1, 3], CT.WEDGE: [0, 2, 1, 3, 5, 4],
         CT.PYRAMID: [0, 3, 2, 1, 4]}


def _grid(cells, points):
    """UnstructuredGrid of ``[(type, conn)]`` in order, positively oriented."""
    out = []
    for vtk_type, conn in cells:
        one = pv.UnstructuredGrid({vtk_type: np.array([conn])}, points)
        if vtk_type in _FLIP and one.compute_cell_sizes()["Volume"][0] < 0:
            conn = list(np.asarray(conn)[_FLIP[vtk_type]])
        out.append((vtk_type, conn))
    flat = np.hstack([[len(c), *c] for _, c in out])
    types = np.array([t for t, _ in out], dtype=np.uint8)
    grid = pv.UnstructuredGrid(flat, types, np.asarray(points, float))
    assert np.all(grid.compute_cell_sizes()["Volume"] > 0)
    return grid


def mixed():
    """Hex, wedge, pyramid and tet columns; regions alternate cell by cell."""
    pts, hexa = _lattice(4, 2, 2)
    pts = list(map(tuple, pts))
    cells = []
    for j in range(2):
        for k in range(2):
            h = hexa(0, j, k)
            cells.append((CT.HEXAHEDRON, h))
            h = hexa(1, j, k)
            cells.append((CT.WEDGE, [h[0], h[1], h[2], h[4], h[5], h[6]]))
            cells.append((CT.WEDGE, [h[0], h[2], h[3], h[4], h[6], h[7]]))
            h = hexa(2, j, k)
            centre = len(pts)
            pts.append(tuple(np.mean([pts[i] for i in h], axis=0)))
            for face in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
                         (2, 3, 7, 6), (3, 0, 4, 7)):
                cells.append((CT.PYRAMID, [h[f] for f in face] + [centre]))
            h = hexa(3, j, k)
            cells.extend((CT.TETRA, [h[i] for i in t]) for t in _KUHN)
    order = np.random.default_rng(7).permutation(len(cells))
    vol = _grid([cells[i] for i in order], np.array(pts))
    region = 1 + np.arange(vol.n_cells) % 2
    vol.cell_data["region"] = region
    vol.cell_data["material"] = region * 1.5

    lo, hi = np.array(vol.bounds[::2]), np.array(vol.bounds[1::2])
    surf = extract_surface(vol, nonlinear_subdivision=0)
    centres = surf.cell_centers().points
    on_box = np.any(np.isclose(centres, lo) | np.isclose(centres, hi), axis=1)
    outer = surf.extract_cells(np.flatnonzero(on_box))
    oc = outer.cell_centers().points
    outer_region = np.where(oc[:, 0] == lo[0], 1, np.where(oc[:, 0] == hi[0], 2, 3))
    one = vol.extract_cells(np.flatnonzero(region == 1))
    # algorithm=None: VTK's vtkDataSetSurfaceFilter ("dataset_surface", the
    # default up to pyvista 0.49) orients the faces of linear wedges inward
    one_surf = extract_surface(one, algorithm=None)
    inner = one_surf.extract_cells(np.flatnonzero(~np.any(
        np.isclose(one_surf.cell_centers().points, lo)
        | np.isclose(one_surf.cell_centers().points, hi), axis=1)))
    # one surface (in volume node ids) with the four side sets
    keys_points = []
    for part in (outer, inner):
        node_of = map_points(part.points, vol.points)
        keys_points += [node_of[ids] for ids in cell_ids(part)]
    set_of = np.concatenate([outer_region, np.full(inner.n_cells, 4)])
    flat = np.hstack([[len(c), *c] for c in keys_points])
    types = np.array([CT.TRIANGLE if len(c) == 3 else CT.QUAD for c in keys_points],
                     dtype=np.uint8)
    surface = pv.UnstructuredGrid(flat, types, vol.points)
    surface.cell_data["region"] = set_of
    names = {1: "inlet", 2: "outlet", 3: "walls", 4: "interface"}
    assert inner.n_cells > 0 and np.all(np.bincount(set_of)[1:] > 0)

    # each region has four element types: blocks 1..8, names with suffixes
    type_name = {CT.HEXAHEDRON: "hex8", CT.PYRAMID: "pyramid5",
                 CT.TETRA: "tetra4", CT.WEDGE: "wedge6"}
    keys = sorted({(r, type_name[t]) for r, t in zip(region, vol.celltypes)})
    block_of = {key: i + 1 for i, key in enumerate(keys)}
    rname = {1: "solid", 2: "fluid"}
    cell_keys = [(r, type_name[t]) for r, t in zip(region, vol.celltypes)]
    case = Case(
        vol, vol, 3,
        np.array([block_of[k] for k in cell_keys]),
        np.array([f"{rname[r]}_{t}" for r, t in cell_keys]),
        write_kwargs=dict(surface=surface, block_names=rname,
                          side_set_names=names),
        side_sets=_surface_keys_by_region(surface, vol.points, set_of, names),
        owner_regions={"interface": {1}},
        regions=region,
        node_sets={"inlet_nodes": np.flatnonzero(vol.points[:, 0] == 0)},
    )
    return _histories(case, 8)


def quadratic():
    """HEX20 and TETRA10 elements (mid-edge nodes shared between cells)."""
    pts, hexa = _lattice(3, 1, 1)
    pts = [tuple(p) for p in pts]
    hex_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
    tet_edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    linear = [(CT.HEXAHEDRON, hexa(0, 0, 0)), (CT.HEXAHEDRON, hexa(1, 0, 0))]
    h = hexa(2, 0, 0)
    linear += [(CT.TETRA, [h[i] for i in t]) for t in _KUHN]
    oriented = _grid(linear, np.array(pts))
    linear = list(zip(oriented.celltypes.tolist(), cell_ids(oriented)))
    mid = {}
    cells = []
    for vtk_type, conn in linear:
        edges = hex_edges if vtk_type == CT.HEXAHEDRON else tet_edges
        extra = []
        for a, b in edges:
            key = tuple(sorted((int(conn[a]), int(conn[b]))))
            if key not in mid:
                mid[key] = len(pts)
                pts.append(tuple((np.array(pts[key[0]]) + pts[key[1]]) / 2))
            extra.append(mid[key])
        quad_type = (CT.QUADRATIC_HEXAHEDRON if vtk_type == CT.HEXAHEDRON
                     else CT.QUADRATIC_TETRA)
        cells.append((quad_type, list(conn) + extra))
    vol = _grid(cells, np.array(pts))
    vol.cell_data["region"] = np.where(
        vol.celltypes == CT.QUADRATIC_HEXAHEDRON, 1, 2)
    lo, hi = np.array(vol.bounds[::2]), np.array(vol.bounds[1::2])
    surf = extract_surface(vol, nonlinear_subdivision=0)
    centres = surf.cell_centers().points
    surf = surf.extract_cells(np.flatnonzero(np.any(
        np.isclose(centres, lo) | np.isclose(centres, hi), axis=1)))
    del surf.cell_data["region"]  # one side set
    names = {1: "hexes", 2: "tets"}
    ids, bnames = _regions_by(vol.cell_data["region"], names)
    case = Case(
        vol, vol, 3, ids, bnames,
        write_kwargs=dict(surface=surf, block_names=names),
        side_sets={"side_set_1": side_keys(surf, map_points(surf.points, vol.points))},
    )
    return _histories(case, 9)


def annulus_2d():
    disc = pv.Disc(inner=0.5, outer=1.0, r_res=3, c_res=24)
    ug = disc.cast_to_unstructured_grid()
    edges = boundary_edges(disc)
    radius = np.linalg.norm(edges.cell_centers().points, axis=1)
    edges.cell_data["region"] = np.where(radius < 0.75, "inner", "outer")
    radius_pts = np.linalg.norm(ug.points, axis=1)
    case = Case(
        disc, ug, 2, np.ones(ug.n_cells, int), np.full(ug.n_cells, "block_1"),
        write_kwargs=dict(surface=edges),
        side_sets=_surface_keys_by_region(
            edges, ug.points, edges.cell_data["region"],
            {"inner": "inner", "outer": "outer"}),
        node_sets={"hole": np.flatnonzero(np.isclose(radius_pts, 0.5))},
    )
    return _histories(case, 10, vector=2, elem_comp=2)


def shells_3d():
    sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)
    patch = pv.Plane(center=(0, 0, 1), direction=(1, 1, 1), i_resolution=3,
                     j_resolution=2)
    mb = pv.MultiBlock({"sphere": sphere, "patch": patch})
    combined = pv.MultiBlock(
        {"sphere": sphere.copy(), "patch": patch.copy()}).combine(merge_points=True)
    # arrays of only some leaves are kept (NaN elsewhere); combine() drops them
    texture = np.full((combined.n_points, 2), np.nan)
    _, at = cKDTree(combined.points).query(patch.points)
    texture[at] = patch.point_data["TextureCoordinates"]
    combined.point_data["TextureCoordinates"] = texture
    is_tri = combined.celltypes == CT.TRIANGLE
    case = Case(
        mb, combined, 3, np.where(is_tri, 1, 2), np.where(is_tri, "sphere", "patch"),
        node_sets={"north": [int(np.argmax(combined.points[:, 2]))]},
    )
    return _histories(case, 11)


CORPUS = {
    "sphere_tets": sphere_tets,
    "cube_tets": cube_tets,
    "plane_2d": plane_2d,
    "hexbeam": hexbeam,
    "voxels": voxels,
    "far_from_origin": far_from_origin,
    "mixed_hex_tet_wedge_pyramid": mixed,
    "quadratic_hex20_tet10": quadratic,
    "annulus_2d": annulus_2d,
    "shells_3d": shells_3d,
}


@pytest.fixture(scope="module", params=list(CORPUS))
def written(request, tmp_path_factory):
    """(case, path) of every corpus mesh, written once per module."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pv.PyVistaFutureWarning)
        case = CORPUS[request.param]()
    path = tmp_path_factory.mktemp(request.param) / f"{request.param}.exo"
    snapshot = {k: np.array(v) for k, v in case.expected.point_data.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a clean write warns about nothing
        case.write(path)
    for k, v in snapshot.items():  # never mutates its input
        np.testing.assert_array_equal(case.expected.point_data[k], v)
    return case, path


# --------------------------------------------------------------------------- #
# exovista's own reader: exact
# --------------------------------------------------------------------------- #
def test_mesh_is_exact(written):
    case, path = written
    grid = read_exo(path)
    exp = case.expected
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.points.dtype == np.float64
    np.testing.assert_array_equal(grid.points, np.asarray(exp.points, np.float64))
    types, offsets, conn = expected_cells(exp)
    np.testing.assert_array_equal(grid.celltypes, types)
    np.testing.assert_array_equal(offsets_of(grid), offsets)
    np.testing.assert_array_equal(grid.cell_connectivity, conn)
    assert grid.field_data["num_dim"].tolist() == [case.num_dim]
    with open_reader(path) as reader:
        assert reader.num_dim == case.num_dim
        assert reader.float_word_size == 8


def test_blocks_from_regions(written):
    case, path = written
    grid = read_exo(path)
    np.testing.assert_array_equal(grid.cell_data["exo_block_id"], case.block_ids)
    np.testing.assert_array_equal(grid.cell_data["exo_block_name"], case.block_names)
    np.testing.assert_array_equal(
        grid.cell_data["exo_elem_id"], np.arange(1, grid.n_cells + 1))
    # file order: blocks by region, cells keep their relative order
    with open_reader(path) as reader:
        blocks = reader.blocks()
        id_map = reader.elem_id_map()
    assert [b.id for b in blocks] == sorted(set(case.block_ids.tolist()))
    for block, start in zip(blocks, np.cumsum([0] + [b.n_elem for b in blocks])):
        ids = id_map[start: start + block.n_elem] - 1
        assert np.all(np.diff(ids) > 0)
        assert set(case.block_ids[ids]) == {block.id}


@pytest.mark.parametrize("time_step", [1, 2, -1])
def test_arrays_at_each_step(written, time_step):
    case, path = written
    grid = read_exo(path, time_step=time_step)
    step = time_step - 1 if time_step > 0 else case.n_steps + time_step
    if step >= case.n_steps:
        pytest.skip("fewer steps")
    expected_points = {k: v[step] for k, v in case.node_history().items()}
    expected_cells_ = {k: v[step] for k, v in case.element_history().items()}
    assert set(grid.point_data.keys()) == set(expected_points)
    assert set(grid.cell_data.keys()) == set(expected_cells_) | {
        "exo_block_id", "exo_block_name", "exo_elem_id"}
    for name, values in expected_points.items():
        np.testing.assert_array_equal(grid.point_data[name], values, err_msg=name)
    for name, values in expected_cells_.items():
        np.testing.assert_array_equal(grid.cell_data[name], values, err_msg=name)
    np.testing.assert_array_equal(grid.field_data["times"], case.times)
    np.testing.assert_array_equal(grid.field_data["time"], case.times[step])
    for name, values in case.global_fields.items():
        np.testing.assert_array_equal(grid.field_data[name], [values[step]])


def test_histories(written):
    case, path = written
    nodes = read_node_fields(path, combine_components=True)
    assert set(nodes) == set(case.node_history())
    for name, values in case.node_history().items():
        np.testing.assert_array_equal(nodes[name], values, err_msg=name)
    elems = read_element_fields(path, combine_components=True)
    assert set(elems) == set(case.element_history())
    for name, values in case.element_history().items():
        np.testing.assert_array_equal(elems[name], values, err_msg=name)
    globs = read_global_fields(path)
    assert set(globs) == set(case.global_fields)
    for name, values in case.global_fields.items():
        np.testing.assert_array_equal(globs[name], values)


def test_side_sets(written):
    case, path = written
    grid, sets = read_exo(path, read_side_sets=True)
    assert sorted(sets.keys()) == sorted(case.side_sets)
    cells = cell_ids(grid)
    for name, keys in case.side_sets.items():
        ss = sets[name]
        assert sorted(side_keys(ss)) == sorted(keys), name
        assert ss.n_cells == len(keys)  # no duplicates, nothing lost
        for key, cell in zip(side_keys(ss), ss.cell_data["cell_id"]):
            assert set(key) <= set(cells[cell].tolist())
        if name in case.owner_regions:
            owners = case.regions[ss.cell_data["cell_id"]]
            assert set(owners.tolist()) == case.owner_regions[name]
    alone = read_side_sets(path)
    for name in case.side_sets:
        np.testing.assert_array_equal(
            alone[name].cell_data["cell_id"], sets[name].cell_data["cell_id"])


def test_node_sets(written):
    case, path = written
    got = read_node_sets(path)
    assert sorted(got) == sorted(case.node_sets)
    for name, nodes in case.node_sets.items():
        np.testing.assert_array_equal(got[name], nodes)


# --------------------------------------------------------------------------- #
# the oracles
# --------------------------------------------------------------------------- #
def test_vtk_reads_the_same(written):
    case, path = written
    last = case.n_steps - 1
    reader, mb = vtk_read(path, time_index=last)
    np.testing.assert_array_equal(reader.time_values, case.times)
    exp = case.expected
    exp_cells = expected_cells(exp)
    node_hist, elem_hist = case.node_history(), case.element_history()
    seen = []
    blocks = mb["Element Blocks"]
    assert sorted(blocks.keys()) == sorted(set(case.block_names.tolist()))
    for name in blocks.keys():
        blk = blocks[name]
        orig = np.asarray(blk.cell_data["GlobalElementId"]) - 1
        seen.extend(orig.tolist())
        assert set(case.block_names[orig]) == {name}
        assert blk.field_data["ElementBlockIds"].tolist() == [case.block_ids[orig[0]]]
        np.testing.assert_array_equal(blk.celltypes, exp_cells[0][orig])
        k = blk.cell_connectivity.size // blk.n_cells
        vtk_coords = np.asarray(blk.points, float)[blk.cell_connectivity.reshape(-1, k)]
        np.testing.assert_allclose(
            vtk_coords, cell_point_coords(exp, orig, exp_cells),
            rtol=0, atol=case.vtk_atol)
        nodes = np.asarray(blk.point_data["GlobalNodeId"]) - 1
        for var, values in node_hist.items():
            n_comp = 1 if values.ndim == 2 else values.shape[2]
            np.testing.assert_array_equal(
                vtk_array(blk.point_data, var, n_comp), values[last][nodes],
                err_msg=var)
        for var, values in elem_hist.items():
            n_comp = 1 if values.ndim == 2 else values.shape[2]
            np.testing.assert_array_equal(
                vtk_array(blk.cell_data, var, n_comp), values[last][orig],
                err_msg=var)
        for var, values in case.global_fields.items():
            np.testing.assert_array_equal(blk.field_data[var], values)
    assert sorted(seen) == list(range(exp.n_cells))

    side_sets = mb["Side Sets"]
    assert sorted(side_sets.keys()) == sorted(case.side_sets)
    for name, keys in case.side_sets.items():
        ss = side_sets[name]
        gids = np.asarray(ss.point_data["GlobalNodeId"]) - 1
        assert sorted(side_keys(ss, gids)) == sorted(keys), name
        if name in case.owner_regions:
            owners = case.regions[np.asarray(ss.cell_data["SourceElementId"])]
            assert set(owners.tolist()) == case.owner_regions[name]
    node_sets = mb["Node Sets"]
    assert sorted(node_sets.keys()) == sorted(case.node_sets)
    for name, nodes in case.node_sets.items():
        gids = np.asarray(node_sets[name].point_data["GlobalNodeId"]) - 1
        assert sorted(gids.tolist()) == sorted(np.asarray(nodes).tolist())


def test_vtk_reads_every_step(written):
    case, path = written
    for step in range(case.n_steps):
        _, mb = vtk_read(path, time_index=step)
        for blk in mb["Element Blocks"]:
            nodes = np.asarray(blk.point_data["GlobalNodeId"]) - 1
            orig = np.asarray(blk.cell_data["GlobalElementId"]) - 1
            np.testing.assert_array_equal(
                blk.point_data["temperature"],
                case.node_fields["temperature"][step][nodes])
            np.testing.assert_array_equal(
                blk.cell_data["pressure"], case.element_fields["pressure"][step][orig])


def test_meshio_reads_it(written):
    case, path = written
    if not case.meshio:
        pytest.skip("meshio does not know these element types")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "Skipping some time data"
        mesh = meshio.read(path, file_format="exodus")
    np.testing.assert_array_equal(mesh.points, np.asarray(case.expected.points, float))
    types, offsets, conn = expected_cells(case.expected)
    counts = collections.Counter(MESHIO_TYPE[t] for t in types.tolist())
    got = collections.Counter()
    for block in mesh.cells:
        got[block.type] += len(block.data)
    assert got == counts
    linear = [c for c in mesh.cells if c.type in {MESHIO_TYPE[t] for t in LINEAR}]
    got_cells = sorted(tuple(row) for c in linear for row in c.data.tolist())
    want = sorted(
        tuple(conn[offsets[i]: offsets[i + 1]].tolist())
        for i in np.flatnonzero(np.isin(types, list(LINEAR))))
    assert got_cells == want
    # names are NUL padded (like the Exodus C library), so meshio, which does
    # not strip blanks, gets the plain variable and set names
    assert set(mesh.point_data) == _meshio_names(case.node_history(), group=True)
    assert set(mesh.cell_data) == _meshio_names(case.element_history())
    # (meshio skips empty node sets but keeps their names, shifting the rest)
    assert set(mesh.point_sets) <= set(case.node_sets)
    assert len(mesh.point_sets) == sum(len(v) > 0 for v in case.node_sets.values())


def _meshio_names(histories, group=False):
    """meshio's array names for the scalar Exodus variables of ``histories``
    (for point data meshio groups ``<name>X/Y[/Z]`` itself, keeping
    ``<name>``)."""
    names = []
    for name, values in histories.items():
        values = np.asarray(values)
        if values.ndim == 3 and values.shape[2] > 1:
            names += [name + s for s in component_suffixes(values.shape[2])]
        else:
            names.append(name)
    if not group:
        return set(names)
    single, double, triple = meshio_categorize(names)
    return {group[0] for group in single + double + triple}


# --------------------------------------------------------------------------- #
# read_exo -> write_exo -> read_exo on the files in test/data
# --------------------------------------------------------------------------- #
def side_set_surface(grid, sets, region_key):
    """All side sets as one surface (grid point ids); region = side-set id."""
    flat, types, region, names = [], [], [], {}
    for key in sets.keys():
        ss = sets[key].cast_to_unstructured_grid()
        set_id = int(sets[key].field_data["exo_side_set_id"][0])
        names[set_id] = key
        for i, ids in enumerate(cell_ids(ss)):
            flat.append([len(ids), *ids.tolist()])
            types.append(ss.celltypes[i])
            region.append(set_id)
    surface = pv.UnstructuredGrid(
        np.hstack(flat), np.array(types, np.uint8), np.asarray(grid.points))
    surface.cell_data[region_key] = np.array(region)
    return surface, names


DATA_FILES = [
    ("mkmesh.gen",),
    ("noh.exo",),
    ("edges.base.exo",),
    ("noh.exo.3.*",),
    ("mkmesh.par.2.*",),
]


@pytest.mark.parametrize("files", DATA_FILES, ids=[f[0] for f in DATA_FILES])
def test_data_file_rewrite(tmp_path, files):
    src = [os.path.join(DATA, f) for f in files]
    if not glob.glob(src[0]):
        pytest.skip(f"missing test data {files[0]}")
    grid, sets = read_exo(*src, read_side_sets=True)
    node_sets = read_node_sets(*src)
    with open_reader(*src) as reader:
        set_ids = [ns.id for ns in reader.node_sets()]
        title = reader.title
    times = grid.field_data["times"]
    blocks = dict(zip(grid.cell_data["exo_block_id"].tolist(),
                      grid.cell_data["exo_block_name"]))
    kwargs = {}
    if sets.n_blocks:
        # region_key names the region array of the volume and the surface
        kwargs["surface"], kwargs["side_set_names"] = side_set_surface(
            grid, sets, "exo_block_id")
    out = tmp_path / "rewritten.exo"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(
            out, grid, region_key="exo_block_id", block_names=blocks,
            node_sets=dict(zip(set_ids, node_sets.values())),
            times=times if times.size else None,
            node_fields=read_node_fields(*src),
            element_fields=read_element_fields(*src),
            global_fields=read_global_fields(*src),
            save_node_arrays=False, save_cell_arrays=False, title=title,
            **kwargs,
        )

    again, sets_again = read_exo(out, read_side_sets=True)
    np.testing.assert_array_equal(again.points, grid.points)
    np.testing.assert_array_equal(again.celltypes, grid.celltypes)
    np.testing.assert_array_equal(again.cell_connectivity, grid.cell_connectivity)
    np.testing.assert_array_equal(offsets_of(again), offsets_of(grid))
    for key in ("exo_block_id", "exo_block_name"):
        np.testing.assert_array_equal(again.cell_data[key], grid.cell_data[key])
    np.testing.assert_array_equal(
        again.cell_data["exo_elem_id"], np.arange(1, grid.n_cells + 1))
    for data, data_again in ((grid.point_data, again.point_data),
                             (grid.cell_data, again.cell_data)):
        assert sorted(data_again.keys()) == sorted(data.keys())
        for key in data.keys():
            if key != "exo_elem_id":
                np.testing.assert_array_equal(data_again[key], data[key], key)
    assert sorted(again.field_data.keys()) == sorted(grid.field_data.keys())
    for key in grid.field_data.keys():
        np.testing.assert_array_equal(again.field_data[key], grid.field_data[key])

    assert sets_again.keys() == sets.keys()
    for key in sets.keys():
        for array in ("cell_id", "exo_side"):
            np.testing.assert_array_equal(
                sets_again[key].cell_data[array], sets[key].cell_data[array])
        assert side_keys(sets_again[key]) == side_keys(sets[key])
    again_sets = read_node_sets(out)
    assert list(again_sets) == list(node_sets)
    for key, nodes in node_sets.items():
        np.testing.assert_array_equal(again_sets[key], nodes)
    for read in (read_node_fields, read_element_fields, read_global_fields):
        before, after = read(*src), read(out)
        assert list(after) == list(before)
        for key in before:
            np.testing.assert_array_equal(after[key], before[key], key)

    # and VTK reads the rewritten file
    reader, mb = vtk_read(out)
    np.testing.assert_array_equal(reader.time_values, times)
    assert sum(b.n_cells for b in mb["Element Blocks"]) == grid.n_cells


# --------------------------------------------------------------------------- #
# writing what a reader returns
# --------------------------------------------------------------------------- #
def _two_region_grid():
    grid = pv.ImageData(dimensions=(4, 3, 2)).cast_to_unstructured_grid()
    grid.cell_data["region"] = np.array([1, 2, 1, 2, 1, 2])
    grid.point_data["T"] = grid.points[:, 0] * 2.0
    return grid


def test_reader_bookkeeping_arrays_are_not_written(tmp_path):
    grid = _two_region_grid()
    write_exo(tmp_path / "a.exo", grid)
    read_back = read_exo(tmp_path / "a.exo")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 1.x-style "skipped exo_block_name"
        write_exo(tmp_path / "b.exo", read_back, region_key="exo_block_id",
                  block_names={1: "one", 2: "two"})
        vtk_grid = vtk_read(tmp_path / "a.exo")[1]["Element Blocks"].combine()
        assert {"ObjectId", "GlobalElementId", "PedigreeElementId"} <= set(
            vtk_grid.cell_data.keys())
        write_exo(tmp_path / "c.exo", vtk_grid)
    for name in ("b.exo", "c.exo"):
        with open_reader(tmp_path / name) as reader:
            assert reader.elem_var_names() == []
            assert reader.node_var_names() == ["T"]
    again = read_exo(tmp_path / "b.exo")
    np.testing.assert_array_equal(again.cell_data["exo_block_id"],
                                  read_back.cell_data["exo_block_id"])
    assert list(again.cell_data["exo_block_name"]) == [
        "one", "two", "one", "two", "one", "two"]
    np.testing.assert_array_equal(again.points, read_back.points)
    np.testing.assert_array_equal(again.cell_connectivity,
                                  read_back.cell_connectivity)


def test_nan_on_a_whole_block_is_undefined_there(tmp_path):
    grid = _two_region_grid()
    region = grid.cell_data["region"]
    damage = np.where(region == 2, np.nan, 0.5)  # undefined on block 2
    grid.cell_data["damage"] = damage
    velocity = np.ones((3, grid.n_cells, 3)) * np.arange(3)[:, None, None]
    velocity[:, region == 1, :] = np.nan  # undefined on block 1, every step
    partly = np.full((3, grid.n_cells), np.nan)
    partly[1, 0], partly[2, 1] = 1.0, 2.0  # one value per block: both stay
    nowhere = np.full((3, grid.n_cells), np.nan)  # NaN on every block
    path = tmp_path / "nan.exo"
    write_exo(path, grid, times=[0.0, 1.0, 2.0],
              element_fields={"velocity": velocity, "partly": partly,
                              "nowhere": nowhere})
    table = netCDF4.Dataset(path).variables["elem_var_tab"][:]
    with open_reader(path) as reader:
        names = reader.elem_var_names()
        ids = [b.id for b in reader.blocks()]
    defined = {n: [ids[b] for b in np.flatnonzero(table[:, i])]
               for i, n in enumerate(names)}
    assert defined == {
        "damage": [1], "velocity_X": [2], "velocity_Y": [2], "velocity_Z": [2],
        "partly": [1, 2], "nowhere": [1, 2],
    }
    fields = read_element_fields(path, combine_components=True)
    np.testing.assert_array_equal(fields["velocity"], velocity)
    np.testing.assert_array_equal(fields["partly"], partly)
    np.testing.assert_array_equal(fields["nowhere"], nowhere)
    np.testing.assert_array_equal(read_exo(path).cell_data["damage"], damage)
    # VTK stores each variable only on its blocks (and does not crash)
    blocks = vtk_read(path, time_index=2)[1]["Element Blocks"]
    assert "damage" in blocks["block_1"].cell_data
    assert "velocity_" not in blocks["block_1"].cell_data
    assert "damage" not in blocks["block_2"].cell_data
    np.testing.assert_array_equal(blocks["block_2"].cell_data["velocity_"],
                                  velocity[2, region == 2])
