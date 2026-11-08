"""Write PyVista meshes to ExodusII files.

:func:`write_exo` converts a PyVista mesh (plus an optional boundary surface,
node sets and time-varying fields) into an :class:`~exovista._model.ExodusModel`
and writes it with :func:`exovista._backend.write_model`:

* cells are grouped into element blocks by (region, Exodus element type), with
  the VTK -> Exodus node permutations of :mod:`exovista._celltypes`;
* the faces (3D) or edges (2D) of a surface mesh are matched topologically to
  element sides and written as side sets;
* point / cell data and explicit time histories become node, element and
  global variables (multi-component arrays are split into scalar components
  named with :func:`~exovista._model.component_suffixes`);
* the original cell order is stored in the element id map
  (``elem_num_map = original cell index + 1``).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from vtkmodules.vtkCommonDataModel import vtkDataObject

from . import _celltypes as ct
from ._backend import write_model
from ._model import (
    MAX_NAME_LENGTH,
    Block,
    ExodusModel,
    NodeSet,
    SideSet,
    component_suffixes,
)

__all__ = ["write_exo"]

logger = logging.getLogger(__name__)

_INT32_MAX = int(np.iinfo(np.int32).max)

#: Bookkeeping arrays never written as variables: those VTK filters attach,
#: those :func:`exovista.read_exo` adds and those VTK's Exodus reader adds (so
#: a mesh read by either reader can be written back without stale ids).
_EXCLUDED_ARRAYS = frozenset(
    {
        "vtkOriginalPointIds",
        "vtkOriginalCellIds",
        "vtkGhostType",
        "exo_block_id",
        "exo_block_name",
        "exo_elem_id",
        "ObjectId",
        "GlobalElementId",
        "GlobalNodeId",
        "PedigreeElementId",
        "PedigreeNodeId",
    }
)

_PRECISIONS = {"double": 8, "float64": 8, "single": 4, "float32": 4}

#: z extent (relative to the bounding-box diagonal) below which a mesh is flat.
_FLAT_RTOL = 1e-12
#: Default side-set matching tolerance, relative to the bounding-box diagonal.
_SIDE_SET_RTOL = 1e-6
#: A face normal is "pointing out" of an element when (face centre - element
#: centre) . normal exceeds this fraction of |face centre - element centre| |n|.
_NORMAL_RTOL = 1e-9


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def write_exo(
    filename,
    volume,
    surface=None,
    *,
    region_key="region",
    block_names=None,
    side_set_names=None,
    node_sets=None,
    save_node_arrays=True,
    save_cell_arrays=True,
    times=None,
    node_fields=None,
    element_fields=None,
    global_fields=None,
    num_dim=None,
    precision="double",
    compression_level=None,
    side_set_tolerance=None,
    title=None,
) -> None:
    """Write a PyVista mesh to an ExodusII file.

    Parameters
    ----------
    filename : str or os.PathLike
        Output path. The file is written to a temporary file next to it and
        moved into place only when complete: on any error no file is created
        (an existing file is left untouched).
    volume : pyvista.DataSet or pyvista.MultiBlock
        The mesh. Any dataset is accepted (it is cast with
        ``cast_to_unstructured_grid()`` when needed). For a ``MultiBlock``,
        every non-empty leaf becomes its own region (region values ``1..N`` in
        leaf order, block names default to the leaf keys); the leaves are
        combined with ``merge_points=True`` and "original cell order" below
        means the order of the combined mesh (leaf after leaf). The input is
        never modified.
    surface : pyvista.DataSet, optional
        Boundary faces (3D meshes) or edges (``num_dim=2`` meshes) to write as
        side sets, e.g. ``volume.extract_surface()`` or a subset of it. Each
        surface cell must match the side of an element (see Notes). One side
        set is written per value of ``surface.cell_data[region_key]`` (or a
        single set without it).
    region_key : str, default "region"
        Name of the cell array that assigns cells to regions, on both
        ``volume`` and ``surface``. Integer or string values. Without it the
        whole mesh (surface) is one region.
    block_names : list or dict or str, optional
        Element-block names: a list ordered like the sorted unique region
        values, a dict ``{region_value: name}`` (regions not in the dict keep
        their default), or a single name when there is one region. Defaults:
        the region value for string regions, the leaf key for ``MultiBlock``
        input, else ``"block_<id>"``. A region whose cells have several
        element types is split into one block per type; a given or
        string-region name then gets the type as a suffix,
        ``"<name>_<exodus type in lower case>"`` (e.g. ``"steel_hex8"``),
        while ``"block_<id>"`` defaults are already unique. Names must be
        unique and at most 256 bytes.
    side_set_names : list or dict or str, optional
        Side-set names, like ``block_names`` but for the surface regions;
        default ``"side_set_<id>"`` (or the region value for string regions).
    node_sets : dict or list, optional
        Node sets: ``{name: indices}`` or a list of index arrays (named
        ``"node_set_<id>"``). Indices are 0-based point indices of ``volume``
        (unique, in range); a boolean mask over the points is also accepted.
        A dict with integer keys >= 1 uses them as node-set ids.
    save_node_arrays : bool, default True
        Write ``volume.point_data`` as node variables (constant in time).
    save_cell_arrays : bool, default True
        Write ``volume.cell_data`` as element variables (constant in time).
        The ``region_key`` array is not written.
    times : array_like, optional
        Time value of each step. If omitted and any variable is written, a
        single step at ``t = 0.0`` is written.
    node_fields : dict, optional
        Time-varying node variables ``{name: values}`` with values of shape
        ``(n_steps, n_points)`` or ``(n_steps, n_points, n_components)``; for a
        single step also ``(n_points,)`` or ``(n_points, n_components)``.
        A name here replaces a point array of the same name.
    element_fields : dict, optional
        Time-varying element variables, like ``node_fields`` but with one
        column per cell in the ORIGINAL ``volume`` cell order. A name here
        replaces a cell array of the same name.
    global_fields : dict, optional
        Global variables ``{name: values}`` of shape ``(n_steps,)`` (or
        ``(n_steps, n_components)``).
    num_dim : {None, 2, 3}, optional
        Spatial dimension of the file. ``None`` (default): 2 when the mesh has
        no 3D cells and all points share one z coordinate (z is then dropped),
        else 3. In a ``num_dim=3`` file, 2D cells are written as shells
        (``TRISHELL*``/``SHELL*``).
    precision : {"double", "single"}, default "double"
        Floating point storage of coordinates, times and variables.
    compression_level : int, optional
        zlib level 0-9 (0 disables compression); default from the
        ``EXODUSII_COMPRESSION_LEVEL`` environment variable (1).
    side_set_tolerance : float, optional
        Largest distance between a surface point and the volume node it is
        matched to. Default ``1e-6`` times the bounding-box diagonal.
    title : str, optional
        Database title (default ``"exovista"``).

    Raises
    ------
    ValueError
        For an empty mesh, unsupported cell types (``POLYGON``,
        ``POLYHEDRON``, ``TRIANGLE_STRIP``, Lagrange cells, ...), a surface
        that does not match the volume, inconsistent names / shapes, etc.
    TypeError
        For inputs of the wrong type.

    Notes
    -----
    **Element blocks.** Cells are grouped by (region, Exodus element type);
    blocks are ordered by sorted region value, then element-type name. If
    every region value is an integer >= 1 and each region has a single
    element type, the block id is the region value, otherwise blocks are
    numbered ``1..N`` in block order. Inside a block, cells keep their
    original relative order, and the element id map stores
    ``original cell index + 1`` for every element, so readers can restore
    the input order.

    **Variables.** Arrays with several components are split into scalar
    variables named ``name + suffix`` (2: ``_X, _Y``; 3: ``_X, _Y, _Z``;
    6: ``_XX, _YY, _ZZ, _XY, _YZ, _XZ``; 9: ``_XX, _XY, ... _ZZ``; else
    ``_0 .. _{n-1}``). VTK's Exodus reader regroups ``_X/_Y[/_Z]`` into a
    vector (named ``name_``). Boolean and integer arrays are written as
    floats; complex and non-numeric arrays are skipped with a warning.
    Bookkeeping arrays are never written: VTK's ``vtkOriginalPointIds``,
    ``vtkOriginalCellIds`` and ``vtkGhostType``, the ``exo_block_id``,
    ``exo_block_name`` and ``exo_elem_id`` arrays of :func:`exovista.read_exo`
    and the ``ObjectId``, ``GlobalElementId``, ``GlobalNodeId``,
    ``PedigreeElementId`` and ``PedigreeNodeId`` arrays of VTK's Exodus
    reader. An element variable that is NaN on every element of a block (all
    components, all steps) is not stored on that block (Exodus truth table),
    which is how :func:`exovista.read_exo` reports variables that are
    undefined on a block; it stays stored everywhere if it is NaN on every
    block.

    **Writing a mesh read by** :func:`exovista.read_exo`: pass
    ``region_key="exo_block_id"`` to keep its element blocks (and their ids)
    and ``block_names`` to keep their names; ``region_key`` also names the
    region array of ``surface``.

    **Side sets.** Surface points are matched to volume nodes within
    ``side_set_tolerance`` and each surface cell's corner nodes are looked up
    among the sides of the solid elements (3D meshes) or the edges of the 2D
    elements (``num_dim=2``); shells, bars and spheres are not side-set
    targets. Quadratic cells match by their corner nodes. A face shared by
    two elements (an internal interface) is assigned to the element the
    surface normal points out of, so the faces of
    ``region_mesh.extract_surface(algorithm=None)`` belong to that region
    (with ``algorithm="dataset_surface"``, pyvista's default up to at least
    0.49, VTK's ``vtkDataSetSurfaceFilter`` orients the faces of linear
    wedges inward, so their interface faces would go to the neighbour); for
    2D edges the in-plane normal ``(dy, -dx)`` of the edge direction is used
    (outward for counter-clockwise boundaries). Ties go to the element with
    the lower block id, then the lower element index. Repeated faces within
    a set are written once (with a warning). Sides keep the surface cell
    order. Ids and names follow the block rules (integer surface regions
    >= 1 are used as side-set ids).
    """
    float_word_size = _float_word_size(precision)
    compression_level = _compression_level(compression_level)
    notes: list[str] = []

    grid, regions = _prepare_volume(volume, region_key)
    cells = _collect_cells(grid, num_dim)
    layout = _layout_blocks(cells, regions, block_names)
    side_sets = []
    if surface is not None:
        side_sets = _build_side_sets(
            surface, cells, layout, region_key, side_set_names,
            side_set_tolerance, notes,
        )
    nsets = _build_node_sets(node_sets, cells.n_points)
    times_arr, node_vars, elem_vars, global_vars = _build_variables(
        grid, cells, layout, region_key,
        times=times,
        node_fields=node_fields,
        element_fields=element_fields,
        global_fields=global_fields,
        save_node_arrays=save_node_arrays,
        save_cell_arrays=save_cell_arrays,
        notes=notes,
    )
    model = ExodusModel(
        title="exovista" if title is None else str(title),
        num_dim=cells.num_dim,
        coords=cells.points[:, : cells.num_dim],
        blocks=layout.blocks,
        side_sets=side_sets,
        node_sets=nsets,
        elem_id_map=layout.order + 1,
        times=times_arr,
        node_vars=node_vars,
        elem_vars=elem_vars,
        global_vars=global_vars,
    )
    for message in notes:
        warnings.warn(message, UserWarning, stacklevel=2)
    logger.debug(
        "writing %s: num_dim=%d, %d nodes, %d elements in %d block(s), "
        "%d side set(s), %d node set(s), %d step(s)",
        filename, model.num_dim, model.num_nodes, model.num_elem,
        len(model.blocks), len(side_sets), len(nsets), model.n_steps,
    )
    write_model(
        filename,
        model,
        float_word_size=float_word_size,
        compression_level=compression_level,
    )


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
def _float_word_size(precision) -> int:
    key = precision.lower() if isinstance(precision, str) else precision
    if key in _PRECISIONS:
        return _PRECISIONS[key]
    raise ValueError(f"precision must be 'double' or 'single', got {precision!r}")


def _compression_level(level):
    if level is None:
        return None
    if isinstance(level, (bool, np.bool_)) or not isinstance(
        level, (int, np.integer)
    ):
        raise TypeError(f"compression_level must be an int 0-9, got {level!r}")
    if not 0 <= int(level) <= 9:
        raise ValueError(f"compression_level must be in 0..9, got {level!r}")
    return int(level)


# --------------------------------------------------------------------------- #
# Input meshes and regions
# --------------------------------------------------------------------------- #
@dataclass
class _Regions:
    """Region of every cell (of the volume or of the surface)."""

    values: list  # sorted unique region values (python scalars)
    code: np.ndarray  # per cell: index into ``values``
    defaults: list  # per region: default name (str) or None
    integral: bool  # all values are integers usable as Exodus ids (>= 1)


def _as_unstructured(obj, what: str) -> pv.UnstructuredGrid:
    if isinstance(obj, vtkDataObject) and not isinstance(obj, pv.DataObject):
        obj = pv.wrap(obj)  # a bare VTK dataset
    if isinstance(obj, pv.UnstructuredGrid):
        return obj
    if isinstance(obj, pv.DataSet):
        return obj.cast_to_unstructured_grid()
    raise TypeError(
        f"{what} must be a pyvista DataSet"
        + (" or MultiBlock" if what == "volume" else "")
        + f", got {type(obj).__name__}"
    )


def _leaves(multiblock: pv.MultiBlock, out: list) -> list:
    """Non-empty dataset leaves of a (nested) MultiBlock as (name, dataset)."""
    for i in range(multiblock.n_blocks):
        block = multiblock[i]
        if block is None:
            continue
        if isinstance(block, pv.MultiBlock):
            _leaves(block, out)
            continue
        if not isinstance(block, pv.DataSet):
            raise TypeError(
                f"MultiBlock leaf {i} is a {type(block).__name__}, not a DataSet"
            )
        if block.n_cells == 0 or block.n_points == 0:
            continue
        name = multiblock.get_block_name(i)
        out.append((str(name) if name else f"block_{len(out) + 1}", block))
    return out


def _unique_strings(names: list[str]) -> list[str]:
    """Make names unique by appending ``_2``, ``_3``, ... to repeats."""
    seen: set[str] = set()
    out = []
    for name in names:
        candidate, k = name, 1
        while candidate.rstrip() in seen:
            k += 1
            candidate = f"{name}_{k}"
        seen.add(candidate.rstrip())
        out.append(candidate)
    return out


def _prepare_volume(volume, region_key):
    """Return ``(grid, regions)``; the input is never modified."""
    if isinstance(volume, vtkDataObject) and not isinstance(volume, pv.DataObject):
        volume = pv.wrap(volume)
    if isinstance(volume, pv.MultiBlock):
        leaves = _leaves(volume, [])
        if not leaves:
            raise ValueError("cannot write an empty mesh: the MultiBlock has no cells")
        names = _unique_strings([name for name, _ in leaves])
        if len(leaves) == 1:
            grid = _as_unstructured(leaves[0][1], "volume")
            labels = np.zeros(grid.n_cells, dtype=np.int64)
        else:
            tag = "__exovista_leaf__"
            combined = pv.MultiBlock()
            for i, (_, leaf) in enumerate(leaves):
                leaf = leaf.copy(deep=False)  # shares arrays, never the caller's
                leaf.cell_data[tag] = np.full(leaf.n_cells, i, dtype=np.int32)
                combined.append(leaf)
            grid = combined.combine(merge_points=True)
            labels = np.asarray(grid.cell_data[tag], dtype=np.int64)
            del grid.cell_data[tag]  # our own object
        regions = _Regions(
            values=list(range(1, len(leaves) + 1)),
            code=labels,
            defaults=names,
            integral=True,
        )
        return grid, regions

    grid = _as_unstructured(volume, "volume")
    if grid.n_cells == 0 or grid.n_points == 0:
        raise ValueError(
            f"cannot write an empty mesh ({grid.n_points} points, "
            f"{grid.n_cells} cells)"
        )
    return grid, _regions_of(grid, region_key, "volume")


def _regions_of(mesh, region_key, what) -> _Regions:
    n = mesh.n_cells
    if region_key is None or region_key not in mesh.cell_data:
        return _Regions([1], np.zeros(n, dtype=np.int64), [None], True)
    arr = np.asarray(mesh.cell_data[region_key])
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(
            f"{what} region array {region_key!r} must hold one value per cell, "
            f"got shape {arr.shape}"
        )
    kind = arr.dtype.kind
    is_string = kind in "USO"
    if is_string:
        arr = arr.astype(str)
    elif kind == "b":
        arr = arr.astype(np.int64)
    elif kind == "f":
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{what} region array {region_key!r} contains NaN or infinity"
            )
        if np.all(arr == np.round(arr)):
            arr = arr.astype(np.int64)
    elif kind not in "iu":
        raise TypeError(
            f"{what} region array {region_key!r} must hold integers or strings, "
            f"got dtype {arr.dtype}"
        )
    uniq, code = np.unique(arr, return_inverse=True)
    values = uniq.tolist()
    integral = (
        uniq.dtype.kind in "iu"
        and int(uniq.min()) >= 1
        and int(uniq.max()) <= _INT32_MAX
    )
    defaults = [str(v) for v in values] if is_string else [None] * len(values)
    return _Regions(values, code.reshape(-1).astype(np.int64), defaults, integral)


def _resolve_names(spec, regions: _Regions, arg: str) -> list:
    """Per-region base names from a ``block_names``-style argument."""
    n = len(regions.values)
    names = list(regions.defaults)
    if spec is None:
        return names
    if isinstance(spec, str):
        if n != 1:
            raise TypeError(
                f"{arg} is a single string but there are {n} regions "
                f"{regions.values}; pass a list or a dict {{region: name}}"
            )
        return [spec]
    if isinstance(spec, Mapping):
        index = {value: i for i, value in enumerate(regions.values)}
        for key, name in spec.items():
            if isinstance(key, np.generic):
                key = key.item()
            if key not in index:
                raise ValueError(
                    f"{arg} has key {key!r}, which is not a region value; "
                    f"region values are {regions.values}"
                )
            if name is not None:
                names[index[key]] = str(name)
        return names
    try:
        given = [str(name) for name in spec]
    except TypeError:
        raise TypeError(
            f"{arg} must be a list, a dict or a str, got {type(spec).__name__}"
        ) from None
    if len(given) != n:
        raise ValueError(
            f"{arg} has {len(given)} name(s) but there are {n} region(s) "
            f"{regions.values}; pass one name per sorted region value or a "
            "dict {region: name}"
        )
    return given


def _check_unique_names(names, what):
    seen: dict[str, int] = {}
    for name in names:
        key = name.rstrip()
        if key:
            seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(k for k, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(f"{what} names must be unique; repeated: {duplicates}")


def _region_ids(regions: _Regions) -> list[int]:
    if regions.integral:
        return [int(v) for v in regions.values]
    return list(range(1, len(regions.values) + 1))


# --------------------------------------------------------------------------- #
# Cells -> element blocks
# --------------------------------------------------------------------------- #
@dataclass
class _Cells:
    """Cells of the volume grouped by Exodus element type."""

    points: np.ndarray  # (n_points, 3) float64
    num_dim: int
    n_cells: int
    types: list  # ExoType present, sorted by name
    type_code: np.ndarray  # per cell: index into ``types``
    conn: list  # per type: (n, n_nodes) Exodus-ordered, ascending cell order
    row: np.ndarray  # per cell: row in ``conn[type_code[cell]]``

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])


def _cell_offsets(grid) -> np.ndarray:
    """Offsets (n_cells + 1) into ``cell_connectivity``."""
    if hasattr(type(grid), "cell_offsets"):  # pyvista >= 0.49
        return np.asarray(grid.cell_offsets, dtype=np.int64)
    return np.asarray(grid.offset, dtype=np.int64)


def _gather(connectivity, offsets, rows, n_nodes):
    """(len(rows), n_nodes) point ids of fixed-size cells ``rows``."""
    if rows.size == offsets.size - 1 and connectivity.size == rows.size * n_nodes:
        return connectivity.reshape(-1, n_nodes)  # every cell, uniform size
    return connectivity[offsets[rows][:, None] + np.arange(n_nodes)]


def _type_counts(celltypes, types) -> str:
    counts = np.bincount(celltypes, minlength=256)
    return ", ".join(f"{ct.vtk_type_name(t)}: {counts[t]}" for t in types)


def _is_flat(points: np.ndarray) -> bool:
    extent = points.max(axis=0) - points.min(axis=0)
    diag = float(np.sqrt(np.sum(extent**2)))
    return bool(extent[2] <= _FLAT_RTOL * diag)


def _collect_cells(grid: pv.UnstructuredGrid, num_dim) -> _Cells:
    celltypes = np.asarray(grid.celltypes).astype(np.intp)
    offsets = _cell_offsets(grid)
    connectivity = np.asarray(grid.cell_connectivity, dtype=np.int64)
    present = [int(t) for t in np.unique(celltypes)]

    unsupported = [t for t in present if t not in ct.SUPPORTED_VTK_TYPES]
    if unsupported:
        raise ct.UnsupportedCellTypeError(
            "cannot write these cell types to Exodus: "
            f"{_type_counts(celltypes, unsupported)}. Convert them first "
            "(e.g. mesh.triangulate() for polygons and triangle strips, "
            "mesh.linear_copy() for higher-order cells) or remove them "
            "(e.g. mesh.extract_cells_by_type(...)). Supported: "
            + ", ".join(
                sorted(ct.vtk_type_name(t) for t in ct.SUPPORTED_VTK_TYPES)
            )
        )

    points = np.asarray(grid.points, dtype=np.float64)
    solid = [t for t in present if ct.exo_info_for(t, 3).dim == 3]
    flat = _is_flat(points)
    if num_dim is None:
        num_dim = 2 if (not solid and flat) else 3
    else:
        if isinstance(num_dim, (bool, np.bool_)) or num_dim not in (2, 3):
            raise ValueError(f"num_dim must be None, 2 or 3, got {num_dim!r}")
        num_dim = int(num_dim)
        if num_dim == 2 and solid:
            raise ValueError(
                "num_dim=2 was requested but the mesh has 3D cells "
                f"({_type_counts(celltypes, solid)})"
            )
        if num_dim == 2 and not flat:
            raise ValueError(
                "num_dim=2 needs all points to share one z coordinate; got z in "
                f"[{points[:, 2].min():g}, {points[:, 2].max():g}]"
            )

    info_of = {t: ct.exo_info_for(t, num_dim) for t in present}
    types = sorted({info.name: info for info in info_of.values()}.values(),
                   key=lambda info: info.name)
    lookup = np.zeros(256, dtype=np.int64)
    for t, info in info_of.items():
        lookup[t] = types.index(info)
    type_code = lookup[celltypes]

    sizes = np.diff(offsets)
    row = np.empty(grid.n_cells, dtype=np.int64)
    conns = []
    for k, info in enumerate(types):
        cells_k = np.flatnonzero(type_code == k)
        row[cells_k] = np.arange(cells_k.size)
        bad = sizes[cells_k] != info.n_nodes
        if bad.any():
            first = int(cells_k[bad][0])
            raise ValueError(
                f"{int(bad.sum())} cell(s) written as {info.name} do not have "
                f"{info.n_nodes} points (cell {first} has {int(sizes[first])})"
            )
        vtk_types = [t for t, i in info_of.items() if i is info]
        if len(vtk_types) == 1:
            raw = _gather(connectivity, offsets, cells_k, info.n_nodes)
            conn = raw[:, ct.vtk_to_exo_perm(vtk_types[0])]
        else:
            conn = np.empty((cells_k.size, info.n_nodes), dtype=np.int64)
            sub_types = celltypes[cells_k]
            for t in vtk_types:
                sel = sub_types == t
                raw = _gather(connectivity, offsets, cells_k[sel], info.n_nodes)
                conn[sel] = raw[:, ct.vtk_to_exo_perm(t)]
        conns.append(conn)

    return _Cells(points, num_dim, grid.n_cells, types, type_code, conns, row)


@dataclass
class _Layout:
    """Element blocks in file order."""

    blocks: list  # Block
    infos: list  # ExoType per block
    cells: list  # per block: original cell indices (ascending)
    order: np.ndarray  # file-order element -> original cell index
    offsets: np.ndarray  # first file-order element of each block (+ total)


def _layout_blocks(cells: _Cells, regions: _Regions, block_names) -> _Layout:
    n_types = len(cells.types)
    n_regions = len(regions.values)
    key = regions.code * n_types + cells.type_code
    order = np.argsort(key, kind="stable")
    counts = np.bincount(key, minlength=n_regions * n_types)
    present = np.flatnonzero(counts)
    region_of = present // n_types
    type_of = present % n_types
    types_per_region = np.bincount(region_of, minlength=n_regions)

    if regions.integral and np.all(types_per_region[region_of] == 1):
        ids = [int(regions.values[r]) for r in region_of]
    else:
        ids = list(range(1, present.size + 1))

    base = _resolve_names(block_names, regions, "block_names")
    names = []
    for block_id, r, t in zip(ids, region_of, type_of):
        name = base[r]
        if name is None:
            name = f"block_{block_id}"
        elif types_per_region[r] > 1:
            name = f"{name}_{cells.types[t].name.lower()}"
        names.append(name)
    _check_unique_names(names, "block")

    offsets = np.concatenate([[0], np.cumsum(counts[present])]).astype(np.int64)
    blocks, infos, block_cells = [], [], []
    for b, (block_id, t) in enumerate(zip(ids, type_of)):
        members = order[offsets[b]: offsets[b + 1]]
        info = cells.types[t]
        conn = cells.conn[t][cells.row[members]]
        blocks.append(Block(block_id, names[b], info.written_name, conn))
        infos.append(info)
        block_cells.append(members)
    return _Layout(blocks, infos, block_cells, order, offsets)


# --------------------------------------------------------------------------- #
# Side sets
# --------------------------------------------------------------------------- #
def _surface_corners(surf: pv.UnstructuredGrid):
    """Corner point ids of every surface cell, grouped by corner count.

    Returns ``{k: (cell_indices, corners (m, k))}`` with the corners in cyclic
    order (the orientation of the surface cell).
    """
    celltypes = np.asarray(surf.celltypes).astype(np.intp)
    offsets = _cell_offsets(surf)
    connectivity = np.asarray(surf.cell_connectivity, dtype=np.int64)
    sizes = np.diff(offsets)
    pieces: dict[int, list] = {}
    bad = []
    for t in (int(t) for t in np.unique(celltypes)):
        cells_t = np.flatnonzero(celltypes == t)
        if t in (pv.CellType.POLYGON, pv.CellType.POLY_LINE):
            ok_sizes = (3, 4) if t == pv.CellType.POLYGON else (2,)
            for size in np.unique(sizes[cells_t]):
                sel = cells_t[sizes[cells_t] == size]
                if int(size) not in ok_sizes:
                    bad.append(f"{sel.size} {ct.vtk_type_name(t)} with {size} points")
                    continue
                corners = _gather(connectivity, offsets, sel, int(size))
                pieces.setdefault(int(size), []).append((sel, corners))
            continue
        info = (
            ct.exo_info_for(t, 3) if t in ct.SUPPORTED_VTK_TYPES else None
        )
        if info is None or info.dim not in (1, 2):
            bad.append(f"{cells_t.size} {ct.vtk_type_name(t)}")
            continue
        raw = _gather(connectivity, offsets, cells_t, info.n_nodes)
        corners = raw[:, ct.vtk_to_exo_perm(t)[: info.n_corners]]
        pieces.setdefault(info.n_corners, []).append((cells_t, corners))
    if bad:
        raise ValueError(
            "surface cells must be faces (triangles, quads) or edges (lines); "
            f"got {', '.join(bad)}. Triangulate polygons and strips first "
            "(surface.triangulate())."
        )
    return {
        k: (
            np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
        )
        for k, parts in pieces.items()
    }


def _match_points(query: np.ndarray, points: np.ndarray, tol: float):
    """Map ``query`` points to volume nodes within ``tol``.

    Returns ``(node, canon)``: the node of every query point and, when some
    query points have several coincident volume nodes (duplicate nodes), an
    array mapping every volume node to the lowest coincident node index
    (``None`` otherwise). ``node`` is already canonical.
    """
    tree = cKDTree(points, balanced_tree=False, compact_nodes=False)
    # cKDTree compares squared distances: keep the bound's square > 0 so that
    # tol=0 still finds exact matches (filtered with ``dist <= tol`` below).
    bound = max(tol * (1.0 + 1e-9), 1e-150)
    n_points = points.shape[0]
    k = min(4, n_points)
    while True:
        dist, idx = tree.query(query, k=k, distance_upper_bound=bound)
        dist = dist.reshape(len(query), k)
        idx = idx.reshape(len(query), k)
        if k >= n_points or not np.isfinite(dist[:, -1]).any():
            break
        k = min(2 * k, n_points)
    found = dist <= tol
    missing = ~found[:, 0]
    if missing.any():
        far, _ = tree.query(query[missing])
        raise ValueError(
            f"{int(missing.sum())} surface point(s) are farther than "
            f"side_set_tolerance={tol:.6g} from every volume node (largest "
            f"distance {far.max():.6g}): the surface does not lie on the volume "
            "mesh. Pass a larger side_set_tolerance if it is only approximately "
            "on the mesh."
        )
    if not found[:, 1:].any():
        return idx[:, 0], None
    masked = np.where(found, idx, n_points)
    rep = masked.min(axis=1)
    canon = np.arange(n_points)
    rows, cols = np.nonzero(found)
    np.minimum.at(canon, idx[rows, cols], rep[rows])
    return canon[rep], canon


def _side_candidates(cells: _Cells, layout: _Layout, mark, canon):
    """Element sides whose corner nodes are all surface nodes, by corner count.

    Returns ``{k: (sorted corner nodes (c, k), file-order element, side)}``.
    """
    pieces: dict[int, list] = {}
    for b, (block, info) in enumerate(zip(layout.blocks, layout.infos)):
        # 3D files: sides of solid elements; 2D files: edges of 2D elements.
        if info.dim != cells.num_dim or not info.sides or block.n_elem == 0:
            continue
        corners = block.conn[:, : info.n_corners]
        if canon is not None:
            corners = canon[corners]
        marked = mark[corners]
        min_len = min(len(side) for side in info.sides)
        rows = np.flatnonzero(marked.sum(axis=1) >= min_len)
        if rows.size == 0:
            continue
        corners, marked = corners[rows], marked[rows]
        for s, side in enumerate(info.sides):
            side = list(side)
            hit = np.flatnonzero(marked[:, side].all(axis=1))
            if hit.size == 0:
                continue
            keys = np.sort(corners[hit][:, side], axis=1)
            elems = layout.offsets[b] + rows[hit]
            pieces.setdefault(len(side), []).append(
                (keys, elems, np.full(hit.size, s + 1, dtype=np.int64))
            )
    return {
        k: tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
        for k, parts in pieces.items()
    }


def _group_rows(a: np.ndarray, b: np.ndarray):
    """Equal-row group ids of the rows of ``a`` and of ``b`` (same width)."""
    rows = np.concatenate([a, b])
    order = np.lexsort(rows.T[::-1])
    srt = rows[order]
    new = np.ones(len(rows), dtype=bool)
    new[1:] = (srt[1:] != srt[:-1]).any(axis=1)
    gid = np.empty(len(rows), dtype=np.int64)
    gid[order] = np.cumsum(new) - 1
    return gid[: len(a)], gid[len(a):], int(gid.max()) + 1 if len(rows) else 0


def _element_centers(cells: _Cells, layout: _Layout, elems: np.ndarray):
    """Centres (mean of the corner nodes) of file-order elements ``elems``."""
    centers = np.empty((elems.size, 3))
    blk = np.searchsorted(layout.offsets, elems, side="right") - 1
    for b in np.unique(blk):
        sel = blk == b
        info = layout.infos[b]
        local = elems[sel] - layout.offsets[b]
        corners = layout.blocks[b].conn[local, : info.n_corners]
        centers[sel] = cells.points[corners].mean(axis=1)
    return centers


def _face_normals(pts: np.ndarray) -> np.ndarray:
    """Normals of faces / edges given their corner coordinates ``(m, k, 3)``."""
    k = pts.shape[1]
    if k == 2:  # in-plane normal (dy, -dx) of the edge direction
        d = pts[:, 1] - pts[:, 0]
        return np.stack([d[:, 1], -d[:, 0], np.zeros(len(d))], axis=1)
    if k == 3:
        return np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    return np.cross(pts[:, 2] - pts[:, 0], pts[:, 3] - pts[:, 1])


def _choose_owner(cells, layout, surf_points, corners, cand_elem, corder, start,
                  mult):
    """Pick one candidate per surface face shared by several elements.

    The owner is the element the surface normal points out of; ties (none or
    several pointing out, degenerate normals) go to the lowest file-order
    element, i.e. the lower block id, then the lower element index.
    """
    pts = surf_points[corners]
    face_center = pts.mean(axis=1)
    normal = _face_normals(pts)
    n_norm = np.linalg.norm(normal, axis=1)
    big = np.int64(layout.order.size + 1)
    best = np.full(len(corners), np.iinfo(np.int64).max, dtype=np.int64)
    chosen = np.empty(len(corners), dtype=np.int64)
    for j in range(int(mult.max())):
        valid = j < mult
        cand = corder[start + np.minimum(j, mult - 1)]
        elem = cand_elem[cand]
        offset = face_center - _element_centers(cells, layout, elem)
        score = np.einsum("ij,ij->i", offset, normal)
        out = score > _NORMAL_RTOL * np.linalg.norm(offset, axis=1) * n_norm
        rank = np.where(out, 0, big) + elem
        rank[~valid] = np.iinfo(np.int64).max
        better = rank < best
        chosen[better] = cand[better]
        best[better] = rank[better]
    return chosen


def _build_side_sets(surface, cells: _Cells, layout: _Layout, region_key,
                     names_spec, tolerance, notes) -> list:
    surf = _as_unstructured(surface, "surface")
    if surf.n_cells == 0:
        notes.append("the surface has no cells; no side sets were written")
        return []
    regions = _regions_of(surf, region_key, "surface")

    diag = float(np.linalg.norm(np.ptp(cells.points, axis=0)))
    if tolerance is None:
        tol = _SIDE_SET_RTOL * diag
    else:
        tol = float(tolerance)
        if not np.isfinite(tol) or tol < 0:
            raise ValueError(
                f"side_set_tolerance must be a finite number >= 0, got {tolerance!r}"
            )

    groups = _surface_corners(surf)
    surf_points = np.asarray(surf.points, dtype=np.float64)
    used = np.unique(np.concatenate([c.reshape(-1) for _, c in groups.values()]))
    node_of = np.full(surf.n_points, -1, dtype=np.int64)
    node_of[used], canon = _match_points(surf_points[used], cells.points, tol)
    mark = np.zeros(cells.n_points, dtype=bool)
    mark[node_of[used]] = True
    candidates = _side_candidates(cells, layout, mark, canon)

    elem_of = np.full(surf.n_cells, -1, dtype=np.int64)
    side_of = np.zeros(surf.n_cells, dtype=np.int64)
    for k, (surf_cells, corners) in groups.items():
        keys = np.sort(node_of[corners], axis=1)
        if k not in candidates:
            continue
        cand_keys, cand_elem, cand_side = candidates[k]
        gc, gs, n_groups = _group_rows(cand_keys, keys)
        counts = np.bincount(gc, minlength=n_groups)
        corder = np.argsort(gc, kind="stable")
        starts = np.cumsum(counts) - counts
        mult = counts[gs]
        hit = mult > 0
        chosen = corder[starts[gs[hit]]]
        shared = np.flatnonzero(mult[hit] > 1)
        if shared.size:
            sub = np.flatnonzero(hit)[shared]
            chosen[shared] = _choose_owner(
                cells, layout, surf_points, corners[sub], cand_elem, corder,
                starts[gs[sub]], mult[sub],
            )
        elem_of[surf_cells[hit]] = cand_elem[chosen]
        side_of[surf_cells[hit]] = cand_side[chosen]

    unmatched = np.flatnonzero(elem_of < 0)
    if unmatched.size:
        targets = (
            "faces of the 3D elements" if cells.num_dim == 3
            else "edges of the 2D elements (use line cells)"
        )
        raise ValueError(
            f"{unmatched.size} of {surf.n_cells} surface cell(s) do not match a "
            f"side of any element (first: surface cell {int(unmatched[0])}). "
            f"Side sets are matched to the {targets}; shell, bar and sphere "
            "elements are not side-set targets."
        )

    ids = _region_ids(regions)
    base = _resolve_names(names_spec, regions, "side_set_names")
    names = [f"side_set_{i}" if n is None else n for i, n in zip(ids, base)]
    _check_unique_names(names, "side set")

    order = np.argsort(regions.code, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(
        regions.code, minlength=len(ids)))])
    side_sets, dropped = [], []
    for r, (set_id, name) in enumerate(zip(ids, names)):
        members = order[bounds[r]: bounds[r + 1]]
        elems, sides = elem_of[members], side_of[members]
        _, first = np.unique(elems * 8 + sides, return_index=True)
        if first.size < members.size:
            dropped.append(f"{name!r}: {members.size - first.size}")
            first.sort()
            elems, sides = elems[first], sides[first]
        side_sets.append(SideSet(set_id, name, elems, sides))
    if dropped:
        notes.append(
            "dropped repeated faces from side sets (each element side is "
            f"written once per set): {', '.join(dropped)}"
        )
    return side_sets


# --------------------------------------------------------------------------- #
# Node sets
# --------------------------------------------------------------------------- #
def _is_int_key(key) -> bool:
    return isinstance(key, (int, np.integer)) and not isinstance(
        key, (bool, np.bool_)
    )


def _node_indices(values, n_points, label) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind == "b":
        if arr.shape != (n_points,):
            raise ValueError(
                f"node set {label}: a boolean mask must have one entry per point "
                f"({n_points}), got shape {arr.shape}"
            )
        return np.flatnonzero(arr)
    arr = arr.reshape(-1)
    if arr.size and arr.dtype.kind == "f" and np.all(arr == np.round(arr)):
        arr = arr.astype(np.int64)
    if arr.size and arr.dtype.kind not in "iu":
        raise TypeError(f"node set {label}: indices must be integers, got {arr.dtype}")
    arr = arr.astype(np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= n_points):
        raise ValueError(
            f"node set {label}: indices must be in 0..{n_points - 1}, got "
            f"{arr.min()}..{arr.max()}"
        )
    if np.unique(arr).size != arr.size:
        raise ValueError(f"node set {label} contains repeated node indices")
    return arr


def _build_node_sets(spec, n_points) -> list:
    if spec is None:
        return []
    if isinstance(spec, Mapping):
        items = list(spec.items())
        keys = [k for k, _ in items]
        if keys and all(_is_int_key(k) and 1 <= int(k) <= _INT32_MAX for k in keys):
            ids = [int(k) for k in keys]
            names = [f"node_set_{i}" for i in ids]
        else:
            ids = list(range(1, len(items) + 1))
            names = [str(k) for k in keys]
        values = [v for _, v in items]
    elif isinstance(spec, (str, bytes)) or not hasattr(spec, "__iter__"):
        raise TypeError(
            f"node_sets must be a dict {{name: indices}} or a list of index "
            f"arrays, got {type(spec).__name__}"
        )
    else:
        values = list(spec)
        ids = list(range(1, len(values) + 1))
        names = [f"node_set_{i}" for i in ids]
    _check_unique_names(names, "node set")
    return [
        NodeSet(i, name, _node_indices(v, n_points, repr(name)))
        for i, name, v in zip(ids, names, values)
    ]


# --------------------------------------------------------------------------- #
# Variables
# --------------------------------------------------------------------------- #
def _check_mapping(value, arg) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{arg} must be a dict {{name: values}}, got {type(value).__name__}"
        )
    for name in value:
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"{arg} names must be non-empty strings, got {name!r}")
    return dict(value)


def _times(times):
    if times is None:
        return None
    arr = np.atleast_1d(np.asarray(times, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(f"times must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("times is empty; omit it for a single step at t=0.0")
    if not np.all(np.isfinite(arr)):
        raise ValueError("times contains NaN or infinite values")
    return arr


def _numeric(values, what) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind not in "biuf":
        raise TypeError(f"{what} must be numeric, got dtype {arr.dtype}")
    return arr.astype(np.float64, copy=False)


def _field(values, n_steps, n, name, kind, implicit_times) -> np.ndarray:
    """Time-varying field as ``(n_steps, n, n_components)`` float64."""
    arr = _numeric(values, f"{kind} field {name!r}")
    if arr.ndim == 1 and n_steps == 1 and arr.shape[0] == n:
        return arr.reshape(1, n, 1)
    if arr.ndim == 2:
        if arr.shape == (n_steps, n):
            return arr[:, :, None]
        if n_steps == 1 and arr.shape[0] == n:
            return arr[None]
    if arr.ndim == 3 and arr.shape[:2] == (n_steps, n):
        return arr
    hint = (
        " (times was not given, so there is one step; pass times=... for a "
        "history)" if implicit_times else ""
    )
    raise ValueError(
        f"{kind} field {name!r} has shape {arr.shape}; expected ({n_steps}, {n}) "
        f"or ({n_steps}, {n}, n_components){hint}"
    )


def _components(name, arr):
    """``(scalar name, values)`` of every component of ``arr[..., n_comp]``."""
    n_comp = arr.shape[-1]
    if n_comp == 1:
        return [(name, arr[..., 0])]
    return [
        (name + suffix, arr[..., c])
        for c, suffix in enumerate(component_suffixes(n_comp))
    ]


def _static(values, name, what, notes):
    """A point / cell array as float64 ``(n, n_components)``, or None (skipped)."""
    arr = np.asarray(values)
    kind = arr.dtype.kind
    if kind == "c":
        notes.append(f"skipped {what} array {name!r}: complex values are not supported")
        return None
    if kind not in "biuf":
        notes.append(f"skipped {what} array {name!r}: non-numeric dtype {arr.dtype}")
        return None
    arr = arr.reshape(arr.shape[0], -1).astype(np.float64, copy=False)
    if arr.shape[1] == 0:
        return None
    longest = max(len(s.rstrip().encode("utf-8")) for s, _ in _components(name, arr))
    if longest > MAX_NAME_LENGTH:
        notes.append(
            f"skipped {what} array {name[:40]!r}...: the variable name is longer "
            f"than Exodus allows ({MAX_NAME_LENGTH} bytes)"
        )
        return None
    return arr


def _add(target: dict, name: str, values, source: str):
    if name.rstrip() in target:
        raise ValueError(
            f"variable name {name!r} (from {source}) is used twice; rename one of "
            "the arrays"
        )
    target[name.rstrip()] = values


def _all_nan(values: np.ndarray) -> bool:
    if values.size and not np.isnan(values.flat[0]):
        return False  # the common case, without scanning the array
    if values.ndim == 2 and values.shape[0] > 1 and values.strides[0] == 0:
        values = values[:1]  # a static array broadcast over the steps
    return bool(np.isnan(values).all())


def _drop_undefined_blocks(elem_vars: dict, groups: list) -> None:
    """Leave out the blocks on which a source array is NaN everywhere.

    A block is dropped from all component variables of an array together
    (VTK's reader cannot handle the components of one vector stored on
    different blocks), and only if the array is defined on some block.
    """
    for names in groups:
        per_block = [elem_vars[name] for name in names]
        block_ids = list(per_block[0])
        undefined = [
            block_id for block_id in block_ids
            if all(_all_nan(values[block_id]) for values in per_block)
        ]
        if undefined and len(undefined) < len(block_ids):
            for values in per_block:
                for block_id in undefined:
                    del values[block_id]


def _build_variables(grid, cells: _Cells, layout: _Layout, region_key, *, times,
                     node_fields, element_fields, global_fields, save_node_arrays,
                     save_cell_arrays, notes):
    node_fields = _check_mapping(node_fields, "node_fields")
    element_fields = _check_mapping(element_fields, "element_fields")
    global_fields = _check_mapping(global_fields, "global_fields")
    times_arr = _times(times)
    implicit = times_arr is None
    n_steps = 1 if implicit else times_arr.size
    n_points, n_cells = cells.n_points, cells.n_cells

    # time-varying fields (their names win over static arrays)
    node_tv: dict = {}
    for name, values in node_fields.items():
        arr = _field(values, n_steps, n_points, name, "node", implicit)
        for scalar, col in _components(name, arr):
            _add(node_tv, scalar, col, f"node field {name!r}")
    elem_tv: dict = {}
    elem_groups: list = []  # scalar variable names of each source array
    for name, values in element_fields.items():
        arr = _field(values, n_steps, n_cells, name, "element", implicit)
        parts = _components(name, arr)
        for scalar, col in parts:
            _add(elem_tv, scalar, col, f"element field {name!r}")
        elem_groups.append([s.rstrip() for s, _ in parts])
    global_vars: dict = {}
    for name, values in global_fields.items():
        arr = _numeric(values, f"global field {name!r}")
        if arr.ndim == 0 and n_steps == 1:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1 and arr.shape[0] == n_steps:
            arr = arr[:, None]
        elif not (arr.ndim == 2 and arr.shape[0] == n_steps):
            raise ValueError(
                f"global field {name!r} has shape {arr.shape}; expected "
                f"({n_steps},) or ({n_steps}, n_components)"
            )
        for scalar, col in _components(name, arr):
            _add(global_vars, scalar, col, f"global field {name!r}")

    # static arrays, broadcast over all steps
    node_static: dict = {}
    if save_node_arrays:
        for name in grid.point_data.keys():
            if name in _EXCLUDED_ARRAYS or name in node_fields:
                continue
            arr = _static(grid.point_data[name], name, "point", notes)
            if arr is None:
                continue
            parts = _components(name, arr)
            clash = [s for s, _ in parts if s.rstrip() in node_tv or
                     s.rstrip() in node_static]
            if clash:
                notes.append(
                    f"skipped point array {name!r}: variable name(s) {clash} "
                    "already used"
                )
                continue
            for scalar, col in parts:
                node_static[scalar.rstrip()] = np.broadcast_to(col, (n_steps, n_points))
    elem_static: dict = {}
    if save_cell_arrays:
        for name in grid.cell_data.keys():
            if (name in _EXCLUDED_ARRAYS or name == region_key
                    or name in element_fields):
                continue
            arr = _static(grid.cell_data[name], name, "cell", notes)
            if arr is None:
                continue
            parts = _components(name, arr)
            clash = [s for s, _ in parts if s.rstrip() in elem_tv or
                     s.rstrip() in elem_static]
            if clash:
                notes.append(
                    f"skipped cell array {name!r}: variable name(s) {clash} "
                    "already used"
                )
                continue
            for scalar, col in parts:
                elem_static[scalar.rstrip()] = col
            elem_groups.append([s.rstrip() for s, _ in parts])

    node_vars = {**node_static, **node_tv}
    elem_vars = {}
    for name, col in elem_static.items():  # (n_cells,) original order
        ordered = col[layout.order]
        elem_vars[name] = {
            block.id: np.broadcast_to(
                ordered[layout.offsets[b]: layout.offsets[b + 1]],
                (n_steps, block.n_elem),
            )
            for b, block in enumerate(layout.blocks)
        }
    for name, values in elem_tv.items():  # (n_steps, n_cells) original order
        ordered = values[:, layout.order]
        elem_vars[name] = {
            block.id: ordered[:, layout.offsets[b]: layout.offsets[b + 1]]
            for b, block in enumerate(layout.blocks)
        }
    _drop_undefined_blocks(elem_vars, elem_groups)

    has_vars = bool(node_vars or elem_vars or global_vars)
    if times_arr is None:
        times_arr = np.zeros(1 if has_vars else 0)
    return times_arr, node_vars, elem_vars, global_vars
