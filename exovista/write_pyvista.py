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
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkDataObject

from . import _celltypes as ct
from ._backend import write_model
from ._model import (
    MAX_NAME_LENGTH,
    Block,
    ExodusModel,
    NodeSet,
    SideSet,
    component_groups,
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
        "ImplicitElementId",
        "ImplicitNodeId",
        "SourceElementId",
        "SourceElementSide",
    }
)

#: Top-level keys of the MultiBlock that VTK's Exodus reader (``pv.read``,
#: ``pv.ExodusIIReader(...).read()``) returns.
_EXODUS_READER_KEYS = frozenset(
    {
        "Element Blocks",
        "Face Blocks",
        "Edge Blocks",
        "Element Sets",
        "Side Sets",
        "Face Sets",
        "Edge Sets",
        "Node Sets",
    }
)
#: Internal region array of side sets converted from VTK reader output.
_SET_KEY = "__exovista_set_id__"
#: Largest integers stored exactly by float32 / float64.
_EXACT_INT = {4: 2**24, 8: 2**53}

_PRECISIONS = {"double": 8, "float64": 8, "single": 4, "float32": 4}

#: z extent (relative to the bounding-box diagonal) below which a mesh is flat.
_FLAT_RTOL = 1e-12
#: Default side-set matching tolerance, relative to the bounding-box diagonal.
_SIDE_SET_RTOL = 1e-6
#: Volume nodes closer than this (relative to the bounding-box diagonal) are
#: duplicates of one node (an unmerged mesh) when matching side sets.
_COINCIDENT_RTOL = 1e-12
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
        (an existing file is left untouched). If ``filename`` is a symbolic
        link, the file it points to is replaced; an existing file keeps its
        permissions. The time dimension has a fixed length (fast to read),
        so time steps cannot be appended to the file afterwards.
    volume : pyvista.DataSet or pyvista.MultiBlock
        The mesh. Any dataset is accepted (it is cast with
        ``cast_to_unstructured_grid()`` when needed). For a ``MultiBlock``,
        every non-empty leaf becomes its own region (region values ``1..N`` in
        leaf order, block names default to the leaf keys; a leaf's own
        ``region_key`` array is ignored); points with identical coordinates
        are merged (as ``combine(merge_points=True)``; to keep coincident
        nodes of touching bodies apart, give each leaf a region array and
        pass ``multiblock.combine(merge_points=False)`` instead) and "original
        cell order" below means the order of the combined mesh (leaf after
        leaf). Point and cell arrays of only some leaves are kept: they are
        NaN on the other leaves' points and cells (so the cell arrays are not
        stored on those blocks, see Notes). The output of VTK's Exodus reader
        (``pv.read("x.exo")``, ``pv.ExodusIIReader(...).read()`` or its
        ``"Element Blocks"``) is recognised: its element blocks keep their
        ids and names, its side sets and node sets are written (unless
        ``surface`` / ``node_sets`` are given), its vectors glued by VTK
        (``vel_`` from ``vel_X, vel_Y, vel_Z``; in 2D files padded with a
        zero third component) are written under their original names, and
        its other entities are ignored with a warning. Its nodes are merged
        by ``GlobalNodeId`` when present (enable it with
        ``reader.reader.GenerateGlobalNodeIdArrayOn()`` to keep coincident
        nodes apart and the file's node order), else by coordinates. Note
        that VTK's reader moves the points by displacement variables
        (``DISPL*``) unless told otherwise. The input is never modified.
    surface : pyvista.DataSet, optional
        Boundary faces (3D meshes) or edges (``num_dim=2`` meshes) to write as
        side sets, e.g. ``volume.extract_surface()`` or a subset of it (for
        quadratic cells use ``volume.extract_surface(nonlinear_subdivision=0)``:
        by default pyvista splits each quadratic face into linear faces that
        match no element side). Each surface cell must match the side of an
        element (see Notes). One side set is written per value of
        ``surface.cell_data[region_key]`` (or a single set without it).
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
        matched to (its nearest node). Default ``1e-6`` times the
        bounding-box diagonal.
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
    reader (and its ``ImplicitElementId``, ``ImplicitNodeId``,
    ``SourceElementId`` and ``SourceElementSide``). Integer arrays beyond
    ``2**53`` (``2**24`` with ``precision="single"``) lose exactness and are
    reported with a warning, as are variables named like the components of
    another array (a scalar ``U_Z`` next to a 2-component ``U``: readers
    would regroup them differently).

    **Undefined values.** An element variable that is NaN on every element
    of a block (all components, all steps) is not stored on that block
    (Exodus truth table), which is how :func:`exovista.read_exo` reports
    variables that are undefined on a block; it stays stored everywhere if
    it is NaN on every block. The components of an array, and all variables
    VTK's reader glues into one array (``<name>X/Y/Z``), are decided
    together (VTK crashes on a glued array stored on different blocks), so
    a block is left out only when all of them are NaN there. meshio (<= 5.3)
    cannot read files whose truth table leaves out a block; fill such
    arrays with a value instead of NaN if the file is meant for meshio.

    **Writing a mesh read by** :func:`exovista.read_exo`: pass
    ``region_key="exo_block_id"`` to keep its element blocks (and their ids)
    and ``block_names`` to keep their names; ``region_key`` also names the
    region array of ``surface``.

    **Side sets.** Each surface point is matched to its nearest volume node
    (within ``side_set_tolerance``; exact duplicates of that node count as
    the same node) and each surface cell's corner nodes are looked up
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

    grid, regions, vtk_sets = _prepare_volume(volume, region_key, notes)
    cells = _collect_cells(grid, num_dim)
    layout = _layout_blocks(cells, regions, block_names)
    side_sets = []
    if surface is not None:
        side_sets = _build_side_sets(
            surface, cells, layout, region_key, side_set_names,
            side_set_tolerance, notes,
        )
    nsets = _build_node_sets(node_sets, cells.n_points)
    if vtk_sets is not None:
        side_sets, nsets = _use_vtk_sets(
            vtk_sets, surface, node_sets, side_sets, nsets, cells, layout,
            side_set_names, side_set_tolerance, notes,
        )
    times_arr, node_vars, elem_vars, global_vars = _build_variables(
        grid, cells, layout, region_key if vtk_sets is None else None,
        times=times,
        node_fields=node_fields,
        element_fields=element_fields,
        global_fields=global_fields,
        save_node_arrays=save_node_arrays,
        save_cell_arrays=save_cell_arrays,
        float_word_size=float_word_size,
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


def _is_exodus_reader_output(mb: pv.MultiBlock) -> bool:
    """True for the MultiBlock layout of VTK's Exodus reader."""
    keys = [mb.get_block_name(i) for i in range(mb.n_blocks)]
    return (
        "Element Blocks" in keys
        and set(keys) <= _EXODUS_READER_KEYS
        and isinstance(mb["Element Blocks"], pv.MultiBlock)
    )


def _merge_exact(points: np.ndarray):
    """Merge points with identical coordinates (like ``merge_points=True``).

    Returns ``(first, inverse)``: the input index of each merged point, in
    order of first occurrence, and the merged index of every input point.
    """
    n = points.shape[0]
    order = np.lexsort(points.T[::-1])  # stable: ties keep input order
    srt = points[order]
    new = np.ones(n, dtype=bool)
    new[1:] = np.any(srt[1:] != srt[:-1], axis=1)
    group = np.empty(n, dtype=np.int64)
    group[order] = np.cumsum(new) - 1
    first = order[new]  # smallest input index of every group
    rank = np.empty(first.size, dtype=np.int64)
    by_first = np.argsort(first, kind="stable")
    rank[by_first] = np.arange(first.size)
    return first[by_first], rank[group]


def _combine_leaves(leaves: list, region_key, notes, node_key=None):
    """One UnstructuredGrid from MultiBlock leaves (leaf after leaf).

    Points with equal coordinates are merged (``merge_points=True``), or,
    with ``node_key``, points with equal ids in that point array. Unlike
    ``MultiBlock.combine``, arrays that only some leaves have are kept: cell
    arrays are NaN on the other leaves' cells, point arrays NaN on points no
    leaf with the array has (a merged point takes the first leaf's value that
    is not NaN). Each leaf's ``region_key`` array is ignored (the leaves are
    the regions).
    """
    grids = [_as_unstructured(leaf, "volume") for leaf in leaves]
    points = np.concatenate(
        [np.asarray(g.points, dtype=np.float64).reshape(-1, 3) for g in grids]
    )
    starts = np.cumsum([0] + [g.n_points for g in grids])
    if node_key is not None and all(node_key in g.point_data for g in grids):
        ids = np.concatenate(
            [np.asarray(g.point_data[node_key]).reshape(-1) for g in grids]
        )
        _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
    else:
        first, inverse = _merge_exact(points)

    celltypes = np.concatenate([np.asarray(g.celltypes, dtype=np.uint8)
                                for g in grids])
    offsets, connectivity, shift = [np.zeros(1, np.int64)], [], 0
    for g, start in zip(grids, starts):
        off = _cell_offsets(g)
        offsets.append(off[1:] + shift)
        shift += int(off[-1])
        connectivity.append(
            inverse[np.asarray(g.cell_connectivity, dtype=np.int64) + start]
        )
    grid = pv.UnstructuredGrid()
    grid.points = points[first]
    cell_array = pv.CellArray.from_arrays(
        np.concatenate(offsets), np.concatenate(connectivity), deep=True
    )
    grid.SetCells(numpy_to_vtk(celltypes, deep=True), cell_array)

    if region_key is not None and any(
        region_key in g.cell_data.keys()
        and np.unique(np.asarray(g.cell_data[region_key])).size > 1
        for g in grids
    ):
        notes.append(
            f"ignored the {region_key!r} cell array of the MultiBlock leaves: "
            "each leaf is one region (combine the leaves yourself, e.g. "
            "multiblock.combine(), to keep their own regions)"
        )
    n_cells = [g.n_cells for g in grids]
    for name, values in _union_arrays(
        [g.cell_data for g in grids], n_cells, "cell", notes, skip={region_key}
    ):
        grid.cell_data[name] = values
    leaf_points = [inverse[a:b] for a, b in zip(starts[:-1], starts[1:])]
    for name, parts in _union_arrays(
        [g.point_data for g in grids], [g.n_points for g in grids], "point",
        notes, combine=False,
    ):
        grid.point_data[name] = _merge_point_values(parts, leaf_points, first.size)
    return grid


def _union_arrays(datas: list, sizes: list, what: str, notes, skip=(),
                  combine=True):
    """Arrays of any leaf, as ``(name, values)``.

    With ``combine`` the per-leaf values are concatenated (NaN for leaves
    without the array), else ``values`` is the list of per-leaf arrays (None
    for leaves without it). Arrays whose leaves disagree on the number of
    components, and non-numeric arrays missing from some leaf, are skipped
    with a note.
    """
    names = list(dict.fromkeys(
        str(name) for data in datas for name in data.keys() if name not in skip
    ))
    for name in names:
        parts = [np.asarray(d[name]) if name in d.keys() else None for d in datas]
        present = [p for p in parts if p is not None]
        widths = {p.reshape(p.shape[0], -1).shape[1] for p in present}
        if len(widths) != 1:
            notes.append(
                f"skipped {what} array {name!r}: the MultiBlock leaves give it "
                f"different numbers of components {sorted(widths)}"
            )
            continue
        width = widths.pop()
        missing = len(present) < len(parts)
        numeric = all(p.dtype.kind in "biuf" for p in present)
        if missing and not numeric:
            notes.append(
                f"skipped {what} array {name!r}: it is not numeric and not on "
                "every MultiBlock leaf"
            )
            continue
        if missing:
            dtype = np.result_type(np.float64, *[p.dtype for p in present])
            parts = [
                np.full((n, width), np.nan, dtype=dtype) if p is None
                else p.reshape(n, width).astype(dtype, copy=False)
                for p, n in zip(parts, sizes)
            ]
        else:
            parts = [p.reshape(n, width) for p, n in zip(parts, sizes)]
        if combine:
            values = np.concatenate(parts)
            yield name, values[:, 0] if width == 1 else values
        else:
            yield name, parts


def _merge_point_values(parts: list, leaf_points: list, n_points: int):
    """Point array of the merged points: the first non-NaN leaf value wins."""
    width = parts[0].shape[1]
    dtype = np.result_type(*[p.dtype for p in parts])
    floating = dtype.kind in "fc"
    out = np.full((n_points, width), np.nan if floating else 0, dtype=dtype)
    for values, index in zip(parts[::-1], leaf_points[::-1]):
        if floating:
            keep = ~np.isnan(values).all(axis=1)
            values, index = values[keep], index[keep]
        out[index] = values
    return out[:, 0] if width == 1 else out


def _vtk_padded_2d(leaves: list) -> set:
    """``(kind, name)`` of the arrays VTK padded from 2 to 3 components.

    In a 2D file VTK's reader returns 2-component vectors with a third
    component of zeros. For a flat mesh at z = 0 without 3D cells (written
    as a 2D file again), 3-component arrays whose third component is zero on
    every leaf are taken to be such padded vectors.
    """
    for leaf in leaves:
        if np.any(np.asarray(leaf.points)[:, 2] != 0):
            return set()
        if any(ct.exo_info_for(int(t), 3).dim == 3
               for t in np.unique(np.asarray(leaf.celltypes))
               if int(t) in ct.SUPPORTED_VTK_TYPES):
            return set()
    padded: dict = {}
    for leaf in leaves:
        for kind, data in (("point", leaf.point_data), ("cell", leaf.cell_data)):
            for name in data.keys():
                values = np.asarray(data[name])
                zero = (
                    values.ndim == 2 and values.shape[1] == 3
                    and values.dtype.kind in "biuf" and not np.any(values[:, 2])
                )
                key = (kind, name)
                padded[key] = padded.get(key, True) and zero
    return {key for key, zero in padded.items() if zero}


def _split_vtk_components(leaf: pv.DataSet, padded=frozenset()) -> pv.DataSet:
    """A shallow copy of a VTK-reader leaf with its glued vectors split.

    VTK's Exodus reader glues variables ``<b>X, <b>Y[, <b>Z]`` into one array
    named ``<b>`` (e.g. ``vel_X`` -> ``vel_``, ``t_XX/t_XY/t_XZ`` -> ``t_X``);
    writing the scalars back under their original names makes the round trip
    exact instead of renaming them ``vel__X`` etc. ``padded`` names the
    3-component arrays that are 2-component vectors padded by VTK.
    """
    out = leaf.copy(deep=False)
    for kind, data in (("point", out.point_data), ("cell", out.cell_data)):
        for name in list(data.keys()):
            values = np.asarray(data[name])
            if values.ndim != 2 or values.shape[1] not in (2, 3):
                continue
            if values.dtype.kind not in "biuf" or name in _EXCLUDED_ARRAYS:
                continue
            n = 2 if (kind, name) in padded else values.shape[1]
            new = [name + c for c in "XYZ"[:n]]
            if any(c in data.keys() for c in new):
                continue
            del data[name]
            for k, component in enumerate(new):
                data[component] = np.ascontiguousarray(values[:, k])
    return out


def _leaf_id(leaf: pv.DataSet, field: str | None):
    """Exodus id of a VTK-reader leaf (``ObjectId`` or a field array)."""
    if "ObjectId" in leaf.cell_data.keys():
        ids = np.unique(np.asarray(leaf.cell_data["ObjectId"]))
        if ids.size == 1:
            return int(ids[0])
    if field and field in leaf.field_data.keys():
        ids = np.asarray(leaf.field_data[field]).reshape(-1)
        if ids.size == 1 and ids.dtype.kind in "iu":
            return int(ids[0])
    return None


#: Names VTK's reader gives unnamed blocks and sets.
_VTK_UNNAMED = re.compile(r"^Unnamed (block|set) ID: -?\d+")


def _vtk_names(leaves: list) -> list:
    """Leaf names, None for VTK's made-up names of unnamed entities."""
    names = [None if _VTK_UNNAMED.match(name) else name for name, _ in leaves]
    given = _unique_strings([n for n in names if n is not None])
    it = iter(given)
    return [None if n is None else next(it) for n in names]


def _vtk_ids(leaves: list, field: str | None) -> list | None:
    ids = [_leaf_id(leaf, field) for _, leaf in leaves]
    if None in ids or len(set(ids)) != len(ids):
        return None
    return ids


@dataclass
class _VtkSets:
    """Side and node sets found in VTK Exodus-reader output."""

    side_leaves: list  # (name, dataset)
    node_leaves: list  # (name, dataset)
    node_ids: np.ndarray | None  # global node id of each merged point


def _from_exodus_reader(blocks: pv.MultiBlock, mb, notes):
    """Volume, regions and sets of VTK Exodus-reader output.

    ``blocks`` is its "Element Blocks" MultiBlock, ``mb`` the whole output
    (None when only the element blocks were passed: then there are no sets).
    """
    leaves = _leaves(blocks, [])
    if not leaves:
        raise ValueError("cannot write an empty mesh: 'Element Blocks' has no cells")
    padded = _vtk_padded_2d([leaf for _, leaf in leaves])
    leaves = [(name, _split_vtk_components(leaf, padded)) for name, leaf in leaves]
    names = _vtk_names(leaves)
    ids = _vtk_ids(leaves, "ElementBlockIds")
    # (the blocks are the regions: a "region" array is an ordinary variable)
    grid = _combine_leaves(
        [leaf for _, leaf in leaves], None, notes, node_key="GlobalNodeId"
    )
    node_ids = None
    if "GlobalNodeId" in grid.point_data.keys():
        node_ids = np.asarray(grid.point_data["GlobalNodeId"]).reshape(-1)
    labels = np.repeat(np.arange(len(leaves)), [leaf.n_cells for _, leaf in leaves])
    if ids is None:
        values, integral = list(range(1, len(leaves) + 1)), True
        code = labels
    else:
        order = np.argsort(ids, kind="stable")
        values = [ids[i] for i in order]
        rank = np.empty(len(ids), dtype=np.int64)
        rank[order] = np.arange(len(ids))
        code = rank[labels]
        names = [names[i] for i in order]
        integral = min(values) >= 1 and max(values) <= _INT32_MAX
    regions = _Regions(values, code.astype(np.int64), names, integral)

    def set_leaves(key):
        block = mb[key] if mb is not None and key in mb.keys() else None
        return [] if not isinstance(block, pv.MultiBlock) else _leaves(block, [])

    ignored = [
        key for key in ("Face Blocks", "Edge Blocks", "Element Sets", "Face Sets",
                        "Edge Sets")
        if set_leaves(key)
    ]
    if ignored:
        notes.append(
            f"ignored the {', '.join(map(repr, ignored))} of the VTK reader "
            "output (only element blocks, side sets and node sets are written)"
        )
    sets = _VtkSets(set_leaves("Side Sets"), set_leaves("Node Sets"), node_ids)
    return grid, regions, sets


def _prepare_volume(volume, region_key, notes):
    """Return ``(grid, regions, vtk_sets)``; the input is never modified.

    ``vtk_sets`` holds the side / node sets of VTK Exodus-reader output (None
    for any other input).
    """
    if isinstance(volume, vtkDataObject) and not isinstance(volume, pv.DataObject):
        volume = pv.wrap(volume)
    if isinstance(volume, pv.MultiBlock):
        if _is_exodus_reader_output(volume):
            return _from_exodus_reader(volume["Element Blocks"], volume, notes)
        leaves = _leaves(volume, [])
        if not leaves:
            raise ValueError("cannot write an empty mesh: the MultiBlock has no cells")
        if all("ElementBlockIds" in leaf.field_data.keys() for _, leaf in leaves):
            # the "Element Blocks" of VTK's Exodus reader output
            return _from_exodus_reader(volume, None, notes)
        names = _unique_strings([name for name, _ in leaves])
        grid = _combine_leaves([leaf for _, leaf in leaves], region_key, notes)
        labels = np.repeat(
            np.arange(len(leaves), dtype=np.int64), [leaf.n_cells for _, leaf in leaves]
        )
        regions = _Regions(
            values=list(range(1, len(leaves) + 1)),
            code=labels,
            defaults=names,
            integral=True,
        )
        return grid, regions, None

    grid = _as_unstructured(volume, "volume")
    if grid.n_cells == 0 or grid.n_points == 0:
        raise ValueError(
            f"cannot write an empty mesh ({grid.n_points} points, "
            f"{grid.n_cells} cells)"
        )
    return grid, _regions_of(grid, region_key, "volume"), None


def _use_vtk_sets(sets: _VtkSets, surface, node_sets, side_sets, nsets, cells,
                  layout, side_set_names, tolerance, notes):
    """Side sets and node sets from VTK Exodus-reader output.

    Sets the caller passed explicitly (``surface``, ``node_sets``) win.
    Returns ``(side_sets, nsets)``.
    """
    if sets.side_leaves and surface is not None:
        notes.append(
            "ignored the 'Side Sets' of the VTK reader output: surface= was given"
        )
    elif sets.side_leaves:
        ids = _vtk_ids(sets.side_leaves, None)
        if ids is None:
            ids = list(range(1, len(sets.side_leaves) + 1))
        names = _vtk_names(sets.side_leaves)
        pieces = []
        for set_id, (_, leaf) in zip(ids, sets.side_leaves):
            piece = _as_unstructured(leaf, "surface").copy(deep=False)
            piece.clear_data()
            piece.cell_data[_SET_KEY] = np.full(piece.n_cells, set_id, np.int64)
            pieces.append(piece)
        surf = pv.MultiBlock(pieces).combine(merge_points=False)
        if side_set_names is None:
            side_set_names = dict(zip(ids, names))
        try:
            side_sets = _build_side_sets(
                surf, cells, layout, _SET_KEY, side_set_names, tolerance, notes
            )
        except ValueError as err:
            notes.append(f"could not convert the VTK reader's side sets: {err}")
            side_sets = []
    if sets.node_leaves and node_sets is not None:
        notes.append(
            "ignored the 'Node Sets' of the VTK reader output: node_sets= was given"
        )
    elif sets.node_leaves:
        ids = _vtk_ids(sets.node_leaves, None)
        if ids is None:
            ids = list(range(1, len(sets.node_leaves) + 1))
        names = [
            f"node_set_{i}" if name is None else name
            for i, name in zip(ids, _vtk_names(sets.node_leaves))
        ]
        nsets, tree = [], None
        for set_id, name, (_, leaf) in zip(ids, names, sets.node_leaves):
            gids = leaf.point_data.get("GlobalNodeId")
            if gids is not None and sets.node_ids is not None:
                gids = np.asarray(gids).reshape(-1)
                order = np.argsort(sets.node_ids)
                pos = np.searchsorted(sets.node_ids[order], gids)
                pos = np.minimum(pos, order.size - 1)
                found = sets.node_ids[order][pos] == gids
                index = order[pos]
            else:
                if tree is None:
                    tree = cKDTree(cells.points)
                dist, index = tree.query(np.asarray(leaf.points, dtype=np.float64))
                found = dist == 0
            if not found.all():
                notes.append(
                    f"skipped node set {name!r} of the VTK reader output: "
                    f"{int((~found).sum())} of its points are not mesh nodes"
                )
                continue
            nsets.append(NodeSet(set_id, name, np.unique(index)))
    return side_sets, nsets


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


def _match_points(query: np.ndarray, points: np.ndarray, tol: float, same: float):
    """Map ``query`` points to their nearest volume node within ``tol``.

    Returns ``(node, canon)``: the node of every query point and, when the
    volume has duplicate nodes (closer than ``same`` to a matched node, e.g.
    an unmerged mesh), an array mapping every volume node to the lowest index
    among its duplicates (``None`` otherwise). ``node`` is already canonical.
    Only the nearest node and its exact duplicates are used: other nodes
    within ``tol`` (thin elements, large tolerances) are distinct nodes.
    """
    tree = cKDTree(points, balanced_tree=False, compact_nodes=False)
    # cKDTree compares squared distances: keep the bound's square > 0 so that
    # tol=0 still finds exact matches (filtered with ``dist <= tol`` below).
    bound = max(tol * (1.0 + 1e-9), 1e-150)
    dist, nearest = tree.query(query, k=1, distance_upper_bound=bound)
    missing = ~(dist <= tol)
    if missing.any():
        far, _ = tree.query(query[missing])
        raise ValueError(
            f"{int(missing.sum())} surface point(s) are farther than "
            f"side_set_tolerance={tol:.6g} from every volume node (largest "
            f"distance {far.max():.6g}): the surface does not lie on the volume "
            "mesh. Pass a larger side_set_tolerance if it is only approximately "
            "on the mesh."
        )
    nearest = np.asarray(nearest, dtype=np.int64)
    matched = np.unique(nearest)
    n_points = points.shape[0]
    bound = max(same * (1.0 + 1e-9), 1e-150)
    k = min(4, n_points)
    while True:
        dist, idx = tree.query(points[matched], k=k, distance_upper_bound=bound)
        dist = dist.reshape(matched.size, k)
        idx = idx.reshape(matched.size, k)
        if k >= n_points or not np.isfinite(dist[:, -1]).any():
            break
        k = min(2 * k, n_points)
    found = dist <= same
    if not found[:, 1:].any():
        return nearest, None
    rep = np.where(found, idx, n_points).min(axis=1)
    canon = np.arange(n_points)
    rows, cols = np.nonzero(found)
    np.minimum.at(canon, idx[rows, cols], rep[rows])
    return canon[nearest], canon


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
    node_of[used], canon = _match_points(
        surf_points[used], cells.points, tol, _COINCIDENT_RTOL * diag
    )
    mark = np.zeros(cells.n_points, dtype=bool)
    mark[node_of[used]] = True

    keys_of = {}
    collapsed = []
    for k, (surf_cells, corners) in groups.items():
        keys = np.sort(node_of[corners], axis=1)
        repeated = np.any(keys[:, 1:] == keys[:, :-1], axis=1)
        if repeated.any():
            collapsed.append(surf_cells[repeated])
        keys_of[k] = keys
    if collapsed:
        bad = np.sort(np.concatenate(collapsed))
        raise ValueError(
            f"{bad.size} surface cell(s) have two or more corners on the same "
            f"volume node (first: surface cell {int(bad[0])}): the surface has "
            "degenerate cells, or side_set_tolerance "
            f"({tol:.6g}) is larger than the distance between distinct nodes"
        )

    candidates = _side_candidates(cells, layout, mark, canon)
    elem_of = np.full(surf.n_cells, -1, dtype=np.int64)
    side_of = np.zeros(surf.n_cells, dtype=np.int64)
    for k, (surf_cells, corners) in groups.items():
        keys = keys_of[k]
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
        hint = ""
        if any(info.n_nodes > info.n_corners for info in layout.infos):
            hint = (
                " The volume has quadratic cells: extract their faces unsplit "
                "with volume.extract_surface(nonlinear_subdivision=0) (the "
                "default subdivides each quadratic face into linear ones)."
            )
        raise ValueError(
            f"{unmatched.size} of {surf.n_cells} surface cell(s) do not match a "
            f"side of any element (first: surface cell {int(unmatched[0])}). "
            f"Side sets are matched to the {targets}; shell, bar and sphere "
            "elements are not side-set targets." + hint
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


def _numeric(values, what, word_size=8, notes=None) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind not in "biuf":
        raise TypeError(f"{what} must be numeric, got dtype {arr.dtype}")
    if notes is not None:
        _check_exact_ints(arr, what, word_size, notes)
    return arr.astype(np.float64, copy=False)


def _check_exact_ints(arr: np.ndarray, what: str, word_size: int, notes) -> None:
    """Note integer values that the float storage cannot hold exactly."""
    if arr.dtype.kind not in "iu" or arr.size == 0:
        return
    limit = _EXACT_INT[word_size]
    biggest = max(int(arr.max()), -int(arr.min()))
    if biggest > limit:
        precision = "single" if word_size == 4 else "double"
        notes.append(
            f"{what} holds integers up to {biggest}, beyond what {precision} "
            f"precision floats store exactly (2**{limit.bit_length() - 1}): "
            "nearby values become equal in the file"
        )


def _field(values, n_steps, n, name, kind, implicit_times, word_size=8,
           notes=None) -> np.ndarray:
    """Time-varying field as ``(n_steps, n, n_components)`` float64."""
    arr = _numeric(values, f"{kind} field {name!r}", word_size, notes)
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


def _static(values, name, what, notes, word_size=8):
    """A point / cell array as float64 ``(n, n_components)``, or None (skipped)."""
    arr = np.asarray(values)
    kind = arr.dtype.kind
    if kind == "c":
        notes.append(f"skipped {what} array {name!r}: complex values are not supported")
        return None
    if kind not in "biuf":
        notes.append(f"skipped {what} array {name!r}: non-numeric dtype {arr.dtype}")
        return None
    _check_exact_ints(arr, f"{what} array {name!r}", word_size, notes)
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


def _vtk_family(name: str) -> str:
    """Key shared by all variables VTK's reader may glue into one array.

    VTK glues variables named ``<base>X, <base>Y[, <base>Z]`` (any case) into
    a vector ``<base>``; tensors ``<b>XX, <b>XY, <b>XZ`` give ``<b>X`` etc.
    Removing up to two trailing x/y/z letters (and then one ``_``) maps every
    member of such a group to one key (a superset of VTK's groups).
    """
    key = name.upper()
    for _ in range(2):
        if key and key[-1] in "XYZ":
            key = key[:-1]
    return key[:-1] if key.endswith("_") else key


def _variable_families(names: list, sources: list) -> list:
    """Groups of element variables whose truth-table rows must agree.

    Joins the components of each source array (``sources``) and every set of
    variables VTK's reader would glue into one array: VTK cannot read a glued
    array whose components are stored on different blocks (it crashes).
    """
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in sources:
        for other in group[1:]:
            union(group[0], other)
    by_family: dict[str, str] = {}
    for name in names:
        key = _vtk_family(name)
        if key in by_family:
            union(by_family[key], name)
        else:
            by_family[key] = name
    families: dict[str, list] = {}
    for name in names:
        families.setdefault(find(name), []).append(name)
    return list(families.values())


def _drop_undefined_blocks(elem_vars: dict, families: list) -> None:
    """Leave out the blocks on which a variable family is NaN everywhere.

    A block is dropped from all variables of a family together (VTK's reader
    crashes on the components of one glued vector stored on different
    blocks), only if every variable of the family is NaN on it, and only if
    the family is defined on some block.
    """
    for names in families:
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


def _check_recombination(names: list, intended: dict, kind: str, notes) -> None:
    """Note multi-component arrays that will not read back as written.

    ``intended`` maps each split array's name to its scalar names; readers
    regroup the scalars by name (:func:`exovista._model.component_groups`),
    so another variable named like a component of it (``U_Z`` next to a
    2-component ``U``) changes the result.
    """
    groups = component_groups(names)
    wrong = [
        base for base, members in intended.items()
        if groups.get(base) != members
    ]
    if wrong:
        notes.append(
            f"{kind} array(s) {wrong} will not be recombined as written when the "
            "file is read: other variables are named like their components "
            "(e.g. a scalar 'U_Z' next to a 2-component 'U'); rename them"
        )


def _build_variables(grid, cells: _Cells, layout: _Layout, region_key, *, times,
                     node_fields, element_fields, global_fields, save_node_arrays,
                     save_cell_arrays, notes, float_word_size=8):
    node_fields = _check_mapping(node_fields, "node_fields")
    element_fields = _check_mapping(element_fields, "element_fields")
    global_fields = _check_mapping(global_fields, "global_fields")
    times_arr = _times(times)
    implicit = times_arr is None
    n_steps = 1 if implicit else times_arr.size
    n_points, n_cells = cells.n_points, cells.n_cells
    ws = float_word_size

    # scalar variable names of each multi-component source array
    node_split: dict = {}
    elem_split: dict = {}

    def split(target, name, parts):
        if len(parts) > 1:
            target[name.rstrip()] = [s.rstrip() for s, _ in parts]

    # time-varying fields (their names win over static arrays)
    node_tv: dict = {}
    for name, values in node_fields.items():
        arr = _field(values, n_steps, n_points, name, "node", implicit, ws, notes)
        parts = _components(name, arr)
        for scalar, col in parts:
            _add(node_tv, scalar, col, f"node field {name!r}")
        split(node_split, name, parts)
    elem_tv: dict = {}
    elem_groups: list = []  # scalar variable names of each source array
    for name, values in element_fields.items():
        arr = _field(values, n_steps, n_cells, name, "element", implicit, ws, notes)
        parts = _components(name, arr)
        for scalar, col in parts:
            _add(elem_tv, scalar, col, f"element field {name!r}")
        elem_groups.append([s.rstrip() for s, _ in parts])
        split(elem_split, name, parts)
    global_vars: dict = {}
    for name, values in global_fields.items():
        arr = _numeric(values, f"global field {name!r}", ws, notes)
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
            arr = _static(grid.point_data[name], name, "point", notes, ws)
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
            split(node_split, name, parts)
    elem_static: dict = {}
    if save_cell_arrays:
        for name in grid.cell_data.keys():
            if (name in _EXCLUDED_ARRAYS or name == region_key
                    or name in element_fields):
                continue
            arr = _static(grid.cell_data[name], name, "cell", notes, ws)
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
            split(elem_split, name, parts)

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
    _drop_undefined_blocks(
        elem_vars, _variable_families(list(elem_vars), elem_groups)
    )
    _check_recombination(list(node_vars), node_split, "point", notes)
    _check_recombination(list(elem_vars), elem_split, "cell", notes)

    has_vars = bool(node_vars or elem_vars or global_vars)
    if times_arr is None:
        times_arr = np.zeros(1 if has_vars else 0)
    return times_arr, node_vars, elem_vars, global_vars
