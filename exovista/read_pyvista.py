"""Read ExodusII databases into PyVista objects.

This is the read counterpart of :mod:`exovista.write_pyvista`, built on
:func:`exovista._backend.open_reader` (serial files and parallel
decompositions) and the element tables of :mod:`exovista._celltypes`.

* :func:`read_exo` -- the mesh as one :class:`pyvista.UnstructuredGrid`
  (float64 points in file node order; all supported element types including
  quadratic cells, shells, bars and spheres) with node, element and global
  variables at one time step, and optionally the side sets.
* :func:`read_side_sets` / :func:`read_node_sets` -- the sets alone.
* :func:`read_node_fields` / :func:`read_element_fields` /
  :func:`read_global_fields` -- whole time histories without building a mesh.

Element order
-------------
Exodus stores elements grouped by block. :func:`exovista.write_exo` records
each element's original cell index in the element id map
(``elem_num_map``). With ``original_order=True`` (the default), whenever the
map is a permutation of ``1..num_elem``, cells (and element variables) are
returned so that cell ``i`` has element id ``i + 1``, i.e. in the order of the
mesh that was written. Otherwise the file order (blocks concatenated) is kept.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Iterable

import numpy as np
import pyvista as pv
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkStringArray

from ._backend import ExodusReader, open_reader
from ._celltypes import ExoType, resolve_exo_type
from ._model import component_suffixes

__all__ = [
    "read_exo",
    "read_side_sets",
    "read_node_sets",
    "read_node_fields",
    "read_element_fields",
    "read_global_fields",
]

# Keys read_exo always fills with their documented meaning.
_RESERVED_CELL_ARRAYS = ("exo_block_id", "exo_block_name", "exo_elem_id")
_RESERVED_FIELD_DATA = ("times", "time", "num_dim")

# netCDF's default fill value for floats: a time value that was never written.
_NETCDF_FLOAT_FILL = 9.969209968386869e36

# Letter suffixes of multi-component variables (see _model.component_suffixes).
_LETTER_SUFFIXES = frozenset(
    s[1:] for n in (2, 3, 6, 9) for s in component_suffixes(n)
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def read_exo(
    filename,
    *files,
    time_step=-1,
    read_node_variables=True,
    read_element_variables=True,
    read_global_variables=True,
    read_side_sets=False,
    combine_components=True,
    original_order=True,
):
    """Read an ExodusII file into a :class:`pyvista.UnstructuredGrid`.

    Parameters
    ----------
    filename : str or os.PathLike
        The Exodus file, or the first part of a parallel decomposition. Glob
        patterns (``"run.exo.4.*"``) are expanded.
    *files : str or os.PathLike
        Remaining parts of a parallel decomposition. The parts are joined into
        one mesh with global node and element numbering.
    time_step : int, default: -1
        **1-based** time step at which node, element and global variables are
        sampled; negative values count from the end (``-1`` is the last
        step). ``0`` and steps outside ``1..n_steps`` raise ``IndexError``.
        Ignored (other than ``0``) when the file has no time steps.
    read_node_variables, read_element_variables : bool, str or list of str
        Which node (point data) / element (cell data) variables to read:
        ``True`` (default) all, ``False`` none, or a name / list of names.
        Names may be combined names (``"velocity"``) or the scalar names
        stored in the file (``"velocity_X"``). Unknown names raise
        ``KeyError``.
    read_global_variables : bool, str or list of str
        Which global variables to store in ``grid.field_data`` (one-element
        arrays holding the value at ``time_step``). Same semantics.
    read_side_sets : bool, default: False
        Also return the side sets (see :func:`read_side_sets`).
    combine_components : bool, default: True
        Recombine scalar variables named with exovista's component suffixes
        (``_X, _Y[, _Z]``; symmetric tensors ``_XX, _YY, _ZZ, _XY, _YZ,
        _XZ``; full tensors ``_XX, _XY, ... _ZZ``; otherwise ``_0 .. _{n-1}``)
        into one multi-component array, but only when all components of the
        array are present and the combined name is not itself a variable.
    original_order : bool, default: True
        If the element id map is a permutation of ``1..num_elem`` (as written
        by :func:`exovista.write_exo`), return the cells sorted by element id,
        restoring the cell order of the mesh that was written. Otherwise, and
        with ``False``, cells are in file order (blocks concatenated).

    Returns
    -------
    pyvista.UnstructuredGrid
        Points are float64 in file node order (never downcast; a 2D file
        gets ``z = 0``). Cell data:

        * ``exo_block_id`` -- Exodus element block id of each cell;
        * ``exo_block_name`` -- block name (``""`` if unnamed);
        * ``exo_elem_id`` -- element id from the element id map (``1..N`` in
          file order if the file has none);
        * the element variables (NaN on blocks a variable is not defined on).

        Point data holds the node variables. Field data holds ``"times"``
        (all time values), ``"time"`` (the selected time; absent when the
        file has no time steps), ``"num_dim"`` and the global variables.
    pyvista.MultiBlock
        Only with ``read_side_sets=True``: the result of
        :func:`read_side_sets` (with the same ``original_order``). The side
        sets share the grid's points.

    Warns
    -----
    UserWarning
        For element blocks of an unsupported element type (they are skipped),
        side-set entries that cannot be resolved, and variables whose names
        clash with the reserved arrays above (they are skipped; use
        :func:`read_element_fields` / :func:`read_global_fields`).

    Notes
    -----
    Files written by exovista 1.x are read as they were meant: their
    triangles (``TRI3`` blocks in a 3D file) keep edge side numbering, and
    their unwritten time value of a single static step reads as ``0.0``.
    """
    with open_reader(filename, *files) as r:
        layout = _Layout(r, original_order)
        times = _read_times(r, layout.legacy)
        step = _resolve_step(time_step, len(times))

        points = _points(r)
        vtk_pts = pv.vtk_points(points, deep=False, force_float=False)
        grid = _build_grid(layout, vtk_pts)

        if step is not None:
            node_plan = _select(
                _plan(r.node_var_names(), combine_components),
                read_node_variables,
                "node",
            )
            for out_name, components in node_plan:
                values = [r.node_var(name, step) for name in components]
                grid.point_data[out_name] = _stack(values, components, out_name)

            elem_plan = _select(
                _plan(r.elem_var_names(), combine_components),
                read_element_variables,
                "element",
            )
            elem_plan = _drop_reserved(elem_plan, _RESERVED_CELL_ARRAYS, "element")
            for out_name, components in elem_plan:
                values = [layout.cell_values(r, name, step) for name in components]
                grid.cell_data[out_name] = _stack(values, components, out_name)

            glob_plan = _select(
                _plan(r.global_var_names(), False), read_global_variables, "global"
            )
            glob_plan = _drop_reserved(glob_plan, _RESERVED_FIELD_DATA, "global")
            for out_name, (name,) in glob_plan:
                grid.field_data[out_name] = np.array([r.global_var(name, step)])

        grid.field_data["times"] = times
        if step is not None:
            grid.field_data["time"] = times[step : step + 1].copy()
        grid.field_data["num_dim"] = np.array([r.num_dim])

        if read_side_sets:
            return grid, _side_sets(r, layout, vtk_pts)
        return grid


def read_side_sets(filename, *files, original_order=True):
    """Read the side sets of an ExodusII file.

    Parameters
    ----------
    filename, *files : str or os.PathLike
        As for :func:`read_exo`.
    original_order : bool, default: True
        Must match the ``original_order`` the grid was read with: it decides
        the ``cell_id`` values (see below).

    Returns
    -------
    pyvista.MultiBlock
        One :class:`pyvista.PolyData` per side set, in file order, keyed by the
        set name; unnamed sets and repeated names get ``"side_set_<id>"``
        (keys are unique). Every set holds *all* points of the mesh (the
        points of :func:`read_exo`, shared between the sets without copying;
        copy a set before modifying its points in place), so point ids index
        the grid. Faces of 3D elements are polygons, edges of 2D elements are
        lines (and ends of bars are vertices); for quadratic elements only the
        corner nodes are used, ordered as by the Exodus library. When a set
        mixes these kinds, PolyData stores vertices first, then lines, then
        polygons; the cell data stays aligned with the cells. Cell data:

        * ``exo_elem_id`` -- 1-based element index in *file order*, as stored
          in the file's side set (not the element id map);
        * ``exo_side`` -- 1-based Exodus side number;
        * ``cell_id`` -- 0-based index of the element's cell in the grid
          returned by :func:`read_exo` with the same ``original_order``;
        * ``orig_elem_id`` and ``face_id`` -- deprecated aliases of
          ``exo_elem_id`` and ``exo_side`` (exovista 1.x names).

        Field data ``exo_side_set_id`` holds the Exodus side-set id.

    Warns
    -----
    UserWarning
        Entries on elements of skipped (unsupported) blocks or with invalid
        side numbers are dropped with a warning.
    """
    with open_reader(filename, *files) as r:
        layout = _Layout(r, original_order)
        vtk_pts = pv.vtk_points(_points(r), deep=False, force_float=False)
        return _side_sets(r, layout, vtk_pts)


def read_node_sets(filename, *files):
    """Read the node sets of an ExodusII file.

    Parameters
    ----------
    filename, *files : str or os.PathLike
        As for :func:`read_exo`.

    Returns
    -------
    dict of str to numpy.ndarray
        Node set name -> 0-based node indices (int64) into the points of
        :func:`read_exo`, in file order. Unnamed sets and repeated names get
        ``"node_set_<id>"`` (keys are unique).
    """
    with open_reader(filename, *files) as r:
        sets = r.node_sets()
    keys = _unique_keys([(s.id, s.name) for s in sets], "node_set")
    return {key: s.nodes for key, s in zip(keys, sets)}


def read_node_fields(filename, *files, names=None, combine_components=False):
    """Read node-variable time histories without building the mesh.

    Inverse of the ``node_fields`` argument of :func:`exovista.write_exo`.

    Parameters
    ----------
    filename, *files : str or os.PathLike
        As for :func:`read_exo`.
    names : str or list of str, optional
        Variables to read (default: all). Unknown names raise ``KeyError``.
    combine_components : bool, default: False
        Recombine component variables as in :func:`read_exo`.

    Returns
    -------
    dict of str to numpy.ndarray
        ``(n_steps, n_nodes)`` float64 arrays, or ``(n_steps, n_nodes,
        n_components)`` for combined arrays. Columns follow the points of
        :func:`read_exo`.
    """
    with open_reader(filename, *files) as r:
        plan = _select(
            _plan(r.node_var_names(), combine_components), _names(names), "node"
        )
        shape = (r.num_steps, r.num_nodes)
        out = {}
        for out_name, components in plan:
            values = [_history(r.node_var(name), shape) for name in components]
            out[out_name] = _stack(values, components, out_name)
        return out


def read_element_fields(
    filename, *files, names=None, combine_components=False, original_order=True
):
    """Read element-variable time histories without building the mesh.

    Inverse of the ``element_fields`` argument of :func:`exovista.write_exo`.

    Parameters
    ----------
    filename, *files : str or os.PathLike
        As for :func:`read_exo`.
    names : str or list of str, optional
        Variables to read (default: all). Unknown names raise ``KeyError``.
    combine_components : bool, default: False
        Recombine component variables as in :func:`read_exo`.
    original_order : bool, default: True
        As for :func:`read_exo`: columns are in the cell order of the grid
        :func:`read_exo` returns with the same ``original_order`` (for files
        written by :func:`exovista.write_exo`, the original cell order).

    Returns
    -------
    dict of str to numpy.ndarray
        ``(n_steps, n_cells)`` float64 arrays, or ``(n_steps, n_cells,
        n_components)`` for combined arrays; NaN on blocks a variable is not
        defined on. Elements of unsupported blocks (skipped by
        :func:`read_exo`, with a warning) are skipped here too.
    """
    with open_reader(filename, *files) as r:
        layout = _Layout(r, original_order)
        plan = _select(
            _plan(r.elem_var_names(), combine_components), _names(names), "element"
        )
        out = {}
        for out_name, components in plan:
            values = [layout.cell_values(r, name, None) for name in components]
            out[out_name] = _stack(values, components, out_name)
        return out


def read_global_fields(filename, *files, names=None):
    """Read global-variable time histories.

    Parameters
    ----------
    filename, *files : str or os.PathLike
        As for :func:`read_exo`.
    names : str or list of str, optional
        Variables to read (default: all). Unknown names raise ``KeyError``.

    Returns
    -------
    dict of str to numpy.ndarray
        ``(n_steps,)`` float64 arrays.
    """
    with open_reader(filename, *files) as r:
        plan = _select(_plan(r.global_var_names(), False), _names(names), "global")
        n_steps = r.num_steps
        return {
            out_name: _history(r.global_var(name), (n_steps,))
            for out_name, (name,) in plan
        }


# --------------------------------------------------------------------------- #
# Element blocks and cell order
# --------------------------------------------------------------------------- #
def _is_legacy_1x(r: ExodusReader) -> bool:
    """True for a file written by exovista 1.x (always float32, fixed title)."""
    return r.title == "pyvista_mesh" and r.float_word_size == 4


class _Layout:
    """The element blocks of a file and the order of the cells read_exo builds.

    Cells are the elements of the supported, non-empty blocks. "Kept-file
    order" is the file order restricted to those elements; :attr:`perm`
    reorders it into the output order (None: identity).
    """

    def __init__(self, r: ExodusReader, original_order: bool):
        self.num_dim = r.num_dim
        self.legacy = _is_legacy_1x(r)
        self.blocks = r.blocks()
        sizes = np.array([b.n_elem for b in self.blocks], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.num_elem = int(self.offsets[-1])

        # resolved cell type, and the type whose side table side sets use
        self.types: list[ExoType | None] = []
        self.side_types: list[ExoType | None] = []
        skipped = []
        for block in self.blocks:
            info = None
            if block.n_elem:
                info = resolve_exo_type(
                    block.elem_type, block.n_nodes_per_elem, self.num_dim
                )
                if info is None:
                    skipped.append(block)
            self.types.append(info)
            self.side_types.append(self._side_type(block, info))
        if skipped:
            details = ", ".join(
                f"{b.id} ({b.elem_type!r}, {b.n_nodes_per_elem} nodes, "
                f"{b.n_elem} elements)"
                for b in skipped
            )
            warnings.warn(
                f"Skipping element block(s) of unsupported element type: "
                f"{details}",
                UserWarning,
                stacklevel=3,
            )

        self.kept = [k for k, info in enumerate(self.types) if info is not None]
        self.elem_ids = self._element_ids(r)
        if len(self.kept) == sum(1 for b in self.blocks if b.n_elem):
            kept_file = np.arange(self.num_elem, dtype=np.int64)
        else:
            kept_file = np.concatenate(
                [np.zeros(0, np.int64)]
                + [
                    np.arange(self.offsets[k], self.offsets[k + 1], dtype=np.int64)
                    for k in self.kept
                ]
            )
        self.perm = None
        if original_order and _is_permutation(self.elem_ids):
            ids = self.elem_ids[kept_file]
            if ids.size > 1 and not np.all(ids[1:] > ids[:-1]):
                self.perm = np.argsort(ids, kind="stable")
        #: file index (0-based) of each output cell
        self.cell_file_index = kept_file if self.perm is None else kept_file[
            self.perm
        ]
        self._cell_of_file = None

    def _side_type(self, block, info):
        if info is None:
            return None
        if (
            self.legacy
            and info.family == "TRISHELL"
            and not block.elem_type.upper().startswith("TRISHELL")
        ):
            # exovista 1.x wrote 2D triangle meshes as TRI3 in 3D files but
            # numbered their sides as triangle edges.
            return resolve_exo_type(block.elem_type, block.n_nodes_per_elem, 2)
        return info

    def _element_ids(self, r: ExodusReader) -> np.ndarray:
        ids = r.elem_id_map()
        if ids is not None and ids.shape != (self.num_elem,):
            warnings.warn(
                f"Ignoring the element id map: it has {ids.size} entries for "
                f"{self.num_elem} elements",
                UserWarning,
                stacklevel=4,
            )
            ids = None
        if ids is None:
            return np.arange(1, self.num_elem + 1, dtype=np.int64)
        return np.asarray(ids, dtype=np.int64)

    @property
    def n_cells(self) -> int:
        return int(self.cell_file_index.size)

    @property
    def cell_of_file(self) -> np.ndarray:
        """Output cell index of each file element (-1 for skipped elements)."""
        if self._cell_of_file is None:
            out = np.full(self.num_elem, -1, dtype=np.int64)
            out[self.cell_file_index] = np.arange(self.n_cells, dtype=np.int64)
            self._cell_of_file = out
        return self._cell_of_file

    def _ordered(self, values: np.ndarray, axis: int = 0) -> np.ndarray:
        """Reorder an array over kept-file-ordered cells to output order."""
        if self.perm is None:
            return values
        return np.take(values, self.perm, axis=axis)

    def per_block(self, values) -> np.ndarray:
        """Per-cell array (output order) from one value per kept block."""
        counts = [self.blocks[k].n_elem for k in self.kept]
        return self._ordered(np.repeat(np.asarray(values), counts))

    def cell_values(self, r: ExodusReader, name: str, step) -> np.ndarray:
        """Element variable over all cells (output order); NaN if undefined.

        ``(n_cells,)`` at a 0-based ``step``, ``(n_steps, n_cells)`` for None.
        """
        n_steps = r.num_steps
        chunks = []
        for k in self.kept:
            block = self.blocks[k]
            values = r.elem_var(name, block.id, step)
            shape = (block.n_elem,) if step is not None else (n_steps, block.n_elem)
            if values is None:
                chunks.append(np.full(shape, np.nan))
            else:
                chunks.append(_history(values, shape))
        if not chunks:
            shape = (0,) if step is not None else (n_steps, 0)
            return np.zeros(shape)
        values = np.concatenate(chunks, axis=-1)
        return self._ordered(values, axis=values.ndim - 1)

    def cells(self):
        """VTK ``(offsets, connectivity, celltypes)`` of the output cells."""
        conns = []
        for k in self.kept:
            info = self.types[k]
            conn = self.blocks[k].conn
            perm = info.exo_to_vtk
            if tuple(perm) != tuple(range(len(perm))):
                conn = conn[:, list(perm)]
            conns.append(conn)
        widths = {c.shape[1] for c in conns}
        celltypes = self.per_block(
            np.array([self.types[k].vtk_type for k in self.kept], dtype=np.uint8)
        ).astype(np.uint8, copy=False)
        n = self.n_cells
        if not conns:
            return np.zeros(1, np.int64), np.zeros(0, np.int64), celltypes
        if len(widths) == 1:
            width = widths.pop()
            conn = conns[0] if len(conns) == 1 else np.concatenate(conns)
            conn = self._ordered(conn)
            offsets = np.arange(0, (n + 1) * width, width, dtype=np.int64)
            return offsets, np.ascontiguousarray(conn, np.int64).reshape(-1), celltypes
        # cells of different sizes: gather the output order from the file order
        flat = np.concatenate([c.reshape(-1) for c in conns])
        sizes = np.concatenate(
            [np.full(c.shape[0], c.shape[1], dtype=np.int64) for c in conns]
        )
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        if self.perm is not None:
            sizes = sizes[self.perm]
            starts = starts[self.perm]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        if self.perm is not None:
            flat = flat[np.repeat(starts - offsets[:-1], sizes) + np.arange(flat.size)]
        return offsets, np.ascontiguousarray(flat, np.int64), celltypes


def _is_permutation(ids: np.ndarray) -> bool:
    """True if ``ids`` is a permutation of ``1..len(ids)``."""
    n = ids.size
    if n == 0 or ids.min() != 1 or ids.max() != n:
        return False
    return bool(np.all(np.bincount(ids - 1, minlength=n) == 1))


def _points(r: ExodusReader) -> np.ndarray:
    """Float64 ``(n_nodes, 3)`` points; lower dimensions are padded with 0."""
    coords = np.asarray(r.coords(), dtype=np.float64)
    if coords.ndim == 2 and coords.shape[1] == 3:
        return np.ascontiguousarray(coords)
    coords = coords.reshape(coords.shape[0], -1)
    points = np.zeros((coords.shape[0], 3), dtype=np.float64)
    points[:, : coords.shape[1]] = coords[:, :3]
    return points


def _build_grid(layout: _Layout, vtk_pts) -> pv.UnstructuredGrid:
    grid = pv.UnstructuredGrid()
    grid.SetPoints(vtk_pts)
    offsets, connectivity, celltypes = layout.cells()
    if layout.n_cells:
        cell_array = pv.CellArray.from_arrays(offsets, connectivity, deep=False)
        grid.SetCells(numpy_to_vtk(celltypes, deep=True), cell_array)

    blocks = [layout.blocks[k] for k in layout.kept]
    grid.cell_data["exo_block_id"] = layout.per_block(
        np.array([b.id for b in blocks], dtype=np.int64)
    ).astype(np.int64, copy=False)
    block_index = layout.per_block(np.arange(len(blocks), dtype=np.int64))
    grid.GetCellData().AddArray(
        _string_array("exo_block_name", [b.name for b in blocks], block_index)
    )
    grid.cell_data["exo_elem_id"] = layout.elem_ids[layout.cell_file_index]
    return grid


def _string_array(name: str, values: list[str], index: np.ndarray) -> vtkStringArray:
    """A vtkStringArray holding ``values[index[i]]`` at ``i``.

    Built with one VTK call per run of equal values (copied from a template
    filled by doubling), not one per entry: far faster for per-cell block
    names, which come in long runs. Unlike pyvista's string conversion this
    also accepts non-ASCII names.
    """
    n = int(index.size)
    out = vtkStringArray()
    out.SetName(name)
    out.SetNumberOfValues(n)
    if n == 0:
        return out
    starts = np.flatnonzero(np.concatenate([[True], index[1:] != index[:-1]]))
    lengths = np.diff(np.concatenate([starts, [n]]))
    run_values = index[starts]
    longest = np.zeros(len(values), dtype=np.int64)
    np.maximum.at(longest, run_values, lengths)
    templates = {}
    for value in np.flatnonzero(longest).tolist():
        template = vtkStringArray()
        size = int(longest[value])
        template.SetNumberOfValues(size)
        template.SetValue(0, str(values[value]))
        filled = 1
        while filled < size:
            count = min(filled, size - filled)
            template.InsertTuples(filled, count, 0, template)
            filled += count
        templates[value] = template
    for start, length, value in zip(
        starts.tolist(), lengths.tolist(), run_values.tolist()
    ):
        out.InsertTuples(start, length, 0, templates[value])
    return out


# --------------------------------------------------------------------------- #
# Sets
# --------------------------------------------------------------------------- #
def _unique_keys(sets: Iterable[tuple[int, str]], prefix: str) -> list[str]:
    """Unique keys for ``(id, name)`` sets: name, else ``f"{prefix}_{id}"``."""
    keys: list[str] = []
    used: set[str] = set()
    for set_id, name in sets:
        key = name
        if not key or key in used:
            key = default = f"{prefix}_{set_id}"
            k = 1
            while key in used:
                key = f"{default}_{k}"
                k += 1
        used.add(key)
        keys.append(key)
    return keys


def _side_sets(r: ExodusReader, layout: _Layout, vtk_pts) -> pv.MultiBlock:
    sets = r.side_sets()
    keys = _unique_keys([(s.id, s.name) for s in sets], "side_set")
    out = pv.MultiBlock()
    for key, ss in zip(keys, sets):
        out[key] = _side_set_mesh(ss, layout, vtk_pts, key)
    return out


def _side_set_mesh(ss, layout: _Layout, vtk_pts, key: str) -> pv.PolyData:
    elems = ss.elems
    sides = ss.sides
    n = elems.size
    block_of = np.searchsorted(layout.offsets, elems, side="right") - 1
    in_range = (elems >= 0) & (elems < layout.num_elem)
    sizes = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    problems = []
    dropped = 0

    # Group entries by (block, side): one vectorized gather per group.
    groups = []
    for k in np.unique(block_of[in_range]).tolist():
        in_block = np.flatnonzero(in_range & (block_of == k))
        info = layout.side_types[k]
        block = layout.blocks[k]
        if info is None or not info.sides:
            dropped += in_block.size
            problems.append(
                f"{in_block.size} on block {block.id} (element type "
                f"{block.elem_type!r} has no supported sides)"
            )
            continue
        local = elems[in_block] - layout.offsets[k]
        block_sides = sides[in_block]
        for side in np.unique(block_sides).tolist():
            sel = block_sides == side
            if not 1 <= side <= info.n_sides:
                dropped += int(sel.sum())
                problems.append(
                    f"{int(sel.sum())} with side {side} on block {block.id} "
                    f"({info.name} has {info.n_sides} sides)"
                )
                continue
            entries = in_block[sel]
            nodes = block.conn[np.ix_(local[sel], info.sides[side - 1])]
            sizes[entries] = nodes.shape[1]
            valid[entries] = True
            groups.append((entries, nodes))
    if np.any(~in_range):
        dropped += int((~in_range).sum())
        problems.append(
            f"{int((~in_range).sum())} on elements outside 1..{layout.num_elem}"
        )
    if problems:
        warnings.warn(
            f"Side set {ss.id} ({key!r}): dropping {dropped} of {n} entries: "
            + "; ".join(problems),
            UserWarning,
            stacklevel=4,
        )

    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    flat = np.zeros(offsets[-1], dtype=np.int64)
    for entries, nodes in groups:
        width = nodes.shape[1]
        flat[offsets[entries][:, None] + np.arange(width)] = nodes

    # PolyData keeps vertices, then lines, then polygons; order entries so.
    kind = np.minimum(sizes, 3)
    keep = np.flatnonzero(valid)
    keep = keep[np.argsort(kind[keep], kind="stable")]

    mesh = pv.PolyData()
    mesh.SetPoints(vtk_pts)
    setters = {1: mesh.SetVerts, 2: mesh.SetLines, 3: mesh.SetPolys}
    for k, setter in setters.items():
        entries = keep[kind[keep] == k]
        if not entries.size:
            continue
        cell_sizes = sizes[entries]
        cell_offsets = np.concatenate([[0], np.cumsum(cell_sizes)]).astype(np.int64)
        starts = offsets[entries]
        conn = flat[
            np.repeat(starts - cell_offsets[:-1], cell_sizes)
            + np.arange(cell_offsets[-1])
        ]
        setter(pv.CellArray.from_arrays(cell_offsets, conn, deep=False))

    elem_ids = elems[keep] + 1
    side_ids = sides[keep].copy()
    mesh.cell_data["exo_elem_id"] = elem_ids
    mesh.cell_data["exo_side"] = side_ids
    mesh.cell_data["cell_id"] = layout.cell_of_file[elems[keep]]
    # deprecated exovista 1.x names
    mesh.cell_data["orig_elem_id"] = elem_ids.copy()
    mesh.cell_data["face_id"] = side_ids.copy()
    mesh.field_data["exo_side_set_id"] = np.array([ss.id])
    return mesh


# --------------------------------------------------------------------------- #
# Variables
# --------------------------------------------------------------------------- #
def _read_times(r: ExodusReader, legacy: bool) -> np.ndarray:
    times = np.asarray(r.times(), dtype=np.float64)
    unwritten = times == _NETCDF_FLOAT_FILL
    if np.any(unwritten):
        # exovista 1.x left the time of its single static step unwritten
        times = times.copy()
        times[unwritten] = 0.0 if legacy else np.nan
    return times


def _resolve_step(time_step, n_steps: int):
    """1-based / negative ``time_step`` -> 0-based index (None: no steps)."""
    if isinstance(time_step, (bool, np.bool_)):
        raise TypeError("time_step must be an integer, not a bool")
    try:
        ts = operator.index(time_step)
    except TypeError:
        raise TypeError(
            f"time_step must be an integer, got {type(time_step).__name__}"
        ) from None
    if ts == 0:
        raise IndexError(
            "time_step is 1-based: use 1 for the first step and -1 for the last"
        )
    if n_steps == 0:
        return None
    index = ts - 1 if ts > 0 else n_steps + ts
    if not 0 <= index < n_steps:
        raise IndexError(
            f"time_step {ts} is out of range: the file has {n_steps} time "
            f"step(s) (valid: 1..{n_steps} or -{n_steps}..-1)"
        )
    return index


def _history(values, shape) -> np.ndarray:
    """``values`` as a float64 array of ``shape`` (empty histories allowed)."""
    if values is None:
        return np.zeros(shape)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 and int(np.prod(shape)) == 0:
        return np.zeros(shape)
    return values.reshape(shape)


def _component_groups(names: list[str]) -> dict[str, list[str]]:
    """Base name -> component variable names (exovista's suffix convention).

    A group is returned only if its suffixes are exactly
    ``component_suffixes(n)`` for its size ``n >= 2`` and the base name is not
    itself a variable. Components are ordered by that convention.
    """
    candidates: dict[str, dict[str, str]] = {}
    for name in names:
        base, sep, suffix = name.rpartition("_")
        if not base or not sep:
            continue
        if suffix not in _LETTER_SUFFIXES and not (
            suffix.isascii() and suffix.isdigit() and str(int(suffix)) == suffix
        ):
            continue
        candidates.setdefault(base, {})["_" + suffix] = name
    existing = set(names)
    groups = {}
    for base, by_suffix in candidates.items():
        n = len(by_suffix)
        if n < 2 or base in existing:
            continue
        expected = component_suffixes(n)
        if set(expected) == set(by_suffix):
            groups[base] = [by_suffix[s] for s in expected]
    return groups


def _plan(names: list[str], combine: bool) -> list[tuple[str, tuple[str, ...]]]:
    """``(output name, component names)`` per output array, in file order."""
    if not combine:
        return [(name, (name,)) for name in names]
    groups = _component_groups(names)
    position = {name: i for i, name in enumerate(names)}
    first = {}
    member = set()
    for base, components in groups.items():
        member.update(components)
        first[min(components, key=position.__getitem__)] = base
    plan = []
    for name in names:
        if name in first:
            base = first[name]
            plan.append((base, tuple(groups[base])))
        elif name not in member:
            plan.append((name, (name,)))
    return plan


def _names(names):
    """``names=None`` of the read_*_fields functions means all variables."""
    return True if names is None else names


def _select(plan, select, kind: str):
    """Entries of ``plan`` chosen by ``select`` (True/False/None/str/list)."""
    if select is True:
        return list(plan)
    if select is False or select is None:
        return []
    requested = [select] if isinstance(select, str) else list(select)
    by_name = {out_name: (out_name, comps) for out_name, comps in plan}
    scalars = {c for _, comps in plan for c in comps}
    chosen, missing, seen = [], [], set()
    for name in requested:
        name = str(name)
        if name in seen:
            continue
        seen.add(name)
        if name in by_name:
            chosen.append(by_name[name])
        elif name in scalars:
            chosen.append((name, (name,)))
        else:
            missing.append(name)
    if missing:
        available = list(by_name) + sorted(scalars - set(by_name))
        raise KeyError(
            f"{kind} variable(s) {missing} not found; available: {available}"
        )
    return chosen


def _drop_reserved(plan, reserved, kind: str):
    clashing = [out_name for out_name, _ in plan if out_name in reserved]
    if not clashing:
        return plan
    where = "read_global_fields" if kind == "global" else "read_element_fields"
    warnings.warn(
        f"Not storing {kind} variable(s) {clashing} in the grid: the name is "
        f"reserved by read_exo; use exovista.{where} to read them",
        UserWarning,
        stacklevel=3,
    )
    return [(out_name, comps) for out_name, comps in plan if out_name not in reserved]


def _stack(values: list[np.ndarray], components, out_name) -> np.ndarray:
    """One scalar array, or the components stacked along a last axis."""
    if len(components) == 1 and out_name == components[0]:
        return values[0]
    return np.stack(values, axis=-1)
