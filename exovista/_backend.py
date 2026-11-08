"""The only bridge between exovista's data model and the ExodusII core.

Everything that touches the vendored Sandia core (:mod:`exovista.file`,
:mod:`exovista.parallel_file`, :mod:`exovista.nc`) goes through this module,
so swapping the core (e.g. for the upstream ``exodusii`` package) is a
one-file change.

Public entry points
-------------------
* :func:`write_model` -- write an :class:`~exovista._model.ExodusModel`.
* :func:`open_reader` -- context manager yielding an :class:`ExodusReader`
  over a serial file or a parallel (Nemesis) decomposition.
* :func:`read_model` -- convenience: read a whole file into an ExodusModel.
"""

from __future__ import annotations

import errno
import glob
import os
import stat
import uuid
import warnings
from contextlib import contextmanager
from typing import Iterator

import numpy as np

from . import exodus_h as ex
from ._model import Block, ExodusModel, NodeSet, SideSet, component_suffixes
from .file import exodusii_file
from .parallel_file import parallel_exodusii_file

__all__ = ["write_model", "open_reader", "read_model", "ExodusReader"]

_COORD_NAMES = ("x", "y", "z")

_FLOAT32_MAX = float(np.finfo(np.float32).max)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_model(
    filename: str | os.PathLike,
    model: ExodusModel,
    *,
    float_word_size: int = 8,
    compression_level: int | None = None,
) -> None:
    """Write ``model`` to an ExodusII file.

    The model is validated first; the data is then written to a temporary
    file in the target directory which replaces ``filename`` only once it is
    complete, so a failure never leaves a truncated or half-written file
    behind (an existing ``filename`` is left untouched).

    Parameters
    ----------
    filename : str or os.PathLike
        Output path.
    model : ExodusModel
        What to write.
    float_word_size : {8, 4}, optional
        Floating point storage: 8 (float64, default) or 4 (float32).
    compression_level : int, optional
        Deflate level 0-9 (0 disables compression). ``None`` uses the
        ``EXODUSII_COMPRESSION_LEVEL`` environment default (1).

    Notes
    -----
    Writes the init parameters, coordinates (coordinate names ``x, y[, z]``),
    element blocks with names (and attributes, if any), the element id map
    (``elem_num_map``) when given, node sets and side sets with names, QA and
    info records when given, the time values, and node, element and global
    variables. Element variables get a truth table: a variable is stored only
    on the blocks for which the model provides values.

    The ``time_step`` dimension gets the fixed length ``model.n_steps`` (it
    is unlimited only when there are no steps): reading a netCDF-4 file with
    an unlimited dimension costs a scan of every variable per variable read.
    The file can therefore not be extended with more time steps in place.
    Compressed time-dependent variables get one time step per chunk, so that
    reading one step decompresses only that step.

    If ``filename`` is a symbolic link, the file it points to is replaced
    (the link is kept); an existing file keeps its permission bits. An
    existing file that is not writable is not replaced: ``PermissionError``
    is raised (as ``open(filename, "w")`` would), although replacing it by a
    rename would be allowed. Errors raised while the file is closed (when
    HDF5 writes deferred data) are raised too, before anything is replaced.
    """
    if not isinstance(model, ExodusModel):
        raise TypeError(f"expected an ExodusModel, got {type(model).__name__}")
    if int(float_word_size) not in (4, 8):
        raise ValueError(f"float_word_size must be 4 or 8, got {float_word_size!r}")
    if compression_level is not None and not 0 <= int(compression_level) <= 9:
        raise ValueError(
            f"compression_level must be in 0..9, got {compression_level!r}"
        )
    model.validate()
    if int(float_word_size) == 4:
        _check_single_precision(model)

    # Replace the file a symbolic link points to, not the link itself.
    target = os.path.realpath(os.fspath(filename))
    directory = os.path.dirname(target)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"directory does not exist: {directory}")
    if os.path.exists(target) and not os.access(target, os.W_OK):
        # Replacing it by a rename would succeed where open(target, "w")
        # fails: refuse, like open() (and exovista 1.x) do.
        raise PermissionError(
            errno.EACCES, "Permission denied (the file is not writable)",
            os.fspath(filename),
        )
    # A short temporary name: "<target name>.<hex>.tmp" would exceed NAME_MAX
    # for long (but valid) target names.
    tmp = os.path.join(directory, f".exovista-{uuid.uuid4().hex[:16]}.tmp")
    try:
        with exodusii_file(
            tmp,
            mode="w",
            float_word_size=float_word_size,
            compression_level=compression_level,
            num_time_steps=model.n_steps,
        ) as f, np.errstate(over="ignore"):
            # (float32 overflow to inf was reported above)
            _write(f, model)
        try:
            # keep the permissions of the file being replaced
            os.chmod(tmp, stat.S_IMODE(os.stat(target).st_mode))
        except FileNotFoundError:
            pass
        os.replace(tmp, target)
    except BaseException:
        _remove_quietly(tmp)
        raise


def _remove_quietly(path: str) -> None:
    """Remove ``path`` if it exists, even when it is read-only (Windows)."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        try:
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
            os.remove(path)
        except OSError:
            pass


def _finite_max_abs(values: np.ndarray) -> float:
    """Largest finite magnitude in ``values`` (0.0 if none)."""
    values = np.asarray(values)
    while values.ndim and values.shape[0] > 1 and values.strides[0] == 0:
        values = values[:1]  # a static array broadcast over the steps
    if values.size == 0:
        return 0.0
    magnitude = np.abs(values)
    magnitude = magnitude[np.isfinite(magnitude)]
    return float(magnitude.max()) if magnitude.size else 0.0


def _check_single_precision(model: ExodusModel) -> None:
    """Refuse coordinates / times and warn for variables that overflow float32."""
    for what, values in (("coordinates", model.coords), ("times", model.times)):
        biggest = _finite_max_abs(values)
        if biggest > _FLOAT32_MAX:
            raise ValueError(
                f"{what} reach {biggest:g}, beyond the single-precision range "
                f"(+-{_FLOAT32_MAX:.4g}); write with double precision"
            )
    overflow = []
    for kind, variables in (
        ("node", model.node_vars),
        ("global", model.global_vars),
    ):
        overflow += [
            f"{kind} {name!r}" for name, values in variables.items()
            if _finite_max_abs(values) > _FLOAT32_MAX
        ]
    for name, per_block in model.elem_vars.items():
        if any(_finite_max_abs(v) > _FLOAT32_MAX for v in per_block.values()):
            overflow.append(f"element {name!r}")
    if overflow:
        warnings.warn(
            "values beyond the single-precision range are stored as +-inf in "
            f"variable(s) {', '.join(overflow)}; write with double precision "
            "to keep them",
            UserWarning,
            stacklevel=4,
        )


def _write(f: exodusii_file, model: ExodusModel) -> None:
    num_dim = model.num_dim
    f.put_init(
        model.title,
        num_dim,
        model.num_nodes,
        model.num_elem,
        len(model.blocks),
        len(model.node_sets),
        len(model.side_sets),
    )
    if model.num_nodes:
        f.put_coord_names(list(_COORD_NAMES[:num_dim]))
        f.put_coords(model.coords)

    for block in model.blocks:
        n_attr = 0 if block.attributes is None else block.attributes.shape[1]
        f.put_element_block(
            block.id,
            block.elem_type,
            block.n_elem,
            block.n_nodes_per_elem,
            num_attr=n_attr or None,
        )
        f.put_element_block_name(block.id, block.name)
        if block.n_elem:
            f.put_element_conn(block.id, block.conn + 1)
            if n_attr:
                f.put_element_attr(block.id, block.attributes)

    if model.elem_id_map is not None and model.num_elem:
        f.put_element_id_map(model.elem_id_map)

    for ns in model.node_sets:
        f.put_node_set_param(ns.id, len(ns.nodes))
        f.put_node_set_name(ns.id, ns.name)
        if len(ns.nodes):
            f.put_node_set_nodes(ns.id, ns.nodes + 1)

    for ss in model.side_sets:
        f.put_side_set_param(ss.id, len(ss.elems))
        f.put_side_set_name(ss.id, ss.name)
        if len(ss.elems):
            f.put_side_set_sides(ss.id, ss.elems + 1, ss.sides)

    if model.qa_records:
        f.put_qa(len(model.qa_records), [tuple(r) for r in model.qa_records])
    if model.info:
        f.put_info(len(model.info), list(model.info))

    # Time values first: every variable is dimensioned on the time axis.
    for step, time in enumerate(model.times, start=1):
        f.put_time(step, float(time))
    n_steps = model.n_steps

    # Whole histories are written at once (time_step=None); the time axis
    # already has n_steps entries.
    if model.global_vars:
        names = list(model.global_vars)
        f.put_global_variable_params(len(names))
        f.put_global_variable_names(names)
        values = np.column_stack([model.global_vars[n] for n in names])
        f.put_global_variable_values(None, values.reshape(n_steps, len(names)))

    if model.node_vars and model.num_nodes:
        names = list(model.node_vars)
        f.put_node_variable_params(len(names))
        f.put_node_variable_names(names)
        for name in names:
            f.put_node_variable_values(None, name, model.node_vars[name])

    if model.elem_vars and model.blocks:
        names = list(model.elem_vars)
        table = np.array(
            [
                [
                    int(block.n_elem > 0 and block.id in model.elem_vars[name])
                    for name in names
                ]
                for block in model.blocks
            ],
            dtype=int,
        )
        f.put_element_variable_params(len(names), truth_table=table)
        f.put_element_variable_names(names)
        for b, block in enumerate(model.blocks):
            for i, name in enumerate(names):
                if table[b, i]:
                    values = model.elem_vars[name][block.id]
                    f.put_element_variable_values(None, block.id, name, values)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _chars_to_str(chars) -> str:
    """Decode a 1D netCDF char array (bytes items) to a stripped string."""
    raw = b"".join(c for c in np.asarray(chars).reshape(-1).tolist() if c)
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace").strip()


def _str_list(names) -> list[str]:
    # Like the Exodus library, only trailing blanks are insignificant.
    if names is None:
        return []
    return [str(n).rstrip() for n in np.atleast_1d(names)]


def _unique_names(names: list[str], kind: str) -> list[str]:
    """Usable, unique variable names (warns when one had to be changed).

    An empty name becomes ``f"{kind}_var_{index}"`` (1-based index) and a
    repeated name gets ``"_"`` appended until it is unique (as VTK's reader
    does), so every stored variable stays reachable by name.
    """
    first: dict[str, int] = {}
    for i, name in enumerate(names):
        if name and name not in first:
            first[name] = i
    used = set(first)
    out, changed = [], []
    for i, name in enumerate(names):
        if name and first[name] == i:
            out.append(name)
            continue
        candidate = name + "_" if name else f"{kind}_var_{i + 1}"
        while candidate in used:
            candidate += "_"
        used.add(candidate)
        out.append(candidate)
        changed.append(f"{i + 1}: {name!r} -> {candidate!r}")
    if changed:
        warnings.warn(
            f"Renamed {kind} variable(s) with an empty or repeated name: "
            + ", ".join(changed),
            UserWarning,
            stacklevel=4,
        )
    return out


class ExodusReader:
    """Read-only view of an open Exodus database (serial or parallel).

    Obtain one with :func:`open_reader`. Indices are 0-based and node /
    element variables follow file order. ``step`` arguments are *resolved*
    0-based time-step indices (the 1-based / negative ``time_step`` semantics
    of the public API live in :mod:`exovista.read_pyvista`); a single step is
    read by slicing the netCDF variable, never by loading the whole history.

    Names keep leading whitespace and lose trailing whitespace, as with the
    Exodus library. Variables are read by their index in the file, and
    variable names are made unique: an empty name becomes
    ``"<kind>_var_<index>"`` and a repeated one gets ``"_"`` appended (with a
    warning). Every variable is a scalar series: a nodal variable stored with
    extra component axes (meshio writes vector point data as
    ``vals_nod_varN`` with shape ``(time_step, num_nodes, n)``) is exposed as
    ``n`` scalar variables named with
    :func:`~exovista._model.component_suffixes` (``vec_X``, ``vec_Y``, ...).

    Element blocks the reader cannot represent (``NSIDED`` / ``NFACED``
    polygons and polyhedra, or blocks without a 2D connectivity array) are
    returned with their element count and an ``(n_elem, 0)`` connectivity,
    so element numbering stays aligned with the file.
    """

    def __init__(self, handle: exodusii_file):
        self._f = handle
        self._parallel = isinstance(handle, parallel_exodusii_file)
        self._times: np.ndarray | None = None
        self._cache: dict[str, object] = {}
        # public name -> (1-based stored index, component or None, n_comp)
        self._node_vars: dict[str, tuple[int, int | None, int]] | None = None

    # -- metadata --------------------------------------------------------- #
    @property
    def num_dim(self) -> int:
        """Spatial dimension stored in the file."""
        return int(self._f.num_dimensions())

    @property
    def title(self) -> str:
        """Database title."""
        return str(self._f.title() or "").strip()

    @property
    def num_nodes(self) -> int:
        """Number of (global) nodes."""
        return int(self._f.num_nodes() or 0)

    @property
    def num_elem(self) -> int:
        """Number of (global) elements."""
        return int(self._f.num_elems() or 0)

    @property
    def num_steps(self) -> int:
        """Number of time steps."""
        return int(self._times_cached().shape[0])

    @property
    def float_word_size(self) -> int:
        """Word size of the stored floating point data (4 or 8)."""
        fh = self._f.files[0]
        return int(getattr(fh, ex.ATT_FLT_WORDSIZE, 8))

    @property
    def is_parallel(self) -> bool:
        """True for a joined parallel decomposition."""
        return self._parallel

    # -- mesh --------------------------------------------------------------- #
    def coords(self) -> np.ndarray:
        """Nodal coordinates, float64 ``(num_nodes, num_dim)``."""
        coords = np.asarray(self._f.get_coords(), dtype=np.float64)
        return coords.reshape(-1, self.num_dim)

    def _cached(self, key: str, compute):
        # The file is read-only, so metadata such as names never changes.
        if key not in self._cache:
            self._cache[key] = compute()
        value = self._cache[key]
        return list(value) if isinstance(value, list) else value

    def _block_ids(self) -> list[int]:
        def compute():
            ids = self._f.get_element_block_ids()
            return [] if ids is None else [int(i) for i in np.atleast_1d(ids)]

        return self._cached("block_ids", compute)

    def _block_iid(self, block_id: int) -> int:
        """1-based index of an element block id (KeyError if unknown)."""
        index = self._cached(
            "block_iid", lambda: {b: i for i, b in enumerate(self._block_ids(), 1)}
        )
        try:
            return index[int(block_id)]
        except KeyError:
            raise KeyError(f"{block_id} is not an element block id") from None

    def blocks(self) -> list[Block]:
        """Element blocks in file order (names may be "").

        NULL blocks (no elements) are returned with an empty ``(0, 0)``
        connectivity and ``elem_type`` "NULL" when the file stores none.
        Blocks whose connectivity is not a 2D ``(n_elem, n_nodes)`` array
        (``NSIDED``, ``NFACED``) keep their element count with an
        ``(n_elem, 0)`` connectivity.
        """
        return self._cached("blocks", self._read_blocks)

    def _read_blocks(self) -> list[Block]:
        f = self._f
        out = []
        for iid, block_id in enumerate(self._block_ids(), start=1):
            name = str(f.get_element_block_name(block_id) or "").rstrip()
            elem_type, n_elem, conn = self._block_data(block_id, iid)
            if conn is None:
                conn = np.zeros((n_elem, 0), dtype=np.int64)
                out.append(Block(block_id, name, elem_type, conn))
                continue
            out.append(
                Block(
                    block_id,
                    name,
                    elem_type,
                    conn,
                    attributes=self._block_attributes(block_id, n_elem),
                )
            )
        return out

    def _block_data(self, block_id: int, iid: int):
        """``(elem_type, n_elem, conn (0-based) or None)`` of one block."""
        f = self._f
        if self._parallel:
            info = f.get_element_block(block_id)
            if info is None or not info.num_block_elems:
                return "NULL" if info is None else str(info.elem_type), 0, None
            n_elem = int(info.num_block_elems)
            elem_type = info.elem_type
            conn = f.get_element_conn(block_id) if elem_type is not None else None
        else:
            n_elem = int(f.get_dimension(ex.DIM_NUM_ELEM_IN_ELEM_BLK(iid), 0) or 0)
            var = f.get_variable(ex.VAR_ELEM_BLK_CONN(iid), raw=True)
            if var is None:
                # NFACED blocks store face connectivity instead
                var = f.get_variable(ex.VAR_FACE_CONN(iid), raw=True)
            elem_type = getattr(var, "elem_type", None) if var is not None else None
            if not n_elem:
                return "NULL" if elem_type is None else str(elem_type), 0, None
            conn = None
            if var is not None and len(var.shape) == 2 and var.shape[0] == n_elem:
                conn = var[:]
        elem_type = "UNKNOWN" if elem_type is None else str(elem_type)
        if conn is None:
            return elem_type, n_elem, None
        conn = np.asarray(np.ma.getdata(conn), dtype=np.int64)
        if conn.ndim != 2 or conn.shape[0] != n_elem:
            return elem_type, n_elem, None
        return elem_type, n_elem, conn - 1

    def _block_attributes(self, block_id: int, n_elem: int) -> np.ndarray | None:
        if self._parallel:
            return None  # not supported by the parallel core
        iid = self._block_iid(block_id)
        attr = self._f.get_variable(ex.VAR_ELEM_ATTRIB(iid))
        if attr is None:
            return None
        return np.asarray(attr, dtype=np.float64).reshape(n_elem, -1)

    def elem_id_map(self) -> np.ndarray | None:
        """1-based element ids in file order (``elem_num_map``), or None.

        None means the file stores no map (ids are 1..num_elem). For a
        parallel decomposition these are the global element ids in the joined
        order (blocks in order, ids increasing within each block).
        """

        def compute():
            if self._parallel:
                return np.asarray(self._f.get_element_id_map(), dtype=np.int64)
            values = self._f.get_variable(ex.VAR_ELEM_NUM_MAP)
            if values is None:
                return None
            return np.asarray(values, dtype=np.int64).reshape(-1)

        ids = self._cached("elem_id_map", compute)
        return None if ids is None else ids.copy()

    def side_sets(self) -> list[SideSet]:
        """Side sets in file order (elements 0-based in file order)."""
        f = self._f
        ids = f.get_side_set_ids()
        out = []
        for set_id in [] if ids is None else np.atleast_1d(ids):
            set_id = int(set_id)
            name = str(f.get_side_set_name(set_id) or "").rstrip()
            elems = f.get_side_set_elems(set_id)
            sides = f.get_side_set_sides(set_id)
            if elems is None or sides is None:
                elems = sides = np.zeros(0, dtype=np.int64)
            elems = np.asarray(elems, dtype=np.int64).reshape(-1) - 1
            sides = np.asarray(sides, dtype=np.int64).reshape(-1)
            out.append(SideSet(set_id, name, elems, sides))
        return out

    def node_sets(self) -> list[NodeSet]:
        """Node sets in file order (0-based node indices)."""
        f = self._f
        ids = f.get_node_set_ids()
        out = []
        for set_id in [] if ids is None else np.atleast_1d(ids):
            set_id = int(set_id)
            name = str(f.get_node_set_name(set_id) or "").rstrip()
            nodes = f.get_node_set_nodes(set_id)
            if nodes is None:
                nodes = np.zeros(0, dtype=np.int64)
            nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
            if self._parallel:
                # The parallel core returns global node ids; coordinates use
                # the contiguous numbering of those ids.
                out.append(NodeSet(set_id, name, f.global_node_index(nodes)))
            else:
                out.append(NodeSet(set_id, name, nodes - 1))
        return out

    # -- results -------------------------------------------------------------- #
    def _times_cached(self) -> np.ndarray:
        if self._times is None:
            times = self._f.get_times()
            times = np.zeros(0) if times is None else times
            self._times = np.asarray(times, dtype=np.float64).reshape(-1)
        return self._times

    def times(self) -> np.ndarray:
        """Time values, float64 ``(num_steps,)`` (empty if none)."""
        return self._times_cached().copy()

    def _node_variables(self) -> dict[str, tuple[int, int | None, int]]:
        """Nodal variable name -> (1-based stored index, component, n_comp)."""
        if self._node_vars is None:
            stored = _unique_names(
                _str_list(self._f.get_node_variable_names()), "node"
            )
            n_comps = [self._node_var_components(i) for i in range(1, len(stored) + 1)]
            taken = {name for name, n in zip(stored, n_comps) if n is None}
            mapping: dict[str, tuple[int, int | None, int]] = {}
            for iid, (name, n) in enumerate(zip(stored, n_comps), start=1):
                if n is None:
                    mapping[name] = (iid, None, 1)
                    continue
                if n == 0:
                    continue  # an empty component axis holds no values
                if n == 1:
                    names = [name]
                else:
                    names = [name + suffix for suffix in component_suffixes(n)]
                    if taken.intersection(names) or len(set(names)) != n:
                        names = [f"{name}_{k}" for k in range(n)]
                if taken.intersection(names):
                    raise ValueError(
                        f"cannot name the {n} components of nodal variable "
                        f"{name!r}: {sorted(taken.intersection(names))} exist"
                    )
                taken.update(names)
                for k, component in enumerate(names):
                    mapping[component] = (iid, k, n)
            self._node_vars = mapping
        return self._node_vars

    def _node_var_components(self, iid: int) -> int | None:
        """Components of a node variable stored with extra trailing axes.

        None for a standard ``(time_step, num_nodes)`` variable.
        """
        if self._parallel:
            return None
        var = self._f.get_variable(ex.VAR_NODE_VAR(iid), raw=True)
        if var is None or len(var.shape) <= 2:
            return None
        return int(np.prod(var.shape[2:]))

    def _elem_variables(self) -> dict[str, int]:
        """Element variable name -> 1-based stored index."""
        return self._cached(
            "elem_vars",
            lambda: {
                name: iid
                for iid, name in enumerate(
                    _unique_names(
                        _str_list(self._f.get_element_variable_names()), "elem"
                    ),
                    start=1,
                )
            },
        )

    def _global_variables(self) -> dict[str, int]:
        """Global variable name -> 1-based stored index."""
        return self._cached(
            "global_vars",
            lambda: {
                name: iid
                for iid, name in enumerate(
                    _unique_names(
                        _str_list(self._f.get_global_variable_names()), "global"
                    ),
                    start=1,
                )
            },
        )

    def node_var_names(self) -> list[str]:
        """Names of the nodal variables (one per component, see class doc)."""
        return list(self._node_variables())

    def elem_var_names(self) -> list[str]:
        """Names of the element variables."""
        return list(self._elem_variables())

    def global_var_names(self) -> list[str]:
        """Names of the global variables."""
        return list(self._global_variables())

    def _step_index(self, step: int | None):
        """0-based ``step`` -> netCDF index of the time axis (all steps: None)."""
        if step is None:
            return slice(None)
        step = int(step)
        n = self.num_steps
        if not 0 <= step < n:
            raise IndexError(f"step {step} is out of range for {n} time step(s)")
        return step

    @staticmethod
    def _lookup(name: str, variables: dict, kind: str):
        try:
            return variables[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a {kind} variable; available: {list(variables)}"
            ) from None

    def node_var(self, name: str, step: int | None = None) -> np.ndarray:
        """Values of nodal variable ``name``.

        Returns ``(num_steps, num_nodes)`` for ``step=None``, else the
        ``(num_nodes,)`` values at 0-based ``step``. Always float64.
        """
        iid, component, n_comp = self._lookup(name, self._node_variables(), "node")
        index = self._step_index(step)
        if self._parallel:
            values = self._f.node_variable_values_by_index(iid, index)
        else:
            values = self._f.get_variable(ex.VAR_NODE_VAR(iid), index=index)
            if values is None:
                # "normal model" files store every nodal variable in one
                # (time_step, num_nod_var, num_nodes) variable
                values = self._f.get_variable(
                    ex.VAR_NODE_VAR_COMBINED, index=(index, iid - 1)
                )
        if values is None:
            warnings.warn(
                f"nodal variable {name!r} has no stored values; it reads as NaN",
                UserWarning,
                stacklevel=2,
            )
            shape = (self.num_nodes,) if step is not None else (
                self.num_steps, self.num_nodes)
            return np.full(shape, np.nan)
        values = np.asarray(values, dtype=np.float64)
        if component is not None:
            lead = 1 if step is not None else 2
            values = values.reshape(values.shape[:lead] + (n_comp,))
            values = np.ascontiguousarray(values[..., component])
        return values

    def elem_var(
        self, name: str, block_id: int, step: int | None = None
    ) -> np.ndarray | None:
        """Values of element variable ``name`` on block ``block_id``.

        Returns ``(num_steps, n_block_elems)`` for ``step=None``, else the
        ``(n_block_elems,)`` values at 0-based ``step``; None when the variable
        is not defined on that block (truth table). Always float64.
        """
        iid = self._lookup(name, self._elem_variables(), "element")
        block_iid = self._block_iid(block_id)
        index = self._step_index(step)
        if self._parallel:
            values = self._f.element_variable_values_by_index(
                int(block_id), iid, index
            )
        else:
            values = self._f.get_variable(ex.VAR_ELEM_VAR(iid, block_iid), index=index)
        if values is None:
            return None
        return np.asarray(values, dtype=np.float64)

    def global_var(self, name: str, step: int | None = None) -> np.ndarray | float:
        """Values of global variable ``name``.

        Returns a float64 ``(num_steps,)`` array for ``step=None``, else the
        float value at 0-based ``step``.
        """
        iid = self._lookup(name, self._global_variables(), "global")
        index = self._step_index(step)
        values = self._f.get_variable(ex.VAR_GLO_VAR, index=(index, iid - 1))
        if values is None:
            values = np.full(self.num_steps, np.nan)[index]
        if step is None:
            return np.asarray(values, dtype=np.float64).reshape(-1)
        return float(np.asarray(values, dtype=np.float64).reshape(()))

    def qa_records(self) -> list[tuple[str, ...]]:
        """QA records as tuples of 4 strings (empty if none)."""
        var = self._f.get_variable(ex.VAR_QA_TITLE, raw=True)
        if var is None:
            return []
        chars = np.ma.getdata(var[:])
        return [tuple(_chars_to_str(field) for field in row) for row in chars]

    def info(self) -> list[str]:
        """Information records (empty if none)."""
        var = self._f.get_variable(ex.VAR_INFO, raw=True)
        if var is None:
            return []
        return [_chars_to_str(row) for row in np.ma.getdata(var[:])]


def _expand(filename, files) -> list[str]:
    paths = [os.fspath(filename), *[os.fspath(f) for f in files]]
    found: list[str] = []
    for path in paths:
        if os.path.exists(path):
            found.append(path)
            continue
        matches = sorted(glob.glob(path)) if glob.has_magic(path) else []
        if not matches:
            raise FileNotFoundError(path)
        found.extend(matches)
    return found


@contextmanager
def open_reader(filename: str | os.PathLike, *files: str) -> Iterator[ExodusReader]:
    """Open an Exodus database for reading; the file(s) are closed on exit.

    Parameters
    ----------
    filename : str or os.PathLike
        The file, or the first part of a parallel decomposition. Glob
        patterns (``"mesh.exo.4.*"``) are expanded.
    *files : str
        Remaining parts of a parallel decomposition. When more than one file
        results, they are joined (global node / element numbering) through
        the parallel core.

    Yields
    ------
    ExodusReader
    """
    paths = _expand(filename, files)
    if len(paths) > 1:
        handle = parallel_exodusii_file(*paths)
    else:
        handle = exodusii_file(paths[0], mode="r")
    try:
        for fh in handle.files:
            if ex.DIM_NUM_DIM not in fh.dimensions:
                raise ValueError(
                    f"{fh.filepath()}: not an ExodusII file (no "
                    f"{ex.DIM_NUM_DIM!r} dimension)"
                )
        yield ExodusReader(handle)
    finally:
        handle.close()


def read_model(filename: str | os.PathLike, *files: str) -> ExodusModel:
    """Read a whole Exodus database (all time steps) into an ExodusModel."""
    with open_reader(filename, *files) as r:
        blocks = r.blocks()
        elem_vars: dict[str, dict[int, np.ndarray]] = {}
        for name in r.elem_var_names():
            per_block = {}
            for block in blocks:
                if not block.n_elem:
                    continue
                values = r.elem_var(name, block.id)
                if values is not None:
                    per_block[block.id] = values
            elem_vars[name] = per_block
        return ExodusModel(
            title=r.title,
            num_dim=r.num_dim,
            coords=r.coords(),
            blocks=blocks,
            side_sets=r.side_sets(),
            node_sets=r.node_sets(),
            elem_id_map=r.elem_id_map(),
            times=r.times(),
            node_vars={n: r.node_var(n) for n in r.node_var_names()},
            elem_vars=elem_vars,
            global_vars={n: r.global_var(n) for n in r.global_var_names()},
            qa_records=r.qa_records() or None,
            info=r.info() or None,
        )
