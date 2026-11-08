"""Plain-data model of an ExodusII database.

These dataclasses sit between the PyVista conversion code
(:mod:`exovista.write_pyvista` / :mod:`exovista.read_pyvista`) and the file
backend (:mod:`exovista._backend`). They carry numpy arrays and use **0-based**
indices internally; the backend converts to and from Exodus' 1-based storage.

Conventions
-----------
* Element connectivity is stored in **Exodus local node order** (use
  :mod:`exovista._celltypes` to convert from/to VTK order).
* Global element indices (side sets) count elements in *file order*: the
  blocks concatenated in the order of :attr:`ExodusModel.blocks`.
* Variable histories are ``(n_steps, n_entities)`` float arrays with one row
  per entry of :attr:`ExodusModel.times`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._celltypes import resolve_exo_type

__all__ = [
    "Block",
    "SideSet",
    "NodeSet",
    "ExodusModel",
    "MAX_NAME_LENGTH",
    "component_suffixes",
    "component_groups",
]

#: Longest entity / variable name an Exodus file can store (``len_name``), in
#: bytes of its UTF-8 encoding.
MAX_NAME_LENGTH = 256

# Ids and the element id map are stored as 32-bit integers.
_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)

# Variable-name suffixes of the components of a multi-component array, by
# number of components (6: VTK symmetric tensor order; 9: row-major tensor).
_COMPONENT_SUFFIXES: dict[int, tuple[str, ...]] = {
    2: ("_X", "_Y"),
    3: ("_X", "_Y", "_Z"),
    6: ("_XX", "_YY", "_ZZ", "_XY", "_YZ", "_XZ"),
    9: ("_XX", "_XY", "_XZ", "_YX", "_YY", "_YZ", "_ZX", "_ZY", "_ZZ"),
}


def component_suffixes(n_components: int) -> tuple[str, ...]:
    """Suffixes naming the scalar Exodus variables of an ``n``-component array.

    2 -> ``_X, _Y``; 3 -> ``_X, _Y, _Z``; 6 (VTK symmetric tensor order) ->
    ``_XX, _YY, _ZZ, _XY, _YZ, _XZ``; 9 (row-major) -> ``_XX, _XY, ... _ZZ``;
    any other ``n`` -> ``_0 ... _{n-1}``. This is the one convention used to
    split arrays into Exodus variables and to recombine them.
    """
    n = int(n_components)
    if n < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components!r}")
    return _COMPONENT_SUFFIXES.get(n, tuple(f"_{i}" for i in range(n)))


# The letter-suffix conventions, largest first (6 is a subset of 9).
_LETTER_CONVENTIONS = tuple(
    tuple(s[1:] for s in _COMPONENT_SUFFIXES[n]) for n in (9, 6, 3, 2)
)
_LETTERS = frozenset(s for convention in _LETTER_CONVENTIONS for s in convention)


def component_groups(names) -> dict[str, list[str]]:
    """Group scalar variable names that are components of one array.

    The inverse of splitting with :func:`component_suffixes`: returns
    ``{base name: [component names in convention order]}`` for every base
    whose components are present. For each base (the text before the last
    ``"_"``) the letter and the digit suffixes are considered separately:

    * letters: the largest complete convention present -- the 9 full-tensor
      suffixes, else the 6 symmetric-tensor ones, else ``X, Y, Z``, else
      ``X, Y``;
    * digits: ``0 .. n-1`` with ``n`` the length of the run of consecutive
      digit suffixes from 0, if ``n`` is a digit convention (``n >= 4`` and
      not 6 or 9).

    Names that are not part of the chosen group (e.g. ``"V_0"`` next to
    ``V_X, V_Y, V_Z``) stay scalars. If both kinds form a group, the larger
    one wins (letters on a tie). A base that is itself one of ``names`` is
    never grouped.
    """
    names = list(names)
    letters: dict[str, dict[str, str]] = {}
    digits: dict[str, dict[int, str]] = {}
    for name in names:
        base, sep, suffix = name.rpartition("_")
        if not base or not sep:
            continue
        if suffix in _LETTERS:
            letters.setdefault(base, {})[suffix] = name
        elif suffix.isascii() and suffix.isdigit() and str(int(suffix)) == suffix:
            digits.setdefault(base, {})[int(suffix)] = name
    existing = set(names)
    groups: dict[str, list[str]] = {}
    for base in dict.fromkeys([*letters, *digits]):
        if base in existing:
            continue
        best: list[str] | None = None
        by_letter = letters.get(base, {})
        for convention in _LETTER_CONVENTIONS:
            if all(s in by_letter for s in convention):
                best = [by_letter[s] for s in convention]
                break
        by_digit = digits.get(base, {})
        n = 0
        while n in by_digit:
            n += 1
        if n >= 4 and n not in _COMPONENT_SUFFIXES and (
            best is None or n > len(best)
        ):
            best = [by_digit[k] for k in range(n)]
        if best is not None:
            groups[base] = best
    return groups


def _int_array(values, ndim: int, what: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        arr = arr.astype(np.int64)
        if ndim == 1:
            arr = arr.reshape(0)
    if arr.ndim != ndim:
        raise ValueError(f"{what} must be {ndim}D, got shape {arr.shape}")
    if arr.dtype.kind not in "iu":
        if arr.dtype.kind == "f" and np.all(np.mod(arr, 1) == 0):
            arr = arr.astype(np.int64)
        else:
            raise TypeError(f"{what} must be integers, got dtype {arr.dtype}")
    return arr.astype(np.int64, copy=False)


def _normalize_name(name) -> str:
    """Entity / variable name as stored: trailing whitespace removed.

    Exodus names are blank padded and every reader (the Exodus library, VTK,
    exovista) strips trailing whitespace, so it cannot survive a round trip.
    """
    return str(name).rstrip()


def _check_name(name: str, what: str) -> str:
    """Normalize ``name`` and check that Exodus can store it."""
    name = _normalize_name(name)
    if "\0" in name:
        raise ValueError(f"{what} name {name!r} contains a NUL character")
    n_bytes = len(name.encode("utf-8"))
    if n_bytes > MAX_NAME_LENGTH:
        raise ValueError(
            f"{what} name {name[:40]!r}... is {n_bytes} bytes long (UTF-8); "
            f"Exodus allows at most {MAX_NAME_LENGTH}"
        )
    return name


def _normalized_keys(mapping: dict, what: str) -> dict:
    """``mapping`` with normalized variable names; reject names that collide."""
    out: dict = {}
    original: dict[str, object] = {}
    for key, value in mapping.items():
        name = _normalize_name(key)
        if name in out:
            raise ValueError(
                f"{what} variable names {original[name]!r} and {key!r} are the "
                "same once trailing whitespace is removed (Exodus strips it)"
            )
        out[name] = value
        original[name] = key
    return out


def _check_id(value: int, what: str) -> None:
    if not _INT32_MIN <= value <= _INT32_MAX:
        raise ValueError(
            f"{what} id {value} does not fit in a 32-bit integer "
            f"({_INT32_MIN}..{_INT32_MAX})"
        )


@dataclass(frozen=True)
class Block:
    """An element block.

    Attributes
    ----------
    id : int
        Exodus block id (any 32-bit integer; usually >= 1).
    name : str
        Block name ("" for unnamed blocks). Trailing whitespace is removed;
        at most 256 bytes (UTF-8).
    elem_type : str
        Exodus element type, e.g. ``"HEX8"`` (as stored in the file on read).
    conn : ndarray of int, shape (n_elem, n_nodes_per_elem)
        0-based node indices, Exodus local node order.
    attributes : ndarray of float, shape (n_elem, n_attr), optional
        Per-element attributes (e.g. SPHERE radius), or None.
    """

    id: int
    name: str
    elem_type: str
    conn: np.ndarray
    attributes: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(self, "name", _check_name(self.name, "Block"))
        object.__setattr__(self, "elem_type", str(self.elem_type).strip())
        conn = np.asarray(self.conn)
        if conn.size == 0 and conn.ndim != 2:
            conn = conn.reshape(0, 0)
        object.__setattr__(self, "conn", _int_array(conn, 2, "Block.conn"))
        if self.attributes is not None:
            attr = np.asarray(self.attributes, dtype=np.float64)
            if attr.ndim == 1:
                attr = attr.reshape(-1, 1)
            if attr.ndim != 2 or attr.shape[0] != self.conn.shape[0]:
                raise ValueError(
                    f"Block {self.id}: attributes must have shape "
                    f"({self.conn.shape[0]}, n_attr), got {attr.shape}"
                )
            object.__setattr__(self, "attributes", attr)

    @property
    def n_elem(self) -> int:
        """Number of elements in the block."""
        return int(self.conn.shape[0])

    @property
    def n_nodes_per_elem(self) -> int:
        """Number of nodes per element."""
        return int(self.conn.shape[1])


@dataclass(frozen=True)
class SideSet:
    """A side set: (element, side) pairs.

    Attributes
    ----------
    id : int
        Exodus side-set id.
    name : str
        Side-set name ("" for unnamed sets).
    elems : ndarray of int, shape (n,)
        0-based global element index in *file order*.
    sides : ndarray of int, shape (n,)
        1-based Exodus side number of each element.
    """

    id: int
    name: str
    elems: np.ndarray
    sides: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(self, "name", _check_name(self.name, "Side set"))
        object.__setattr__(self, "elems", _int_array(self.elems, 1, "SideSet.elems"))
        object.__setattr__(self, "sides", _int_array(self.sides, 1, "SideSet.sides"))
        if self.elems.shape != self.sides.shape:
            raise ValueError(
                f"Side set {self.id}: elems and sides lengths differ "
                f"({len(self.elems)} != {len(self.sides)})"
            )


@dataclass(frozen=True)
class NodeSet:
    """A node set.

    Attributes
    ----------
    id : int
        Exodus node-set id.
    name : str
        Node-set name ("" for unnamed sets).
    nodes : ndarray of int, shape (n,)
        0-based node indices.
    """

    id: int
    name: str
    nodes: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(self, "name", _check_name(self.name, "Node set"))
        object.__setattr__(self, "nodes", _int_array(self.nodes, 1, "NodeSet.nodes"))


@dataclass(frozen=True)
class ExodusModel:
    """Everything exovista writes to (or reads from) an Exodus file.

    Attributes
    ----------
    title : str
        Database title.
    num_dim : int
        Spatial dimension (1, 2 or 3).
    coords : ndarray of float64, shape (n_nodes, num_dim)
        Nodal coordinates.
    blocks : list of Block
        Element blocks in file order.
    side_sets, node_sets : list of SideSet / NodeSet
        Sets in file order.
    elem_id_map : ndarray of int, shape (n_elem,), optional
        1-based element ids in file order (Exodus ``elem_num_map``), or None.
        Ids are unique and stored as 32-bit integers (1 .. 2**31 - 1), like
        the block, side-set and node-set ids.
    times : ndarray of float64, shape (n_steps,)
        Time value of each step (may be empty when there are no variables).
    node_vars : dict of str to ndarray, shape (n_steps, n_nodes)
        Nodal variables. In all variable dicts, trailing whitespace is removed
        from the names (Exodus readers strip it); names that then collide
        raise ``ValueError``.
    elem_vars : dict of str to dict of int to ndarray, shape (n_steps, n_blk)
        Element variables keyed by name, then by block id. A block missing
        from the inner dict has no values for that variable (truth table 0).
    global_vars : dict of str to ndarray, shape (n_steps,)
        Global variables.
    qa_records : list of 4-tuples of str, optional
        QA records (code name, version, date, time).
    info : list of str, optional
        Information records.
    """

    title: str
    num_dim: int
    coords: np.ndarray
    blocks: list[Block] = field(default_factory=list)
    side_sets: list[SideSet] = field(default_factory=list)
    node_sets: list[NodeSet] = field(default_factory=list)
    elem_id_map: np.ndarray | None = None
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    node_vars: dict[str, np.ndarray] = field(default_factory=dict)
    elem_vars: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    global_vars: dict[str, np.ndarray] = field(default_factory=dict)
    qa_records: list[tuple[str, str, str, str]] | None = None
    info: list[str] | None = None

    def __post_init__(self):
        object.__setattr__(self, "title", str(self.title or ""))
        object.__setattr__(self, "num_dim", int(self.num_dim))
        coords = np.asarray(self.coords, dtype=np.float64)
        if coords.ndim == 1 and self.num_dim == 1:
            coords = coords.reshape(-1, 1)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "blocks", list(self.blocks))
        object.__setattr__(self, "side_sets", list(self.side_sets))
        object.__setattr__(self, "node_sets", list(self.node_sets))
        times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "times", times)
        if self.elem_id_map is not None:
            object.__setattr__(
                self, "elem_id_map", _int_array(self.elem_id_map, 1, "elem_id_map")
            )
        object.__setattr__(
            self,
            "node_vars",
            {
                k: np.asarray(v, dtype=np.float64)
                for k, v in _normalized_keys(self.node_vars, "node").items()
            },
        )
        object.__setattr__(
            self,
            "elem_vars",
            {
                k: {int(b): np.asarray(a, dtype=np.float64) for b, a in v.items()}
                for k, v in _normalized_keys(self.elem_vars, "element").items()
            },
        )
        object.__setattr__(
            self,
            "global_vars",
            {
                k: np.asarray(v, dtype=np.float64).reshape(-1)
                for k, v in _normalized_keys(self.global_vars, "global").items()
            },
        )

    # ------------------------------------------------------------------ #
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return int(self.coords.shape[0])

    @property
    def num_elem(self) -> int:
        """Total number of elements over all blocks."""
        return int(sum(b.n_elem for b in self.blocks))

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(self.times.shape[0])

    def block_offsets(self) -> np.ndarray:
        """0-based global index of the first element of each block (+ total)."""
        sizes = [b.n_elem for b in self.blocks]
        return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    def has_variables(self) -> bool:
        """True if any node, element or global variable is present."""
        return bool(self.node_vars or self.elem_vars or self.global_vars)

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Check internal consistency; raise ``ValueError`` on any problem.

        Run by :func:`exovista._backend.write_model` before anything is written,
        so an inconsistent model never produces a file.
        """
        if self.num_dim not in (1, 2, 3):
            raise ValueError(f"num_dim must be 1, 2 or 3, got {self.num_dim}")
        if self.coords.ndim != 2 or self.coords.shape[1] != self.num_dim:
            raise ValueError(
                f"coords must have shape (n_nodes, {self.num_dim}), "
                f"got {self.coords.shape}"
            )
        if not np.all(np.isfinite(self.coords)):
            raise ValueError("coords contain NaN or infinite values")
        n_nodes = self.num_nodes

        for entities, what in (
            (self.blocks, "element block"),
            (self.side_sets, "side set"),
            (self.node_sets, "node set"),
        ):
            ids = [e.id for e in entities]
            _check_unique(ids, what)
            for i in ids:
                _check_id(i, what)

        for block in self.blocks:
            if not block.elem_type:
                raise ValueError(f"Block {block.id} has no element type")
            if block.n_elem and not block.n_nodes_per_elem:
                raise ValueError(
                    f"Block {block.id} ({block.elem_type}) has elements but no "
                    "nodal connectivity; polygon / polyhedron (NSIDED / NFACED) "
                    "blocks cannot be written"
                )
            conn = block.conn
            if conn.size and (conn.min() < 0 or conn.max() >= n_nodes):
                raise ValueError(
                    f"Block {block.id} connectivity references nodes outside "
                    f"0..{n_nodes - 1}"
                )
            # A type name with a node count ("HEX8") must match the width.
            named = None
            if any(c.isdigit() for c in block.elem_type):
                named = resolve_exo_type(block.elem_type, None, self.num_dim)
            if (
                named is not None
                and block.n_elem
                and named.n_nodes != block.n_nodes_per_elem
            ):
                raise ValueError(
                    f"Block {block.id}: {block.elem_type} needs {named.n_nodes} "
                    f"nodes per element, got {block.n_nodes_per_elem}"
                )

        num_elem = self.num_elem
        offsets = self.block_offsets()
        for ss in self.side_sets:
            if ss.elems.size == 0:
                continue
            if ss.elems.min() < 0 or ss.elems.max() >= num_elem:
                raise ValueError(
                    f"Side set {ss.id} references elements outside 0..{num_elem - 1}"
                )
            if ss.sides.min() < 1:
                raise ValueError(f"Side set {ss.id} has side numbers < 1")
            bidx = np.searchsorted(offsets, ss.elems, side="right") - 1
            for k, block in enumerate(self.blocks):
                sel = bidx == k
                if not np.any(sel):
                    continue
                info = resolve_exo_type(
                    block.elem_type, block.n_nodes_per_elem, self.num_dim
                )
                if info is not None and ss.sides[sel].max() > info.n_sides:
                    raise ValueError(
                        f"Side set {ss.id}: {block.elem_type} elements have "
                        f"{info.n_sides} sides, got side {ss.sides[sel].max()}"
                    )

        for ns in self.node_sets:
            if ns.nodes.size and (ns.nodes.min() < 0 or ns.nodes.max() >= n_nodes):
                raise ValueError(
                    f"Node set {ns.id} references nodes outside 0..{n_nodes - 1}"
                )

        if self.elem_id_map is not None:
            id_map = self.elem_id_map
            if id_map.shape != (num_elem,):
                raise ValueError(
                    f"elem_id_map has length {id_map.shape[0]}, expected "
                    f"num_elem={num_elem}"
                )
            if id_map.size and (id_map.min() < 1 or id_map.max() > _INT32_MAX):
                raise ValueError(
                    f"elem_id_map values must be in 1..{_INT32_MAX} (32-bit "
                    f"element ids), got {id_map.min()}..{id_map.max()}"
                )
            if np.unique(id_map).size != id_map.size:
                raise ValueError("elem_id_map contains duplicate element ids")

        n_steps = self.n_steps
        if self.has_variables() and n_steps == 0:
            raise ValueError("variables were given but times is empty")
        names = list(self.node_vars) + list(self.elem_vars) + list(self.global_vars)
        for name in names:
            _check_name(name, "Variable")
        for name, values in self.node_vars.items():
            _check_shape(values, (n_steps, n_nodes), f"node variable {name!r}")
        block_sizes = {b.id: b.n_elem for b in self.blocks}
        for name, per_block in self.elem_vars.items():
            for block_id, values in per_block.items():
                if block_id not in block_sizes:
                    raise ValueError(
                        f"element variable {name!r} refers to unknown block {block_id}"
                    )
                _check_shape(
                    values,
                    (n_steps, block_sizes[block_id]),
                    f"element variable {name!r} on block {block_id}",
                )
        for name, values in self.global_vars.items():
            _check_shape(values, (n_steps,), f"global variable {name!r}")

        for record in self.qa_records or []:
            if len(record) != 4:
                raise ValueError(f"QA records need 4 strings, got {record!r}")


def _check_unique(ids, what):
    seen = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate {what} id {i}")
        seen.add(i)


def _check_shape(values, shape, what):
    if values.shape != shape:
        raise ValueError(f"{what} has shape {values.shape}, expected {shape}")
