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

__all__ = ["Block", "SideSet", "NodeSet", "ExodusModel", "MAX_NAME_LENGTH"]

#: Longest entity / variable name an Exodus file can store (``len_name``).
MAX_NAME_LENGTH = 256


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


def _check_name(name: str, what: str) -> str:
    name = str(name)
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(
            f"{what} name {name[:40]!r}... has {len(name)} characters; Exodus "
            f"allows at most {MAX_NAME_LENGTH}"
        )
    return name


@dataclass(frozen=True)
class Block:
    """An element block.

    Attributes
    ----------
    id : int
        Exodus block id (any integer; usually >= 1).
    name : str
        Block name ("" for unnamed blocks).
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
    times : ndarray of float64, shape (n_steps,)
        Time value of each step (may be empty when there are no variables).
    node_vars : dict of str to ndarray, shape (n_steps, n_nodes)
        Nodal variables.
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
                str(k): np.asarray(v, dtype=np.float64)
                for k, v in self.node_vars.items()
            },
        )
        object.__setattr__(
            self,
            "elem_vars",
            {
                str(k): {int(b): np.asarray(a, dtype=np.float64) for b, a in v.items()}
                for k, v in self.elem_vars.items()
            },
        )
        object.__setattr__(
            self,
            "global_vars",
            {
                str(k): np.asarray(v, dtype=np.float64).reshape(-1)
                for k, v in self.global_vars.items()
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

        _check_unique([b.id for b in self.blocks], "element block")
        _check_unique([s.id for s in self.side_sets], "side set")
        _check_unique([s.id for s in self.node_sets], "node set")

        for block in self.blocks:
            if not block.elem_type:
                raise ValueError(f"Block {block.id} has no element type")
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

        if self.elem_id_map is not None and self.elem_id_map.shape != (num_elem,):
            raise ValueError(
                f"elem_id_map has length {self.elem_id_map.shape[0]}, expected "
                f"num_elem={num_elem}"
            )

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
