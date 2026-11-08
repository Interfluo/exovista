"""Single source of truth for VTK <-> ExodusII element types.

This module knows, for every element type exovista supports:

* the canonical Exodus element-type name written to disk (e.g. ``"HEX20"``);
* the Exodus names accepted when reading (case-insensitive; e.g. ``HEX``,
  ``HEX8``, ``HEXAHEDRON``; the node count decides the order, as in Exodus);
* the VTK cell type produced on read;
* the node permutation between VTK and Exodus local node order;
* the number of corner (vertex) nodes;
* the Exodus side table: the local corner-node indices (Exodus node order,
  0-based) of every 1-based Exodus side, in Exodus side order. Corner nodes are
  listed in the order (outward-normal orientation and starting node) of the
  Exodus library's ``ex_get_side_set_node_list``, which is what VTK's ExodusII
  reader returns for side sets.

Conventions
-----------
Connectivity arrays are 2D integer arrays ``(n_cells, n_nodes_per_cell)``.
``exo_conn = vtk_conn[:, vtk_to_exo]`` and ``vtk_conn = exo_conn[:, exo_to_vtk]``.

Every permutation and side table here is verified against VTK's own ExodusII
reader in ``test/test_celltypes.py``.

Two-dimensional cells map to planar Exodus types (``TRI*``/``QUAD*``) when the
file has ``num_dim == 2`` and to shell types (``TRISHELL*``/``SHELL4``/
``SHELL8``) when ``num_dim == 3``. Biquadratic quads are written as ``QUAD9``
in both cases: VTK's ExodusII reader (<= 9.7) reads ``SHELL9`` blocks with an
8-node stride, corrupting every element after the first. ``SHELL9`` is still
accepted on read. Like the Exodus library, a ``TRI`` element in a 3D file has
shell-like sides (two faces then three edges).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ExoType",
    "VTK",
    "SUPPORTED_VTK_TYPES",
    "EXO_TYPES",
    "UnsupportedCellTypeError",
    "exo_type_for",
    "exo_info_for",
    "resolve_exo_type",
    "vtk_type_for",
    "vtk_to_exo_perm",
    "to_exodus_conn",
    "to_vtk_conn",
    "side_corner_nodes",
    "n_corners",
    "vtk_type_name",
]


class VTK:
    """VTK cell type ids used by exovista (values of ``vtkCellType.h``)."""

    VERTEX = 1
    LINE = 3
    TRIANGLE = 5
    PIXEL = 8
    QUAD = 9
    TETRA = 10
    VOXEL = 11
    HEXAHEDRON = 12
    WEDGE = 13
    PYRAMID = 14
    QUADRATIC_EDGE = 21
    QUADRATIC_TRIANGLE = 22
    QUADRATIC_QUAD = 23
    QUADRATIC_TETRA = 24
    QUADRATIC_HEXAHEDRON = 25
    QUADRATIC_WEDGE = 26
    QUADRATIC_PYRAMID = 27
    BIQUADRATIC_QUAD = 28
    TRIQUADRATIC_HEXAHEDRON = 29


class UnsupportedCellTypeError(ValueError):
    """Raised for a VTK cell type or Exodus element type exovista cannot map."""


@dataclass(frozen=True)
class ExoType:
    """Definition of one Exodus element type.

    Attributes
    ----------
    name : str
        Canonical Exodus element-type name written to disk (e.g. ``"HEX8"``).
    family : str
        Exodus topology family (``"HEX"``, ``"TRI"``, ``"SHELL"``, ...).
    n_nodes : int
        Nodes per element.
    n_corners : int
        Corner (vertex) nodes per element; they are always the first
        ``n_corners`` nodes in both VTK and Exodus order.
    dim : int
        Topological dimension of the element (0-3).
    vtk_type : int
        VTK cell type produced when reading this Exodus type.
    exo_to_vtk : tuple of int
        ``vtk_conn = exo_conn[:, exo_to_vtk]``.
    sides : tuple of tuple of int
        Exodus side table. ``sides[k]`` lists the 0-based local (Exodus order)
        corner nodes of 1-based side ``k + 1``, outward oriented.
    """

    name: str
    family: str
    n_nodes: int
    n_corners: int
    dim: int
    vtk_type: int
    exo_to_vtk: tuple[int, ...]
    sides: tuple[tuple[int, ...], ...]

    @property
    def vtk_to_exo(self) -> tuple[int, ...]:
        """Inverse of :attr:`exo_to_vtk`: ``exo_conn = vtk_conn[:, vtk_to_exo]``."""
        return tuple(int(i) for i in np.argsort(self.exo_to_vtk))

    @property
    def n_sides(self) -> int:
        """Number of Exodus sides."""
        return len(self.sides)


def _identity(n: int) -> tuple[int, ...]:
    return tuple(range(n))


# --------------------------------------------------------------------------- #
# Exodus side tables (corner nodes, Exodus local order, 0-based, per 1-based
# side). The linear 3D tables are the ones previously verified in
# write_pyvista.py (``_vtk_face_nodes`` reordered by ``vtk2exo_faceorder``),
# written here in Exodus node order. They agree with the SEACAS Exodus
# ``ex_get_side_set_node_list`` tables.
# --------------------------------------------------------------------------- #
_TET_SIDES = ((0, 1, 3), (1, 2, 3), (0, 3, 2), (0, 2, 1))
_HEX_SIDES = (
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (0, 4, 7, 3),
    (0, 3, 2, 1),
    (4, 5, 6, 7),
)
_WEDGE_SIDES = ((0, 1, 4, 3), (1, 2, 5, 4), (0, 3, 5, 2), (0, 2, 1), (3, 4, 5))
_PYRAMID_SIDES = ((0, 1, 4), (1, 2, 4), (2, 3, 4), (0, 4, 3), (0, 3, 2, 1))
_TRI_SIDES = ((0, 1), (1, 2), (2, 0))
_QUAD_SIDES = ((0, 1), (1, 2), (2, 3), (3, 0))
# Shells: sides 1 and 2 are the two faces (positive / negative normal), the
# remaining sides are the edges. (The Exodus library starts the reversed
# triangle at node 3, the reversed quadrilateral at node 1.)
_TRISHELL_SIDES = ((0, 1, 2), (2, 1, 0), (0, 1), (1, 2), (2, 0))
_SHELL_SIDES = ((0, 1, 2, 3), (0, 3, 2, 1), (0, 1), (1, 2), (2, 3), (3, 0))
# BAR / TRUSS: the sides are the two end nodes.
_BAR_SIDES = ((0,), (1,))
# BEAM: the element itself, with opposite orientations.
_BEAM_SIDES = ((0, 1), (1, 0))

# Exodus -> VTK node permutations for the types whose node order differs.
# HEX20: Exodus puts the vertical (1-5, 2-6, 3-7, 4-8) mid-edge nodes before
# the top ones; VTK puts the top ones first.
_HEX20_EXO_TO_VTK = (*range(12), 16, 17, 18, 19, 12, 13, 14, 15)
# HEX27: Exodus node 21 (1-based) is the centroid and 22-27 are the mid-face
# nodes of the faces z-, z+, x-, x+, y-, y+. VTK has the mid-face nodes of
# x-, x+, y-, y+, z-, z+ followed by the centroid.
_HEX27_EXO_TO_VTK = (*_HEX20_EXO_TO_VTK, 23, 24, 25, 26, 21, 22, 20)
# WEDGE15: Exodus lists vertical mid-edge nodes (1-4, 2-5, 3-6) before the top
# triangle's; VTK lists the top triangle's first.
_WEDGE15_EXO_TO_VTK = (*range(9), 12, 13, 14, 9, 10, 11)


def _make(name, family, n_nodes, n_corners, dim, vtk_type, sides, perm=None):
    return ExoType(
        name=name,
        family=family,
        n_nodes=n_nodes,
        n_corners=n_corners,
        dim=dim,
        vtk_type=vtk_type,
        exo_to_vtk=tuple(perm) if perm is not None else _identity(n_nodes),
        sides=sides,
    )


EXO_TYPES: dict[str, ExoType] = {
    t.name: t
    for t in (
        # 0D / 1D
        _make("SPHERE", "SPHERE", 1, 1, 0, VTK.VERTEX, ()),
        _make("BAR2", "BAR", 2, 2, 1, VTK.LINE, _BAR_SIDES),
        _make("BAR3", "BAR", 3, 2, 1, VTK.QUADRATIC_EDGE, _BAR_SIDES),
        # read-only: written as BAR*, but BEAM has different sides
        _make("BEAM2", "BEAM", 2, 2, 1, VTK.LINE, _BEAM_SIDES),
        _make("BEAM3", "BEAM", 3, 2, 1, VTK.QUADRATIC_EDGE, _BEAM_SIDES),
        # 2D planar (num_dim == 2)
        _make("TRI3", "TRI", 3, 3, 2, VTK.TRIANGLE, _TRI_SIDES),
        _make("TRI6", "TRI", 6, 3, 2, VTK.QUADRATIC_TRIANGLE, _TRI_SIDES),
        _make("QUAD4", "QUAD", 4, 4, 2, VTK.QUAD, _QUAD_SIDES),
        _make("QUAD8", "QUAD", 8, 4, 2, VTK.QUADRATIC_QUAD, _QUAD_SIDES),
        _make("QUAD9", "QUAD", 9, 4, 2, VTK.BIQUADRATIC_QUAD, _QUAD_SIDES),
        # 2D cells in 3D space (num_dim == 3)
        _make("TRISHELL3", "TRISHELL", 3, 3, 2, VTK.TRIANGLE, _TRISHELL_SIDES),
        _make(
            "TRISHELL6", "TRISHELL", 6, 3, 2, VTK.QUADRATIC_TRIANGLE,
            _TRISHELL_SIDES,
        ),
        _make("SHELL4", "SHELL", 4, 4, 2, VTK.QUAD, _SHELL_SIDES),
        _make("SHELL8", "SHELL", 8, 4, 2, VTK.QUADRATIC_QUAD, _SHELL_SIDES),
        _make("SHELL9", "SHELL", 9, 4, 2, VTK.BIQUADRATIC_QUAD, _SHELL_SIDES),
        # 3D
        _make("TETRA4", "TETRA", 4, 4, 3, VTK.TETRA, _TET_SIDES),
        _make("TETRA10", "TETRA", 10, 4, 3, VTK.QUADRATIC_TETRA, _TET_SIDES),
        _make("HEX8", "HEX", 8, 8, 3, VTK.HEXAHEDRON, _HEX_SIDES),
        _make(
            "HEX20", "HEX", 20, 8, 3, VTK.QUADRATIC_HEXAHEDRON, _HEX_SIDES,
            _HEX20_EXO_TO_VTK,
        ),
        _make(
            "HEX27", "HEX", 27, 8, 3, VTK.TRIQUADRATIC_HEXAHEDRON, _HEX_SIDES,
            _HEX27_EXO_TO_VTK,
        ),
        _make("WEDGE6", "WEDGE", 6, 6, 3, VTK.WEDGE, _WEDGE_SIDES),
        _make(
            "WEDGE15", "WEDGE", 15, 6, 3, VTK.QUADRATIC_WEDGE, _WEDGE_SIDES,
            _WEDGE15_EXO_TO_VTK,
        ),
        _make("PYRAMID5", "PYRAMID", 5, 5, 3, VTK.PYRAMID, _PYRAMID_SIDES),
        _make(
            "PYRAMID13", "PYRAMID", 13, 5, 3, VTK.QUADRATIC_PYRAMID,
            _PYRAMID_SIDES,
        ),
    )
}

# Exodus name prefixes accepted on read -> topology family.
_FAMILY_ALIASES: dict[str, str] = {
    "SPHERE": "SPHERE",
    "CIRCLE": "SPHERE",
    "BAR": "BAR",
    "TRUSS": "BAR",
    "BEAM": "BEAM",
    "TRI": "TRI",
    "TRIANGLE": "TRI",
    "TRISHELL": "TRISHELL",
    "QUAD": "QUAD",
    "QUADRILATERAL": "QUAD",
    "SHELL": "SHELL",
    "TET": "TETRA",
    "TETRA": "TETRA",
    "TETRAHEDRON": "TETRA",
    "HEX": "HEX",
    "HEXAHEDRON": "HEX",
    "WEDGE": "WEDGE",
    "PYRAMID": "PYRAMID",
    "PYRA": "PYRAMID",
}

# family -> {n_nodes: ExoType}
_BY_FAMILY: dict[str, dict[int, ExoType]] = {}
for _t in EXO_TYPES.values():
    _BY_FAMILY.setdefault(_t.family, {})[_t.n_nodes] = _t
del _t


@dataclass(frozen=True)
class _VtkMapping:
    planar: str | None  # Exodus type for num_dim == 2 (None: not allowed)
    spatial: str  # Exodus type for num_dim == 3
    vtk_to_exo: tuple[int, ...]


def _native(planar, spatial):
    """Mapping for the VTK type that an Exodus type reads back as."""
    return _VtkMapping(planar, spatial, EXO_TYPES[spatial].vtk_to_exo)


_VTK_TO_EXO: dict[int, _VtkMapping] = {
    VTK.VERTEX: _native("SPHERE", "SPHERE"),
    VTK.LINE: _native("BAR2", "BAR2"),
    VTK.QUADRATIC_EDGE: _native("BAR3", "BAR3"),
    VTK.TRIANGLE: _native("TRI3", "TRISHELL3"),
    VTK.QUADRATIC_TRIANGLE: _native("TRI6", "TRISHELL6"),
    VTK.QUAD: _native("QUAD4", "SHELL4"),
    VTK.PIXEL: _VtkMapping("QUAD4", "SHELL4", (0, 1, 3, 2)),
    VTK.QUADRATIC_QUAD: _native("QUAD8", "SHELL8"),
    # Not SHELL9 in 3D: VTK (<= 9.7) reads SHELL9 blocks with an 8-node stride,
    # corrupting every element after the first. QUAD9 is read correctly.
    VTK.BIQUADRATIC_QUAD: _native("QUAD9", "QUAD9"),
    VTK.TETRA: _native(None, "TETRA4"),
    VTK.QUADRATIC_TETRA: _native(None, "TETRA10"),
    VTK.HEXAHEDRON: _native(None, "HEX8"),
    VTK.VOXEL: _VtkMapping(None, "HEX8", (0, 1, 3, 2, 4, 5, 7, 6)),
    VTK.QUADRATIC_HEXAHEDRON: _native(None, "HEX20"),
    VTK.TRIQUADRATIC_HEXAHEDRON: _native(None, "HEX27"),
    VTK.WEDGE: _native(None, "WEDGE6"),
    VTK.QUADRATIC_WEDGE: _native(None, "WEDGE15"),
    VTK.PYRAMID: _native(None, "PYRAMID5"),
    VTK.QUADRATIC_PYRAMID: _native(None, "PYRAMID13"),
}

#: VTK cell types exovista can write.
SUPPORTED_VTK_TYPES: frozenset[int] = frozenset(_VTK_TO_EXO)

_NAME_RE = re.compile(r"^([A-Z]+?)(\d*)$")


def vtk_type_name(vtk_type: int) -> str:
    """Return a readable name for a VTK cell type id (e.g. ``"POLYHEDRON"``)."""
    try:
        import pyvista as pv

        return pv.CellType(int(vtk_type)).name
    except (ImportError, ValueError):
        return f"VTK cell type {int(vtk_type)}"


def _vtk_mapping(vtk_type: int) -> _VtkMapping:
    try:
        return _VTK_TO_EXO[int(vtk_type)]
    except KeyError:
        raise UnsupportedCellTypeError(
            f"{vtk_type_name(vtk_type)} cells cannot be written to Exodus"
        ) from None


def exo_info_for(vtk_type: int, num_dim: int) -> ExoType:
    """Return the :class:`ExoType` a VTK cell type is written as.

    Parameters
    ----------
    vtk_type : int
        VTK cell type id (``pyvista.CellType`` values are accepted).
    num_dim : int
        Spatial dimension of the Exodus file (2 or 3). 2D cells become
        ``TRI*``/``QUAD*`` when ``num_dim == 2`` and ``TRISHELL*``/``SHELL*``
        when ``num_dim == 3``.

    Raises
    ------
    UnsupportedCellTypeError
        If the cell type is not supported, or is a 3D cell in a 2D file.
    """
    mapping = _vtk_mapping(vtk_type)
    if int(num_dim) == 2:
        if mapping.planar is None:
            raise UnsupportedCellTypeError(
                f"{vtk_type_name(vtk_type)} cells cannot be written to a "
                "num_dim=2 Exodus file"
            )
        return EXO_TYPES[mapping.planar]
    if int(num_dim) != 3:
        raise ValueError(f"num_dim must be 2 or 3, got {num_dim!r}")
    return EXO_TYPES[mapping.spatial]


def exo_type_for(vtk_type: int, num_dim: int) -> str:
    """Return the canonical Exodus element-type name for a VTK cell type.

    See :func:`exo_info_for` for the parameters and errors.
    """
    return exo_info_for(vtk_type, num_dim).name


def vtk_to_exo_perm(vtk_type: int) -> np.ndarray:
    """Permutation ``p`` such that ``exo_conn = vtk_conn[:, p]`` for ``vtk_type``."""
    return np.asarray(_vtk_mapping(vtk_type).vtk_to_exo, dtype=np.intp)


def resolve_exo_type(
    exo_name: str, n_nodes: int | None = None, num_dim: int | None = None
) -> ExoType | None:
    """Resolve an Exodus element-type name (as found in a file) to an ExoType.

    Parameters
    ----------
    exo_name : str
        Element type as stored in the file (``elem_type`` attribute). The match
        is case-insensitive and ignores surrounding whitespace. Any alias in
        the family (``HEX``, ``HEX8``, ``HEXAHEDRON``, ``BEAM2``, ...) works.
    n_nodes : int, optional
        Nodes per element in the block. As in Exodus, it decides the element
        order (``HEX`` with 20 nodes is ``HEX20``). Defaults to the number in
        the name, else the linear element.
    num_dim : int, optional
        Spatial dimension of the file. A ``TRI`` element in a 3D file is
        treated as a ``TRISHELL`` (shell-like sides), as the Exodus library
        does.

    Returns
    -------
    ExoType or None
        ``None`` if the type is unknown or the node count is not supported.
    """
    key = str(exo_name).strip().upper()
    match = _NAME_RE.match(key)
    if match is None:
        return None
    prefix, digits = match.groups()
    family = _FAMILY_ALIASES.get(prefix)
    if family is None:
        return None
    if family == "TRI" and num_dim is not None and int(num_dim) == 3:
        family = "TRISHELL"
    by_nodes = _BY_FAMILY[family]
    if n_nodes is None:
        n_nodes = int(digits) if digits else min(by_nodes)
    return by_nodes.get(int(n_nodes))


def _resolve_or_raise(exo_name, n_nodes=None, num_dim=None) -> ExoType:
    info = resolve_exo_type(exo_name, n_nodes, num_dim)
    if info is None:
        extra = "" if n_nodes is None else f" with {n_nodes} nodes"
        raise UnsupportedCellTypeError(
            f"Unsupported Exodus element type {exo_name!r}{extra}"
        )
    return info


def vtk_type_for(
    exo_name: str, n_nodes: int | None = None, num_dim: int | None = None
) -> tuple[int, np.ndarray] | None:
    """Return ``(vtk_type, exo_to_vtk)`` for an Exodus element type, or None.

    ``vtk_conn = exo_conn[:, exo_to_vtk]``. See :func:`resolve_exo_type` for
    the parameters.
    """
    info = resolve_exo_type(exo_name, n_nodes, num_dim)
    if info is None:
        return None
    return info.vtk_type, np.asarray(info.exo_to_vtk, dtype=np.intp)


def to_exodus_conn(vtk_type: int, conn: np.ndarray) -> np.ndarray:
    """Permute VTK-ordered connectivity ``(n_cells, n_nodes)`` to Exodus order.

    Node ids are not changed (no 0/1-based conversion).
    """
    perm = vtk_to_exo_perm(vtk_type)
    conn = np.asarray(conn)
    if conn.ndim != 2 or conn.shape[1] != len(perm):
        raise ValueError(
            f"{vtk_type_name(vtk_type)} connectivity must have shape "
            f"(n, {len(perm)}), got {conn.shape}"
        )
    return conn[:, perm]


def to_vtk_conn(
    exo_name: str, conn: np.ndarray, num_dim: int | None = None
) -> np.ndarray:
    """Permute Exodus-ordered connectivity ``(n_cells, n_nodes)`` to VTK order.

    The element order is taken from ``conn.shape[1]``. Raises
    :class:`UnsupportedCellTypeError` for unknown types.
    """
    conn = np.asarray(conn)
    if conn.ndim != 2:
        raise ValueError(f"connectivity must be 2D, got shape {conn.shape}")
    info = _resolve_or_raise(exo_name, conn.shape[1], num_dim)
    return conn[:, np.asarray(info.exo_to_vtk, dtype=np.intp)]


def side_corner_nodes(
    exo_name: str, n_nodes: int | None = None, num_dim: int | None = None
) -> list[tuple[int, ...]]:
    """Return the Exodus side table of an element type.

    Entry ``k`` lists the 0-based local corner nodes (Exodus node order,
    outward oriented) of 1-based side ``k + 1``. Parameters as for
    :func:`resolve_exo_type`; raises :class:`UnsupportedCellTypeError` for
    unknown types.
    """
    return list(_resolve_or_raise(exo_name, n_nodes, num_dim).sides)


def n_corners(exo_name: str, n_nodes: int | None = None) -> int:
    """Number of corner (vertex) nodes of an Exodus element type."""
    return _resolve_or_raise(exo_name, n_nodes).n_corners
