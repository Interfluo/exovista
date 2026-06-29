"""ExodusII to PyVista import module.

This module is the read counterpart to :mod:`exovista.write_pyvista`. It loads
an ExodusII database into a :class:`pyvista.UnstructuredGrid`, reconstructing:

- Node coordinates (2D meshes are embedded in 3D with ``z = 0``).
- Element blocks (concatenated, in block order) with the originating Exodus
  block id and name recorded as cell data.
- Node variables (point data) and element variables (cell data) sampled at a
  chosen time step.
- The time history, exposed via ``grid.field_data["times"]``.

It handles both serial files and globbed parallel decompositions through
:func:`exovista.File`.
"""

import numpy as np
import pyvista as pv

from .file import exodusii_file
from .parallel_file import parallel_exodusii_file
from .write_pyvista import _vtk_face_nodes, vtk2exo_faceorder


# Inverse of write_pyvista.vtk2exo. Keys are the canonical Exodus element type
# names (uppercased, as stored on disk via ATT_NAME_ELEM_TYPE). PIXEL/VOXEL are
# absent because write_exo normalizes them to QUAD/HEX before writing.
_exo2vtk = {
    "TRI": pv.CellType.TRIANGLE,
    "TRI3": pv.CellType.TRIANGLE,
    "TRIANGLE": pv.CellType.TRIANGLE,
    "TRIANGLE3": pv.CellType.TRIANGLE,
    "QUAD": pv.CellType.QUAD,
    "QUAD4": pv.CellType.QUAD,
    "QUADRILATERAL": pv.CellType.QUAD,
    "TET": pv.CellType.TETRA,
    "TET4": pv.CellType.TETRA,
    "TETRA": pv.CellType.TETRA,
    "TETRA4": pv.CellType.TETRA,
    "TETRAHEDRON": pv.CellType.TETRA,
    "HEX": pv.CellType.HEXAHEDRON,
    "HEX8": pv.CellType.HEXAHEDRON,
    "HEXAHEDRON": pv.CellType.HEXAHEDRON,
    "WEDGE": pv.CellType.WEDGE,
    "WEDGE6": pv.CellType.WEDGE,
    "PYRAMID": pv.CellType.PYRAMID,
    "PYRAMID5": pv.CellType.PYRAMID,
    "PYRA": pv.CellType.PYRAMID,
    "PYRA5": pv.CellType.PYRAMID,
}


def _resolve_vtk_cell_type(elem_type):
    """Map an Exodus element-type string to a PyVista/VTK cell type."""
    key = str(elem_type).strip().upper()
    if key in _exo2vtk:
        return _exo2vtk[key]
    # Fall back to the alphabetic prefix (e.g. "HEX20" -> "HEX") so linear
    # readers degrade gracefully on higher-order names they share a base with.
    base = key.rstrip("0123456789")
    if base in _exo2vtk:
        return _exo2vtk[base]
    raise ValueError(
        f"Unsupported Exodus element type {elem_type!r} for PyVista import"
    )


def _open(filename, *files):
    """Open ``filename`` (and any extra parallel parts) for reading."""
    if files:
        return parallel_exodusii_file(filename, *files)
    return exodusii_file(filename, mode="r")


def read_exo(filename, *files, time_step=-1, read_node_variables=True,
             read_element_variables=True, read_side_sets=False):
    """Read an ExodusII file into a :class:`pyvista.UnstructuredGrid`.

    This is the inverse of :func:`exovista.write_exo`.

    Parameters
    ----------
    filename : str
        Path to the ExodusII file. For a parallel decomposition pass the
        remaining parts as additional positional arguments.
    *files : str
        Additional file parts of a parallel decomposition.
    time_step : int, optional
        1-based time step at which to sample node/element variables. Negative
        values index from the end, so the default ``-1`` selects the last step.
        Ignored when the file has no variables.
    read_node_variables : bool, str, or list of str, optional
        Which node variables to load into ``grid.point_data``. ``True``
        (default) loads all, ``False`` loads none, and a name or list of names
        loads only those variables (useful to avoid reading fields you do not
        need). Unknown names raise ``KeyError``.
    read_element_variables : bool, str, or list of str, optional
        Which element variables to load into ``grid.cell_data``. Same
        semantics as ``read_node_variables``.
    read_side_sets : bool, optional
        If True, also reconstruct side sets as surface geometry and return them
        alongside the volume mesh. Default is False.

    Returns
    -------
    pyvista.UnstructuredGrid
        The reconstructed mesh. ``grid.field_data["times"]`` holds the time
        history (empty if the file is static) and ``grid.field_data["num_dim"]``
        records the original spatial dimension.
    pyvista.MultiBlock
        Only returned when ``read_side_sets=True``. A block per side set (named
        after the side set) of :class:`pyvista.PolyData` faces. Each face cell
        carries ``orig_elem_id`` (1-based Exodus element id) and ``face_id``
        (1-based local side id) cell data. The surfaces share the volume's node
        coordinates, so node ids index back into ``grid``.
    """
    f = _open(filename, *files)
    try:
        points = _read_points(f)
        blocks = _read_blocks(f)
        grid = _build_grid(
            f, points, blocks, time_step,
            read_node_variables, read_element_variables,
        )
        if read_side_sets:
            side_sets = _reconstruct_side_sets(f, points, blocks)
            return grid, side_sets
        return grid
    finally:
        f.close()


def _read_points(f):
    """Return node coordinates embedded in 3D (lower dims padded with zeros)."""
    coords = np.asarray(f.get_coords(), dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    points = np.zeros((coords.shape[0], 3), dtype=float)
    points[:, : coords.shape[1]] = coords
    return points


def _read_blocks(f):
    """Parse element blocks once, in block-id order.

    Returns a list of dicts with the block ``id``, ``name``, resolved VTK
    ``vtk_type``, 0-based ``conn`` array, and the ``start`` global element
    offset (so 1-based Exodus element ids map back to a block and local row).
    """
    block_ids = f.get_element_block_ids()
    if block_ids is None:
        block_ids = []

    blocks = []
    start = 0
    for block_id in block_ids:
        block = f.get_element_block(block_id)
        if block is None or block.num_block_elems == 0:
            continue
        conn = np.asarray(f.get_element_conn(block_id), dtype=np.int64)
        conn = conn.reshape(block.num_block_elems, block.num_elem_nodes) - 1
        blocks.append({
            "id": block_id,
            "name": block.name,
            "vtk_type": _resolve_vtk_cell_type(block.elem_type),
            "conn": conn,
            "start": start,
        })
        start += conn.shape[0]
    return blocks


def _build_grid(f, points, blocks, time_step,
                read_node_variables, read_element_variables):
    cells = []
    cell_types = []
    cell_block_ids = []
    cell_block_names = []

    for block in blocks:
        conn0 = block["conn"]
        n_cells = conn0.shape[0]
        n_per = conn0.shape[1]
        prefixed = np.hstack(
            [np.full((n_cells, 1), n_per, dtype=np.int64), conn0]
        )
        cells.append(prefixed.ravel())
        cell_types.append(np.full(n_cells, int(block["vtk_type"]), dtype=np.uint8))
        cell_block_ids.append(np.full(n_cells, block["id"], dtype=np.int64))
        cell_block_names.extend([block["name"]] * n_cells)

    if cells:
        grid = pv.UnstructuredGrid(
            np.concatenate(cells), np.concatenate(cell_types), points
        )
        grid.cell_data["exo_block_id"] = np.concatenate(cell_block_ids)
        grid.cell_data["exo_block_name"] = np.array(cell_block_names, dtype=object)
    else:
        # No elements: return a point cloud so downstream code still has points.
        grid = pv.UnstructuredGrid(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8), points
        )

    times = f.get_times()
    times = np.asarray([]) if times is None else np.asarray(times, dtype=float)
    grid.field_data["times"] = times
    grid.field_data["num_dim"] = np.array([f.num_dimensions()])

    if read_node_variables:
        _read_node_variables(f, grid, time_step, select=read_node_variables)
    if read_element_variables and cells:
        ordered_blocks = [(b["id"], b["conn"].shape[0]) for b in blocks]
        _read_element_variables(
            f, grid, time_step, ordered_blocks, select=read_element_variables
        )

    return grid


def _exo_face_local_nodes(vtk_type, side_id):
    """Local node indices of Exodus side ``side_id`` (1-based) for ``vtk_type``.

    Reuses the face tables from :mod:`exovista.write_pyvista`, reordered from VTK
    to Exodus side ordering, so reading is the exact inverse of writing.
    """
    vtk_faces = _vtk_face_nodes[vtk_type]
    face_order = vtk2exo_faceorder[vtk_type]
    ordered_faces = [vtk_faces[i] for i in face_order]
    return ordered_faces[side_id - 1]


def _reconstruct_side_sets(f, points, blocks):
    """Rebuild each side set as a PolyData of faces over the volume nodes."""
    side_sets = pv.MultiBlock()
    set_ids = f.get_side_set_ids()
    if set_ids is None:
        return side_sets

    # Block lookup by global (0-based) element index.
    starts = np.array([b["start"] for b in blocks], dtype=np.int64)

    for set_id in set_ids:
        ss = f.get_side_set(set_id)
        elems = np.asarray(ss.elems, dtype=np.int64)   # 1-based global elem ids
        sides = np.asarray(ss.sides, dtype=np.int64)   # 1-based local side ids

        faces = []
        face_lengths = []
        for elem_id, side_id in zip(elems, sides):
            e0 = elem_id - 1
            bidx = int(np.searchsorted(starts, e0, side="right") - 1)
            block = blocks[bidx]
            local_row = e0 - block["start"]
            local_nodes = _exo_face_local_nodes(block["vtk_type"], side_id)
            global_nodes = block["conn"][local_row][list(local_nodes)]
            faces.append(global_nodes)
            face_lengths.append(len(local_nodes))

        poly = _faces_to_polydata(points, faces, face_lengths)
        poly.cell_data["orig_elem_id"] = elems
        poly.cell_data["face_id"] = sides
        side_sets[ss.name] = poly

    return side_sets


def _faces_to_polydata(points, faces, face_lengths):
    """Build a PolyData from face node lists over the shared ``points`` array.

    2-node faces (2D element edges) become lines; 3+ node faces become polygons.
    """
    if not faces:
        return pv.PolyData(points)

    lines = []
    polys = []
    for nodes, length in zip(faces, face_lengths):
        cell = [length, *[int(n) for n in nodes]]
        if length == 2:
            lines.extend(cell)
        else:
            polys.extend(cell)

    # Pass cells through the constructor so PolyData does not auto-generate a
    # vertex cell for every point (which would inflate the cell count).
    kwargs = {}
    if polys:
        kwargs["faces"] = np.array(polys, dtype=np.int64)
    if lines:
        kwargs["lines"] = np.array(lines, dtype=np.int64)
    return pv.PolyData(points, **kwargs)


def _normalize_selection(select, available):
    """Resolve a ``True``/``False``/``str``/list selection of variable names.

    ``True`` selects every available name, ``False``/``None`` selects nothing,
    and a name or iterable of names selects exactly those (preserving the
    caller's order). Names not present in ``available`` raise ``KeyError`` so a
    typo fails loudly instead of silently reading nothing.
    """
    available = [] if available is None else [str(n) for n in available]
    if select is True:
        return available
    if select is False or select is None:
        return []
    requested = [select] if isinstance(select, str) else [str(n) for n in select]
    missing = [n for n in requested if n not in available]
    if missing:
        raise KeyError(
            f"Variable(s) {missing} not found; available names are {available}"
        )
    return requested


def _read_node_variables(f, grid, time_step, select=True):
    names = _normalize_selection(select, f.get_node_variable_names())
    for name in names:
        values = f.get_node_variable_values(name, time_step=time_step)
        if values is None:
            continue
        values = np.asarray(values)
        if values.shape[0] == grid.n_points:
            grid.point_data[str(name)] = values


def _read_element_variables(f, grid, time_step, ordered_blocks, select=True):
    names = _normalize_selection(select, f.get_element_variable_names())
    for name in names:
        chunks = []
        for block_id, n_block_cells in ordered_blocks:
            values = f.get_element_variable_values(
                block_id, name, time_step=time_step
            )
            if values is None:
                # Variable not defined on this block (truth table); pad with NaN
                # so the concatenated array stays aligned with the cells.
                chunks.append(np.full(n_block_cells, np.nan))
            else:
                chunks.append(np.asarray(values, dtype=float))
        if chunks:
            combined = np.concatenate(chunks)
            if combined.shape[0] == grid.n_cells:
                grid.cell_data[str(name)] = combined


def read_node_fields(filename, *files, names=None):
    """Read node-variable time histories from an Exodus file.

    Inverse of the ``node_fields`` argument to :func:`exovista.write_exo`. The
    mesh (coordinates, connectivity) is *not* reconstructed, so this is the
    lightweight way to pull a transient field's full history out of a file.

    Parameters
    ----------
    filename : str
        Path to the ExodusII file. For a parallel decomposition pass the
        remaining parts as additional positional arguments.
    *files : str
        Additional file parts of a parallel decomposition.
    names : str or list of str, optional
        Name or list of names to read. ``None`` (default) reads every node
        variable. Unknown names raise ``KeyError``.

    Returns
    -------
    dict of {str: ndarray}
        Maps each variable name to an array of shape ``(num_times, num_nodes)``
        -- the same layout :func:`write_exo` accepts in ``node_fields``.
    """
    f = _open(filename, *files)
    try:
        select = True if names is None else names
        wanted = _normalize_selection(select, f.get_node_variable_names())
        out = {}
        for name in wanted:
            values = f.get_node_variable_values(name, time_step=None)
            if values is not None:
                out[str(name)] = np.asarray(values)
        return out
    finally:
        f.close()


def read_element_fields(filename, *files, names=None):
    """Read element-variable time histories from an Exodus file.

    Inverse of the ``element_fields`` argument to :func:`exovista.write_exo`.
    Element variables are stored per block with a truth table, so blocks are
    concatenated in block-id order (NaN-padding any block on which a variable
    is not defined) to reconstruct the global element ordering that
    :func:`write_exo` accepted. The mesh itself is not reconstructed.

    Parameters
    ----------
    filename : str
        Path to the ExodusII file. For a parallel decomposition pass the
        remaining parts as additional positional arguments.
    *files : str
        Additional file parts of a parallel decomposition.
    names : str or list of str, optional
        Name or list of names to read. ``None`` (default) reads every element
        variable. Unknown names raise ``KeyError``.

    Returns
    -------
    dict of {str: ndarray}
        Maps each variable name to an array of shape ``(num_times, num_elems)``
        -- the same layout :func:`write_exo` accepts in ``element_fields``.
    """
    f = _open(filename, *files)
    try:
        select = True if names is None else names
        wanted = _normalize_selection(select, f.get_element_variable_names())
        if not wanted:
            return {}

        block_ids = f.get_element_block_ids()
        block_ids = [] if block_ids is None else list(block_ids)
        block_sizes = []
        for block_id in block_ids:
            block = f.get_element_block(block_id)
            block_sizes.append(0 if block is None else block.num_block_elems)
        num_times = f.num_times()

        out = {}
        for name in wanted:
            chunks = []
            for block_id, n_block_cells in zip(block_ids, block_sizes):
                if n_block_cells == 0:
                    continue
                values = f.get_element_variable_values(
                    block_id, name, time_step=None
                )
                if values is None:
                    # Not defined on this block (truth table); pad with NaN so
                    # the columns stay aligned with the global element order.
                    chunks.append(np.full((num_times, n_block_cells), np.nan))
                else:
                    chunks.append(np.asarray(values, dtype=float))
            if chunks:
                out[str(name)] = np.concatenate(chunks, axis=1)
        return out
    finally:
        f.close()
