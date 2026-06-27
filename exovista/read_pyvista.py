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
             read_element_variables=True):
    """Read an ExodusII file into a :class:`pyvista.UnstructuredGrid`.

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
    read_node_variables : bool, optional
        If True (default) node variables are loaded into ``grid.point_data``.
    read_element_variables : bool, optional
        If True (default) element variables are loaded into ``grid.cell_data``.

    Returns
    -------
    pyvista.UnstructuredGrid
        The reconstructed mesh. ``grid.field_data["times"]`` holds the time
        history (empty if the file is static) and ``grid.field_data["num_dim"]``
        records the original spatial dimension.
    """
    f = _open(filename, *files)
    try:
        return _build_grid(
            f, time_step, read_node_variables, read_element_variables
        )
    finally:
        f.close()


def _build_grid(f, time_step, read_node_variables, read_element_variables):
    num_dim = f.num_dimensions()

    # Coordinates: embed lower-dimensional meshes in 3D space for PyVista.
    coords = np.asarray(f.get_coords(), dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n_points = coords.shape[0]
    points = np.zeros((n_points, 3), dtype=float)
    points[:, : coords.shape[1]] = coords

    block_ids = f.get_element_block_ids()
    if block_ids is None:
        block_ids = []

    cells = []
    cell_types = []
    cell_block_ids = []
    cell_block_names = []
    # Map each global (concatenated) cell back to its (block_id, name) so element
    # variables can be reassembled in the same order they are appended here.
    ordered_blocks = []

    for block_id in block_ids:
        block = f.get_element_block(block_id)
        if block is None or block.num_block_elems == 0:
            continue
        vtk_type = _resolve_vtk_cell_type(block.elem_type)
        conn = np.asarray(f.get_element_conn(block_id), dtype=np.int64)
        conn = conn.reshape(block.num_block_elems, block.num_elem_nodes)
        # Exodus connectivity is 1-based; VTK/PyVista is 0-based.
        conn0 = conn - 1
        n_per = conn0.shape[1]
        prefixed = np.hstack(
            [np.full((conn0.shape[0], 1), n_per, dtype=np.int64), conn0]
        )
        cells.append(prefixed.ravel())
        cell_types.append(np.full(conn0.shape[0], int(vtk_type), dtype=np.uint8))
        cell_block_ids.append(np.full(conn0.shape[0], block_id, dtype=np.int64))
        cell_block_names.extend([block.name] * conn0.shape[0])
        ordered_blocks.append((block_id, conn0.shape[0]))

    if cells:
        cells_arr = np.concatenate(cells)
        types_arr = np.concatenate(cell_types)
        grid = pv.UnstructuredGrid(cells_arr, types_arr, points)
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
    grid.field_data["num_dim"] = np.array([num_dim])

    if read_node_variables:
        _read_node_variables(f, grid, time_step)
    if read_element_variables and cells:
        _read_element_variables(f, grid, time_step, ordered_blocks)

    return grid


def _read_node_variables(f, grid, time_step):
    names = f.get_node_variable_names()
    if names is None:
        return
    for name in names:
        values = f.get_node_variable_values(name, time_step=time_step)
        if values is None:
            continue
        values = np.asarray(values)
        if values.shape[0] == grid.n_points:
            grid.point_data[str(name)] = values


def _read_element_variables(f, grid, time_step, ordered_blocks):
    names = f.get_element_variable_names()
    if names is None:
        return
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
