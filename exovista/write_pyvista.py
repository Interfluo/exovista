"""
PyVista to ExodusII export module.

This module provides functionality for exporting PyVista UnstructuredGrid meshes
to ExodusII format with support for:
- Element blocks (split by cell type and user-defined regions)
- Side sets (mapped from surface meshes)
- Node arrays (point data)
- Custom block and side set names

Supported Cell Types:
    - 2D: TRIANGLE, QUAD, PIXEL
    - 3D: TETRA, HEXAHEDRON, VOXEL, WEDGE, PYRAMID

Note: VOXEL and PIXEL cells are automatically permuted to Exodus ordering.
"""

import logging
from typing import Optional

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from .file import exodusii_file


vtk2exo = {
    pv.CellType.TRIANGLE: 'tri3',
    pv.CellType.QUAD: 'quad4',
    pv.CellType.PIXEL: 'quad4',
    pv.CellType.TETRA: 'tet4',
    pv.CellType.HEXAHEDRON: 'hex8',
    pv.CellType.VOXEL: 'hex8',
    pv.CellType.WEDGE: 'wedge6',
    pv.CellType.PYRAMID: 'pyramid5',
    # add more as needed
}

vtk2exo_faceorder = {
    pv.CellType.TRIANGLE: [0, 1, 2],
    pv.CellType.QUAD: [0, 1, 2, 3],
    pv.CellType.PIXEL: [0, 1, 2, 3],
    pv.CellType.TETRA: [0, 1, 2, 3],
    pv.CellType.HEXAHEDRON: [2, 1, 3, 0, 4, 5],
    pv.CellType.VOXEL: [2, 1, 3, 0, 4, 5],
    pv.CellType.WEDGE: [2, 3, 4, 0, 1],
    pv.CellType.PYRAMID: [1, 2, 3, 4, 0],
    # add more as needed
}

# Mapping for node permutation when converting VTK cell to Exodus cell
# Some VTK cells (like VOXEL, PIXEL) have different node ordering than their Exodus counterparts
vtk2exo_permute = {
    pv.CellType.PIXEL: [0, 1, 3, 2],  # Pixel -> Quad
    pv.CellType.VOXEL: [0, 1, 3, 2, 4, 5, 7, 6],  # Voxel -> Hex
}

# VTK face definitions: for each cell type, the local node indices forming each face.
# For 2D cells, "faces" are edges. For 3D cells, faces are actual faces.
# These are in VTK face ordering (will be reordered by vtk2exo_faceorder).
_vtk_face_nodes = {
    pv.CellType.TRIANGLE: [
        [0, 1],  # edge 0
        [1, 2],  # edge 1
        [2, 0],  # edge 2
    ],
    pv.CellType.QUAD: [
        [0, 1],  # edge 0
        [1, 2],  # edge 1
        [2, 3],  # edge 2
        [3, 0],  # edge 3
    ],
    pv.CellType.PIXEL: [
        [0, 1],  # edge 0
        [1, 3],  # edge 1
        [2, 3],  # edge 2
        [0, 2],  # edge 3
    ],
    pv.CellType.TETRA: [
        [0, 1, 3],  # face 0
        [1, 2, 3],  # face 1
        [2, 0, 3],  # face 2
        [0, 2, 1],  # face 3
    ],
    pv.CellType.HEXAHEDRON: [
        [0, 4, 7, 3],  # face 0
        [1, 2, 6, 5],  # face 1
        [0, 1, 5, 4],  # face 2
        [3, 7, 6, 2],  # face 3
        [0, 3, 2, 1],  # face 4
        [4, 5, 6, 7],  # face 5
    ],
    pv.CellType.VOXEL: [
        [2, 0, 6, 4],  # face 0
        [1, 3, 5, 7],  # face 1
        [0, 1, 4, 5],  # face 2
        [3, 2, 7, 6],  # face 3
        [1, 0, 3, 2],  # face 4
        [4, 5, 6, 7],  # face 5
    ],
    pv.CellType.WEDGE: [
        [0, 1, 2],     # face 0
        [3, 5, 4],     # face 1
        [0, 3, 4, 1],  # face 2
        [1, 4, 5, 2],  # face 3
        [2, 5, 3, 0],  # face 4
    ],
    pv.CellType.PYRAMID: [
        [0, 3, 2, 1],  # face 0
        [0, 1, 4],     # face 1
        [1, 2, 4],     # face 2
        [2, 3, 4],     # face 3
        [3, 0, 4],     # face 4
    ],
}


def _compute_face_centers_vectorized(coords, connectivity, cell_type):
    """
    Compute face centers for all cells of a given type using vectorized numpy ops.

    Parameters
    ----------
    coords : np.ndarray, shape (n_total_points, 3)
        Global coordinate array.
    connectivity : np.ndarray, shape (n_cells, n_nodes_per_cell)
        Global node indices for each cell.
    cell_type : pv.CellType
        The VTK cell type.

    Returns
    -------
    np.ndarray, shape (n_cells * n_faces, 3)
        Face center coordinates in Exodus face ordering.
    """
    vtk_faces = _vtk_face_nodes[cell_type]
    face_order = vtk2exo_faceorder[cell_type]
    n_cells = connectivity.shape[0]
    n_faces = len(vtk_faces)

    # Reorder faces from VTK to Exodus ordering
    ordered_faces = [vtk_faces[i] for i in face_order]

    all_centers = np.empty((n_cells, n_faces, 3), dtype=np.float64)
    for exo_fid, face_nodes in enumerate(ordered_faces):
        # face_nodes: list of local node indices for this face
        # connectivity[:, face_nodes] -> (n_cells, n_nodes_in_face) global node IDs
        face_global_ids = connectivity[:, face_nodes]  # (n_cells, n_nodes_in_face)
        # coords[face_global_ids] -> (n_cells, n_nodes_in_face, 3)
        face_coords = coords[face_global_ids]
        # Mean across nodes -> (n_cells, 3)
        all_centers[:, exo_fid, :] = face_coords.mean(axis=1)

    # Reshape to (n_cells * n_faces, 3)
    return all_centers.reshape(-1, 3)


def _get_block_face_info_vectorized(coords, connectivity, cell_ids, cell_type, block_id):
    """
    Compute face center info for a block, returning arrays instead of PolyData.

    Parameters
    ----------
    coords : np.ndarray, shape (n_total_points, 3)
        Global coordinate array.
    connectivity : np.ndarray, shape (n_cells, n_nodes_per_cell)
        Global node indices for each cell.
    cell_ids : np.ndarray
        Global cell IDs for the cells in this block.
    cell_type : pv.CellType
        The VTK cell type.
    block_id : int or str
        The block ID (region) for this block.

    Returns
    -------
    tuple of (points, face_cell_ids, face_ids, block_ids)
        All as numpy arrays.
    """
    n_faces = len(_vtk_face_nodes[cell_type])
    n_cells = len(cell_ids)

    # Face centers: (n_cells * n_faces, 3)
    centers = _compute_face_centers_vectorized(coords, connectivity, cell_type)

    # cell_ids repeated for each face: [c0,c0,...,c1,c1,...,...]
    face_cell_ids = np.repeat(cell_ids, n_faces)

    # face_ids cycle: [0,1,...,n_faces-1, 0,1,...,n_faces-1, ...]
    face_ids = np.tile(np.arange(n_faces), n_cells)

    # block_ids: constant
    block_ids = np.full(n_cells * n_faces, block_id)

    return centers, face_cell_ids, face_ids, block_ids


def process_meshes(volume: pv.UnstructuredGrid, surface: pv.PolyData, region_key: str = "region") -> tuple[dict, pv.MultiBlock]:
    """
    Process volume and surface meshes to prepare for ExodusII export.

    Splits the volume mesh into blocks by cell type and region. Maps surface faces
    to corresponding volume faces using KD-tree nearest-neighbor lookup.

    Parameters
    ----------
    volume : pv.UnstructuredGrid
        The volume mesh.
    surface : pv.PolyData
        The surface mesh.
    region_key : str, optional
        The name of the cell data array defining regions (default is "region").

    Returns
    -------
    tuple[dict, pv.MultiBlock]
        Dictionary of volume blocks and MultiBlock of surface side sets.
    """
    # Check and fallback for volume regions
    if region_key not in volume.cell_data:
        logging.warning(f"Region key '{region_key}' not found in volume cell data. Setting all regions to 1.")
        volume.cell_data[region_key] = np.ones(volume.n_cells, dtype=int)

    volume_regions = np.unique(volume[region_key])
    logging.info(f"Volume regions found: {volume_regions}")

    if surface is not None:
        if region_key not in surface.cell_data:
            logging.warning(f"Region key '{region_key}' not found in surface cell data. Setting all regions to 1.")
            surface.cell_data[region_key] = np.ones(surface.n_cells, dtype=int)
        surface_regions = np.unique(surface[region_key])
        logging.info(f"Surface regions found: {surface_regions}")
    else:
        logging.info("No surface provided. Skipping surface processing.")

    # Split volume blocks by cell_type and region - only iterate over types that exist
    volume_blocks = {}
    cell_id, counter = 0, 0
    unique_cell_types = np.unique(volume.celltypes)

    for ct_value in unique_cell_types:
        cell_type = pv.CellType(ct_value)
        if cell_type not in vtk2exo:
            logging.warning(f"Unsupported cell type {cell_type.name} ({ct_value}). Skipping.")
            continue

        volume_cell_type = volume.extract_cells_by_type(cell_type)
        if volume_cell_type.n_cells == 0:
            continue

        logging.info(f"Processing cell type: {cell_type.name} with {volume_cell_type.n_cells} cells")
        for m in volume_cell_type.split_values(scalars=region_key):
            region_value = m[region_key][0] if region_key in m.cell_data else 1
            logging.info(f"  - Sub-block with region {region_value} and {m.n_cells} cells")
            volume_blocks[counter] = {
                "region": region_value,
                "cell_type": cell_type,
                "mesh": m,
                "cell_ids": np.arange(cell_id, cell_id + m.n_cells),
            }
            cell_id += m.n_cells
            counter += 1

    if not volume_blocks:
        logging.warning("No volume blocks found after processing. Export may fail.")

    if surface is not None:
        # Build face center KD-tree using vectorized computation
        all_centers = []
        all_cell_ids = []
        all_face_ids = []
        all_block_ids = []

        coords = volume.points

        for key, item in volume_blocks.items():
            mesh = item["mesh"]
            n_nodes_per_cell = mesh.get_cell(0).n_points
            # Extract global node indices from the orig_pts mapping
            local_conn = mesh.cells.reshape(-1, n_nodes_per_cell + 1)[:, 1:]
            global_conn = mesh['orig_pts'][local_conn]

            centers, fc_ids, f_ids, b_ids = _get_block_face_info_vectorized(
                coords, global_conn, item["cell_ids"], item["cell_type"], item["region"]
            )
            all_centers.append(centers)
            all_cell_ids.append(fc_ids)
            all_face_ids.append(f_ids)
            all_block_ids.append(b_ids)

        all_centers = np.concatenate(all_centers)
        all_cell_ids = np.concatenate(all_cell_ids)
        all_face_ids = np.concatenate(all_face_ids)
        all_block_ids = np.concatenate(all_block_ids)

        logging.info(f"Generated {len(all_centers)} face center points for interpolation.")

        # Single KD-tree query for all surface cell centers
        tree = cKDTree(all_centers)
        query_points = surface.cell_centers().points
        _, indices = tree.query(query_points)

        surface["cell_ids"] = all_cell_ids[indices]
        surface["face_ids"] = all_face_ids[indices]
        surface["block_ids"] = all_block_ids[indices]
        logging.info("Surface face mapping completed.")

        surfaces = surface.split_values(scalars=region_key)
    else:
        surfaces = pv.MultiBlock()

    return volume_blocks, surfaces


def write_exo(filename,
              volume: pv.UnstructuredGrid,
              surface: pv.PolyData = None,
              region_key: str = "region",
              block_names: list[str] = None,
              side_set_names: list[str] = None,
              save_node_arrays: bool = True,
              ):

    """
    Write PyVista volume and surface meshes to an ExodusII file.

    Handles element blocks and side sets based on region assignments.
    Automatically splits blocks by cell type as required by ExodusII.

    Parameters
    ----------
    filename : str
        Path to the output ExodusII file.
    volume : pv.UnstructuredGrid
        The volume mesh.
    surface : pv.PolyData
        The surface mesh.
    region_key : str, optional
        The name of the cell data array defining regions (default is "region").
    block_names: list[str], optional
        list of names for the element blocks, default naming is ["block_0", "block_1" ...]
    side_set_names: list[str], optional
        list of names for the side sets, default naming is ["set_0", "set_1" ...]
    save_node_arrays: bool = True,
        if True, will save node arrays from volume mesh to the exo file

    Returns
    -------
    None
    """

    logging.info(f"Writing ExodusII file: {filename}")
    logging.info(f"Using region key: '{region_key}'")

    f = open(filename, "w+")
    f.close()

    # Process meshes
    volume.point_data["orig_pts"] = np.arange(volume.n_points)
    volume.cell_data["orig_cell_ids"] = np.arange(volume.n_cells)
    volume_blocks, surfaces = process_meshes(volume, surface, region_key=region_key)

    # get basic info
    n_element_blocks = len(volume_blocks)
    n_side_sets = surfaces.n_blocks
    n_nodes = volume.n_points
    n_cells = volume.n_cells
    if block_names is None:
        block_names = [f"block_{i}" for i in range(n_element_blocks)]
    if side_set_names is None:
        side_set_names = [f"side_{i}" for i in range(n_side_sets)]

    logging.info(f"Element blocks: {n_element_blocks}, Side sets: {n_side_sets}, Nodes: {n_nodes}, Cells: {n_cells}")
    if n_element_blocks == 0:
        logging.warning("No element blocks to write. File may be empty or invalid.")
    if n_side_sets == 0:
        logging.warning("No side sets to write.")

    # Write exodus file
    with exodusii_file(filename, mode="w") as exof:
        exof.put_init(title="pyvista_mesh", num_dim=3, num_nodes=n_nodes, num_elem=n_cells,
                      num_elem_blk=n_element_blocks, num_side_sets=n_side_sets, num_node_sets=0)
        exof.put_coords(np.array(volume.points))
        logging.info("Initialized ExodusII file and wrote coordinates.")

        # Write element blocks
        logging.info("Saving element blocks...")
        counter = 1
        for key, item in volume_blocks.items():
            mesh = item["mesh"]
            n_block_cells = mesh.n_cells
            if n_block_cells == 0:
                logging.warning(f"Skipping empty block {key} with region {item['region']}")
                continue
            n_cell_nodes = mesh.get_cell(0).n_points
            exo_cell_type = vtk2exo[item["cell_type"]]
            if exo_cell_type is None:
                logging.warning(f"Unsupported cell type {item['cell_type'].name} in block {key}. Skipping.")
                continue
            logging.info(f"  - Block {counter}: {exo_cell_type} with {n_block_cells} cells, region {item['region']}")
            connectivity = mesh['orig_pts'][mesh.cells.reshape(-1, n_cell_nodes+1)[:, 1:]] + 1

            # Apply permutation if needed (e.g. VOXEL -> HEX)
            if item["cell_type"] in vtk2exo_permute:
                perm = vtk2exo_permute[item["cell_type"]]
                connectivity = connectivity[:, perm]
            exof.put_element_block(counter, elem_type=exo_cell_type, num_block_elems=n_block_cells, num_nodes_per_elem=n_cell_nodes)
            exof.put_element_conn(counter, connectivity)
            exof.put_element_block_name(counter, block_names[counter-1])
            counter += 1

        # Write node arrays
        if save_node_arrays:
            logging.info("Saving node arrays...")
            node_array_names, node_arrays = [], []
            for name in volume.array_names:
                array = volume[name]
                if len(array) == n_nodes:
                    node_array_names.append(name)
                    node_arrays.append(array)
                    logging.info(f"  - node array {name} shape =  {array.shape}")
            exof.put_node_variable_params(len(node_array_names))
            exof.put_node_variable_names(node_array_names)
            for i in range(len(node_array_names)):
                exof.put_node_variable_values(1, node_array_names[i], node_arrays[i])

        logging.info("Saving side sets...")
        # Write side sets
        for i, s in enumerate(surfaces):
            n_sides = s.n_cells
            if n_sides == 0:
                logging.warning(f"Skipping empty side set {i}")
                continue
            logging.info(f"  - Side set {i} with {n_sides} sides")
            exof.put_side_set_param(i+1, n_sides)
            exof.put_side_set_sides(i+1, s["cell_ids"]+1, s["face_ids"]+1)
            exof.put_side_set_name(i+1, side_set_names[i])

        exof.close()
        logging.info(f"{filename} saved successfully.")

    return None
