import logging

import numpy as np
import pyvista as pv
from .file import exodusii_file

from scipy.interpolate import NearestNDInterpolator


vtk2exo = {
    pv.CellType.TRIANGLE: 'tri3',
    pv.CellType.QUAD: 'quad4',
    pv.CellType.TETRA: 'tet4',
    pv.CellType.HEXAHEDRON: 'hex8',
    pv.CellType.WEDGE: 'wedge6',
    pv.CellType.PYRAMID: 'pyramid5',
    # add more as needed
}


def cell_2_face_center(cell: pv.Cell):
    """
    Compute the center points of all faces in a given PyVista cell.

    Parameters
    ----------
    cell : pv.Cell
        The PyVista cell object.

    Returns
    -------
    list[np.ndarray]
        List of face center coordinates as NumPy arrays.
    """
    face_centers = []
    for i in range(cell.n_faces):
        face_centers.append(cell.get_face(i).center)
    return face_centers


def get_face_center_info(mesh: pv.UnstructuredGrid, cell_ids: list[int], block_id: int|str) -> pv.PolyData:
    """
    Generate a PolyData object containing face center points for all cells in the mesh,
    annotated with cell IDs, face IDs, and block IDs.

    This is used for mapping surface faces to volume faces via interpolation.

    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        The input volume mesh.
    cell_ids : list[int]
        Global cell IDs for the cells in the mesh.
    block_id : int | str
        The block ID (region) for this mesh block.

    Returns
    -------
    pv.PolyData
        Point cloud of face centers with associated data arrays.
    """
    points, face_cell_ids, face_ids, block_ids = [], [], [], []

    for cell_index in range(mesh.n_cells):
        for fid, fc in enumerate(cell_2_face_center(mesh.get_cell(cell_index))):
            points.append(fc)
            face_cell_ids.append(cell_ids[cell_index])
            face_ids.append(fid)
            block_ids.append(block_id)

    points = pv.PolyData(np.array(points))
    points["cell_ids"] = np.array(face_cell_ids)
    points["face_ids"] = np.array(face_ids)
    points["block_ids"] = np.array(block_ids)
    return points


def _process_meshes(volume: pv.UnstructuredGrid, surface: pv.PolyData, region_key: str = "region") -> tuple[dict, pv.MultiBlock]:
    """
    Process volume and surface meshes to prepare for ExodusII export.

    Splits the volume mesh into blocks by cell type and region. Maps surface faces
    to corresponding volume faces using nearest-neighbor interpolation.

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

    # Check and fallback for surface regions
    if region_key not in surface.cell_data:
        logging.warning(f"Region key '{region_key}' not found in surface cell data. Setting all regions to 1.")
        surface.cell_data[region_key] = np.ones(surface.n_cells, dtype=int)

    volume_regions = np.unique(volume[region_key])
    surface_regions = np.unique(surface[region_key])
    logging.info(f"Volume regions found: {volume_regions}")
    logging.info(f"Surface regions found: {surface_regions}")

    # Split volume blocks by cell_type and region
    volume_blocks = {}
    cell_id, counter = 0, 0
    for cell_type in pv.CellType:
        volume_cell_type = volume.extract_cells_by_type(cell_type)
        if volume_cell_type.n_cells > 0:
            logging.info(f"Processing cell type: {cell_type.name} with {volume_cell_type.n_cells} cells")
            for m in volume_cell_type.split_values(scalars=region_key):
                region_value = m[region_key][0] if region_key in m.cell_data else 1
                logging.info(f"  - Sub-block with region {region_value} and {m.n_cells} cells")
                volume_blocks[counter] = {"region": region_value, "cell_type": cell_type, "mesh": m, "cell_ids": [i+cell_id for i in range(m.n_cells)]}
                cell_id += m.n_cells
                counter += 1

    if not volume_blocks:
        logging.warning("No volume blocks found after processing. Export may fail.")

    # Compute face centers for interpolation
    face_points = pv.MultiBlock([get_face_center_info(item["mesh"], item["cell_ids"], item["region"]) for key, item in volume_blocks.items()]).combine()
    logging.info(f"Generated {face_points.n_points} face center points for interpolation.")

    # Map surface to volume faces
    for s in ["block_ids", "cell_ids", "face_ids"]:
        surface[s] = NearestNDInterpolator(face_points.points, face_points[s])(surface.cell_centers().points)
    logging.info("Surface face mapping completed.")

    return volume_blocks, surface.split_values(scalars=region_key)


def process_meshes(volume: pv.UnstructuredGrid, surface: pv.PolyData, region_key: str = "region") -> tuple[dict, pv.MultiBlock]:
    """
    Process volume and surface meshes to prepare for ExodusII export.

    Splits the volume mesh into blocks by cell type and region. Maps surface faces
    to corresponding volume faces using nearest-neighbor interpolation.

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
        # Check and fallback for surface regions
        if region_key not in surface.cell_data:
            logging.warning(f"Region key '{region_key}' not found in surface cell data. Setting all regions to 1.")
            surface.cell_data[region_key] = np.ones(surface.n_cells, dtype=int)

        surface_regions = np.unique(surface[region_key])
        logging.info(f"Surface regions found: {surface_regions}")
    else:
        surface_regions = []
        logging.info("No surface provided. Skipping surface processing.")

    # Split volume blocks by cell_type and region
    volume_blocks = {}
    cell_id, counter = 0, 0
    for cell_type in pv.CellType:
        volume_cell_type = volume.extract_cells_by_type(cell_type)
        if volume_cell_type.n_cells > 0:
            logging.info(f"Processing cell type: {cell_type.name} with {volume_cell_type.n_cells} cells")
            for m in volume_cell_type.split_values(scalars=region_key):
                region_value = m[region_key][0] if region_key in m.cell_data else 1
                logging.info(f"  - Sub-block with region {region_value} and {m.n_cells} cells")
                volume_blocks[counter] = {"region": region_value, "cell_type": cell_type, "mesh": m, "cell_ids": [i+cell_id for i in range(m.n_cells)]}
                cell_id += m.n_cells
                counter += 1

    if not volume_blocks:
        logging.warning("No volume blocks found after processing. Export may fail.")

    if surface is not None:
        # Compute face centers for interpolation
        face_points = pv.MultiBlock([get_face_center_info(item["mesh"], item["cell_ids"], item["region"]) for key, item in volume_blocks.items()]).combine()
        logging.info(f"Generated {face_points.n_points} face center points for interpolation.")

        # Map surface to volume faces
        for s in ["block_ids", "cell_ids", "face_ids"]:
            surface[s] = NearestNDInterpolator(face_points.points, face_points[s])(surface.cell_centers().points)
        logging.info("Surface face mapping completed.")

        surfaces = surface.split_values(scalars=region_key)
    else:
        surfaces = pv.MultiBlock()

    return volume_blocks, surfaces


def write_exo(filename, volume: pv.UnstructuredGrid, surface: pv.PolyData = None, region_key: str = "region"):
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

    Returns
    -------
    None
    """

    logging.info(f"Writing ExodusII file: {filename}")
    logging.info(f"Using region key: '{region_key}'")

    f = open(filename, "w+")
    f.close()

    # Process meshes
    volume_blocks, surfaces = process_meshes(volume, surface, region_key=region_key)

    # get basic info
    n_element_blocks = len(volume_blocks)
    n_side_sets = surfaces.n_blocks
    n_nodes = volume.n_points
    n_cells = volume.n_cells
    logging.info(f"Element blocks: {n_element_blocks}, Side sets: {n_side_sets}, Nodes: {n_nodes}, Cells: {n_cells}")

    if n_element_blocks == 0:
        logging.warning("No element blocks to write. File may be empty or invalid.")
    if n_side_sets == 0:
        logging.warning("No side sets to write.")

    # Write exodus file
    with exodusii_file(filename, mode="w") as exof:
        exof.put_init(title="pyvista_mesh", num_dim=3, num_nodes=n_nodes, num_elem=n_cells,
                      num_elem_blk=n_element_blocks, num_side_sets=n_side_sets, num_node_sets=0)
        exof.put_coords(volume.points)
        logging.info("Initialized ExodusII file and wrote coordinates.")

        logging.info("Saving element blocks...")
        # Write element blocks
        counter = 0
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
            if 'vtkOriginalPointIds' not in mesh.point_data:
                logging.warning(f"'vtkOriginalPointIds' missing in block {key}. This may cause connectivity issues.")
            # print(f"{item['cell_type'].name} -> {exo_cell_type}")
            connectivity = mesh['vtkOriginalPointIds'][mesh.cells.reshape(-1, n_cell_nodes+1)[:, 1:]] + 1
            exof.put_element_block(counter, elem_type=exo_cell_type, num_block_elems=n_block_cells, num_nodes_per_elem=n_cell_nodes)
            exof.put_element_conn(counter, connectivity)
            counter += 1

        logging.info("Saving side sets...")
        # Write side sets
        for i, s in enumerate(surfaces):
            n_sides = s.n_cells
            if n_sides == 0:
                logging.warning(f"Skipping empty side set {i}")
                continue
            logging.info(f"  - Side set {i} with {n_sides} sides")
            exof.put_side_set_param(i, n_sides)
            exof.put_side_set_sides(i, s["cell_ids"]+1, s["face_ids"]+1)

        exof.close()
        logging.info(f"{filename} saved successfully.")

    return None

