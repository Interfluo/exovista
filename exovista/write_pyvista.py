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
    face_centers = []
    for i in range(cell.n_faces):
        face_centers.append(cell.get_face(i).center)
    return face_centers


def get_face_center_info(mesh: pv.UnstructuredGrid, cell_ids: list[int], block_id: int|str) -> pv.PolyData:
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


def process_meshes(volume: pv.UnstructuredGrid, surface: pv.PolyData) -> tuple[dict, pv.MultiBlock]:
    volume_regions = np.unique(volume["region"])
    surface_regions = np.unique(surface["region"])
    # print(f"{volume_regions = }, {surface_regions = }")

    # need to split the volume blocks further down by cell_type (each exo element block
    # can only contain one cell type). We also need to keep track of the cell orders
    # because the side sets will reference the order of elements as added to the exo file.
    volume_blocks = {}
    cell_id, counter = 0, 0
    for cell_type in pv.CellType:
        volume_cell_type = volume.extract_cells_by_type(cell_type)
        if volume_cell_type.n_cells > 0:
            for m in volume_cell_type.split_values(scalars="region"):
                volume_blocks[counter] = {"region": m['region'][0], "cell_type": cell_type, "mesh": m, "cell_ids": [i+cell_id for i in range(m.n_cells)]}
                cell_id += m.n_cells
                counter += 1

    # we want to find which element faces each side set (surface) face corresponds to,
    # we need to know the element number (cell_id) and face number
    face_points = pv.MultiBlock([get_face_center_info(item["mesh"], item["cell_ids"], item["region"]) for key, item in volume_blocks.items()]).combine()
    for s in ["block_ids", "cell_ids", "face_ids"]:
        surface[s] = NearestNDInterpolator(face_points.points, face_points[s])(surface.cell_centers().points)

    return volume_blocks, surface.split_values(scalars="region")


def write_exo(filename, volume, surface):
    logging.info(f"writing exo file {filename}")
    f = open(filename, "w+")
    f.close()
    volume_blocks, surfaces = process_meshes(volume, surface)

    # get basic info
    n_element_blocks = len(volume_blocks)
    n_side_sets = surfaces.n_blocks
    n_nodes = volume.n_points
    n_cells = volume.n_cells
    logging.info(f"{n_element_blocks = }, {n_side_sets = }, {n_nodes = }, {n_cells = }")

    # write exodus file
    with exodusii_file(filename, mode="w") as exof:
        exof.put_init(title="pyvista_mesh", num_dim=3, num_nodes=n_nodes, num_elem=n_cells,
                      num_elem_blk=n_element_blocks, num_side_sets=n_side_sets, num_node_sets=0)
        exof.put_coords(volume.points)

        logging.info("saving element blocks")
        # write element blocks
        counter = 0
        for key, item in volume_blocks.items():
            mesh = item["mesh"]
            n_block_cells = mesh.n_cells
            n_cell_nodes = mesh.get_cell(0).n_points
            exo_cell_type = vtk2exo[item["cell_type"]]
            # print(f"{item['cell_type'].name} -> {exo_cell_type}")
            connectivity = mesh['vtkOriginalPointIds'][mesh.cells.reshape(-1, n_cell_nodes+1)[:, 1:]] + 1
            exof.put_element_block(counter, elem_type=exo_cell_type, num_block_elems=n_block_cells, num_nodes_per_elem=n_cell_nodes)
            exof.put_element_conn(counter, connectivity)
            counter += 1

        logging.info("saving side sets")
        # write side sets
        for i, s in enumerate(surfaces):
            exof.put_side_set_param(i, s.n_cells)
            exof.put_side_set_sides(i, s["cell_ids"]+1, s["face_ids"]+1)

        exof.close()
        logging.info(f"{filename} saved successfully")

    return None


