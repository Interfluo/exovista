"""Write a mesh that mixes hexahedra and tetrahedra.

An Exodus element block holds a single element type, so a region with
several cell types is split into one block per type. Given a name for the
region, the blocks are named ``<name>_<type>`` (here ``solid_hex8`` and
``solid_tetra4``).
"""

import numpy as np
import pyvista as pv

import exovista


def main():
    # Hexahedra (VTK voxels) on the left, the same grid split into
    # tetrahedra on the right.
    hexes = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    tets = hexes.translate((5.0, 0.0, 0.0)).triangulate()
    combined = hexes.merge(tets)
    print("cell types:", np.unique(combined.celltypes))

    # One region for everything...
    filename = "example_mixed_elements.exo"
    exovista.write_exo(filename, combined, block_names="solid")
    mesh = exovista.read_exo(filename)
    names, counts = np.unique(mesh.cell_data["exo_block_name"], return_counts=True)
    print("one region   :", dict(zip(names.tolist(), counts.tolist())))

    # ... or one region per part, with block ids 10 and 20.
    combined.cell_data["region"] = np.where(
        combined.celltypes == pv.CellType.TETRA, 20, 10)
    exovista.write_exo(filename, combined,
                       block_names={10: "hex_part", 20: "tet_part"})
    mesh = exovista.read_exo(filename)
    ids, counts = np.unique(mesh.cell_data["exo_block_id"], return_counts=True)
    print("two regions  :", dict(zip(ids.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
