"""Write side sets on a tetrahedral mesh.

The ``surface`` argument holds the faces to write; each value of its region
array becomes one side set. Faces are matched to element sides by their
nodes, so any subset of the boundary works.
"""

import numpy as np
import pyvista as pv

import exovista


def main():
    # A 2 x 2 x 2 box of hexahedra, split into tetrahedra.
    tets = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid().triangulate()

    # The top (z = 2) and bottom (z = 0) faces only, as two named side sets.
    surface = tets.extract_surface()
    z = surface.cell_centers().points[:, 2]
    keep = np.flatnonzero((z > 1.9) | (z < 0.1))
    surface = surface.extract_cells(keep)
    surface.cell_data["region"] = np.where(z[keep] > 1.9, "top", "bottom")

    filename = "example_side_sets.exo"
    exovista.write_exo(filename, tets, surface=surface)

    side_sets = exovista.read_side_sets(filename)
    for name in side_sets.keys():
        faces = side_sets[name]
        print(f"{name}: {faces.n_cells} faces, element sides "
              f"{np.unique(faces.cell_data['exo_side'])}")

    # Node sets are written from 0-based point indices.
    bottom = np.flatnonzero(tets.points[:, 2] == 0)
    exovista.write_exo(filename, tets, surface=surface,
                       node_sets={"bottom_nodes": bottom})
    node_sets = exovista.read_node_sets(filename)
    print("node sets:", {name: len(nodes) for name, nodes in node_sets.items()})


if __name__ == "__main__":
    main()
