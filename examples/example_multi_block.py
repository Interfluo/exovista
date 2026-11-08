"""Write a hexahedral mesh split into element blocks by a region array.

Also shows the two other ways to define blocks: a ``pyvista.MultiBlock``
(one block per leaf, named after the leaf) and a string region array.
"""

import numpy as np
import pyvista as pv

import exovista


def main():
    # A 4 x 2 x 2 grid of hexahedra; left and right halves are two regions.
    grid = pv.ImageData(dimensions=(5, 3, 3)).cast_to_unstructured_grid()
    centers = grid.cell_centers().points
    grid.cell_data["region"] = np.where(centers[:, 0] > 2, 2, 1)
    grid.point_data["temperature"] = grid.points[:, 0] * 10

    # block_names: a list in sorted region order, or a dict {region: name}
    filename = "example_multi_block.exo"
    exovista.write_exo(filename, grid,
                       block_names=["left_material", "right_material"])
    mesh = exovista.read_exo(filename)
    print("region array :", sorted(set(mesh.cell_data["exo_block_name"])))

    # a MultiBlock: each leaf becomes a block named after its key
    left = grid.extract_cells(np.flatnonzero(grid.cell_data["region"] == 1))
    right = grid.extract_cells(np.flatnonzero(grid.cell_data["region"] == 2))
    exovista.write_exo(filename, pv.MultiBlock({"left": left, "right": right}))
    mesh = exovista.read_exo(filename)
    print("MultiBlock   :", sorted(set(mesh.cell_data["exo_block_name"])))

    # string region values name the blocks directly
    grid.cell_data["region"] = np.where(centers[:, 0] > 2, "aluminium", "copper")
    exovista.write_exo(filename, grid)
    mesh = exovista.read_exo(filename)
    print("string region:", sorted(set(mesh.cell_data["exo_block_name"])))


if __name__ == "__main__":
    main()
