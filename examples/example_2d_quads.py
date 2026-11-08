"""Write a 2D quad mesh with two materials, then read it back.

A mesh whose cells are all 2D and whose points share one z coordinate is
written as a planar (``num_dim=2``) Exodus file: QUAD4 elements with x/y
coordinates only.
"""

import numpy as np
import pyvista as pv

import exovista


def main():
    # A 2 x 2 square of quads:
    # 6 -- 7 -- 8
    # |    |    |
    # 3 -- 4 -- 5
    # |    |    |
    # 0 -- 1 -- 2
    points = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0],
        [0, 2, 0], [1, 2, 0], [2, 2, 0],
    ], dtype=float)
    quads = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7],
    ])
    grid = pv.UnstructuredGrid({pv.CellType.QUAD: quads}, points)

    # Two materials in a checkerboard: the region values become block ids.
    grid.cell_data["region"] = np.array([1, 2, 2, 1])
    grid.point_data["height"] = grid.points[:, 0] + grid.points[:, 1]

    filename = "example_2d_quads.exo"
    exovista.write_exo(filename, grid, block_names={1: "copper", 2: "steel"})
    print(f"wrote {filename}")

    mesh = exovista.read_exo(filename)
    print("num_dim       :", int(mesh.field_data["num_dim"][0]))
    print("block ids     :", mesh.cell_data["exo_block_id"])
    print("block names   :", np.asarray(mesh.cell_data["exo_block_name"]).tolist())
    same = np.array_equal(mesh.cell_connectivity, grid.cell_connectivity)
    print("cells in input order:", same)


if __name__ == "__main__":
    main()
