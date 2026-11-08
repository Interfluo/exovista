"""Round trip: write a PyVista mesh to ExodusII, then read it back.

``read_exo`` returns an UnstructuredGrid with float64 points, the cells in
the order they were written, the variables at one time step, the block id
and name of every cell, and (on request) the side sets as PolyData.
"""

import numpy as np
import pyvista as pv

import exovista


def main():
    # A small hex grid split into two regions along x.
    grid = pv.ImageData(dimensions=(4, 3, 3)).cast_to_unstructured_grid()
    centers = grid.cell_centers().points
    grid.cell_data["region"] = np.where(centers[:, 0] > 1.5, 2, 1)
    grid.point_data["height"] = grid.points[:, 2]
    grid.point_data["velocity"] = np.column_stack(
        [grid.points[:, 1], -grid.points[:, 0], np.zeros(grid.n_points)])

    # The boundary faces, split into two side sets (top vs. the rest).
    surface = grid.extract_surface(algorithm=None)
    top = surface.cell_centers().points[:, 2] > 1.5
    surface.cell_data["region"] = np.where(top, 1, 2)

    # A short transient element field.
    times = np.linspace(0.0, 1.0, 3)
    energy = np.outer(times, np.arange(grid.n_cells, dtype=float))

    filename = "example_read_exo.exo"
    exovista.write_exo(filename, grid, surface, times=times,
                       side_set_names={1: "top", 2: "other"},
                       element_fields={"energy": energy})

    # The last time step by default; time_step is 1-based (-1 = last).
    mesh, side_sets = exovista.read_exo(filename, read_side_sets=True)
    print("points, cells:", mesh.n_points, mesh.n_cells, mesh.points.dtype)
    print("blocks       :", np.unique(mesh.cell_data["exo_block_id"]))
    print("point arrays :", list(mesh.point_data.keys()))  # velocity is (n, 3)
    print("cell arrays  :", list(mesh.cell_data.keys()))
    print("times        :", mesh.field_data["times"])
    print("energy == input at t=1:", np.array_equal(mesh.cell_data["energy"],
                                                    energy[-1]))
    for name in side_sets.keys():
        faces = side_sets[name]
        print(f"side set {name!r}: {faces.n_cells} faces, cells "
              f"{np.unique(faces.cell_data['cell_id'])}")


if __name__ == "__main__":
    main()
