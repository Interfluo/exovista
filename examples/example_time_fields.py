"""Write time-varying fields to an ExodusII file.

A time history is stored as:

* a time-varying *node* field (a temperature pulse), shape (n_steps, n_points);
* a time-varying *element* field (per-cell energy), shape (n_steps, n_cells),
  columns in the original cell order;
* a *global* field (one value per step).

Open the file in ParaView and use the time controls to animate it.
"""

import numpy as np
import pyvista as pv

import exovista


def main():
    grid = pv.ImageData(dimensions=(4, 4, 4)).cast_to_unstructured_grid()
    # two element blocks split by x position
    grid.cell_data["region"] = np.where(grid.cell_centers().points[:, 0] > 1.5, 2, 1)

    times = np.linspace(0.0, 1.0, 11)  # 11 steps from t = 0 to t = 1
    x = grid.points[:, 0]
    temperature = np.exp(-((x - 1.5) ** 2)) * np.cos(2 * np.pi * times[:, None])
    energy = np.outer(times, np.arange(grid.n_cells, dtype=float))

    filename = "example_time_fields.exo"
    exovista.write_exo(
        filename, grid, times=times,
        node_fields={"temperature": temperature},
        element_fields={"energy": energy},
        global_fields={"max_temperature": temperature.max(axis=1)},
    )
    print(f"wrote {len(times)} steps to {filename}")

    mesh = exovista.read_exo(filename, time_step=6)
    print("times        :", mesh.field_data["times"])
    print("step 6 is t  =", mesh.field_data["time"][0])
    print("point arrays :", list(mesh.point_data.keys()))
    print("cell arrays  :", list(mesh.cell_data.keys()))
    print("global value :", mesh.field_data["max_temperature"][0])


if __name__ == "__main__":
    main()
