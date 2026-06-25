
"""
Example of writing time-varying fields to an ExodusII file.

This demonstrates how to store a time history of results on a mesh:
    - a time-varying *node* field (temperature diffusing over time)
    - a time-varying *element* field (per-cell energy)

The resulting `.exo` file contains multiple time steps and can be opened in
ParaView (use the "time" controls to animate) or read back with ExoVista.
"""

import logging
import numpy as np
import pyvista as pv
import exovista

logging.basicConfig(level=logging.INFO)


def main():
    # Build a simple 3x3x3 hexahedral grid.
    grid = pv.ImageData(dimensions=(4, 4, 4)).cast_to_unstructured_grid()

    # Two element blocks split by x position.
    grid.cell_data["region"] = 1 * (grid.cell_centers().points[:, 0] > 1.5) + 1

    n_nodes = grid.n_points
    n_cells = grid.n_cells

    # Define the time history.
    times = np.linspace(0.0, 1.0, 11)  # 11 time steps from t=0 to t=1
    n_steps = len(times)

    # Time-varying node field: a Gaussian "pulse" of temperature that decays.
    # Shape must be (n_steps, n_nodes).
    x = grid.points[:, 0]
    temperature = np.empty((n_steps, n_nodes))
    for i, t in enumerate(times):
        temperature[i] = np.exp(-((x - 1.5) ** 2)) * np.cos(2 * np.pi * t)

    # Time-varying element field: energy ramping up linearly in time.
    # Shape must be (n_steps, n_cells), in the original cell order.
    base = np.arange(n_cells, dtype=float)
    energy = np.outer(times, base)  # (n_steps, n_cells)

    output_filename = "example_time_fields.exo"
    logging.info(f"Writing {n_steps} time steps to {output_filename}...")
    exovista.write_exo(
        output_filename,
        grid,
        times=times,
        node_fields={"temperature": temperature},
        element_fields={"energy": energy},
    )
    logging.info("Done.")

    # Read the file back and confirm the time history was stored.
    f = exovista.File(output_filename, mode="r")
    try:
        logging.info(f"num_times       = {f.num_times()}")
        logging.info(f"times           = {np.asarray(f.get_times())}")
        logging.info(f"node variables  = {list(f.get_node_variable_names())}")
        logging.info(f"elem variables  = {list(f.get_element_variable_names())}")
    finally:
        f.close()


if __name__ == "__main__":
    main()
