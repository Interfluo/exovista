
"""
Example of reading time-history fields back out of an ExodusII file.

This demonstrates the lightweight read counterparts to ``write_exo``'s
``node_fields`` / ``element_fields`` arguments:

    - ``read_node_fields``    -> dict of {name: (num_times, num_nodes)}
    - ``read_element_fields`` -> dict of {name: (num_times, num_elems)}

These return the *full* time history of each variable without reconstructing
the mesh, so they are the cheap way to pull a transient result out of a file.
It also shows selecting a subset of variables on ``read_exo`` and the 1-based
``time_step`` indexing rules.
"""

import logging
import numpy as np
import pyvista as pv
import exovista

logging.basicConfig(level=logging.INFO)


def main():
    # A small hex grid split into two element blocks along x.
    grid = pv.ImageData(dimensions=(4, 3, 3)).cast_to_unstructured_grid()
    grid.cell_data["region"] = (
        grid.cell_centers().points[:, 0] > 1.5
    ).astype(int)

    # A transient node field and a transient element field.
    times = np.linspace(0.0, 1.0, 5)  # 5 steps: t = 0, 0.25, 0.5, 0.75, 1.0
    x = grid.points[:, 0]
    temperature = np.array([np.cos(2 * np.pi * t) * x for t in times])
    energy = np.outer(times, np.arange(grid.n_cells, dtype=float))

    filename = "example_read_field_history.exo"
    logging.info(f"Writing {len(times)} time steps to {filename}...")
    exovista.write_exo(
        filename, grid,
        times=times,
        node_fields={"temperature": temperature},
        element_fields={"energy": energy},
    )

    # --- Full time history, no mesh reconstruction ---------------------------
    node_fields = exovista.read_node_fields(filename)
    elem_fields = exovista.read_element_fields(filename)
    temp_hist = node_fields["temperature"]   # (num_times, num_nodes)
    energy_hist = elem_fields["energy"]       # (num_times, num_elems)
    logging.info(f"temperature history shape: {temp_hist.shape}")
    logging.info(f"energy history shape:      {energy_hist.shape}")
    logging.info(f"round-trip matches write: "
                 f"{np.allclose(temp_hist, temperature)}")

    # Read just one named field (skips reading the rest).
    only_energy = exovista.read_element_fields(filename, names="energy")
    logging.info(f"selected element fields: {list(only_energy)}")

    # --- Selecting variables when reading the mesh itself --------------------
    mesh = exovista.read_exo(
        filename,
        read_node_variables="temperature",   # name or list of names
        read_element_variables=False,         # skip element variables
    )
    logging.info(f"mesh point arrays: {list(mesh.point_data.keys())}")
    logging.info(f"mesh cell arrays:  {list(mesh.cell_data.keys())}")

    # --- time_step is 1-based; negatives index from the end ------------------
    f = exovista.File(filename, mode="r")
    try:
        first = np.asarray(f.get_node_variable_values("temperature", time_step=1))
        last = np.asarray(f.get_node_variable_values("temperature", time_step=-1))
        logging.info(f"first step matches t=0 row: "
                     f"{np.allclose(first, temperature[0])}")
        logging.info(f"last step matches final row: "
                     f"{np.allclose(last, temperature[-1])}")
        # time_step=0 is not a valid 1-based step and raises.
        try:
            f.get_node_variable_values("temperature", time_step=0)
        except IndexError as exc:
            logging.info(f"time_step=0 correctly raised: {exc}")
    finally:
        f.close()

    logging.info("Done.")


if __name__ == "__main__":
    main()
