
import pyvista as pv
import numpy as np
import exovista
import logging

logging.basicConfig(level=logging.INFO)


def main():
    """
    Round-trip example: write a PyVista mesh to ExodusII, then read it back.

    Demonstrates exovista.read_exo, the inverse of exovista.write_exo. The
    reconstructed UnstructuredGrid carries the originating Exodus block ids,
    any node/element variables sampled at the requested time step, and the time
    history in field_data.
    """
    # A small hex grid split into two regions along x.
    grid = pv.ImageData(dimensions=(4, 3, 3)).cast_to_unstructured_grid()
    centers = grid.cell_centers().points
    grid.cell_data["region"] = (centers[:, 0] > 1.5).astype(int)
    grid.point_data["height"] = grid.points[:, 2]

    # A short transient element field.
    times = np.linspace(0.0, 1.0, 3)
    energy = np.outer(times, np.arange(grid.n_cells, dtype=float))

    filename = "example_read_exo.exo"
    logging.info(f"Writing {filename}...")
    exovista.write_exo(
        filename, grid, region_key="region",
        times=times, element_fields={"energy": energy},
    )

    logging.info(f"Reading {filename} back into PyVista...")
    mesh = exovista.read_exo(filename)  # defaults to the last time step

    logging.info(f"  points: {mesh.n_points}, cells: {mesh.n_cells}")
    logging.info(f"  element blocks: {np.unique(mesh.cell_data['exo_block_id'])}")
    logging.info(f"  point arrays: {list(mesh.point_data.keys())}")
    logging.info(f"  cell arrays:  {list(mesh.cell_data.keys())}")
    logging.info(f"  times: {mesh.field_data['times']}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
