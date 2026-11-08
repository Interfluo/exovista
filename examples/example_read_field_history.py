"""Read whole time histories back out of an ExodusII file.

The read counterparts of ``write_exo``'s ``node_fields``, ``element_fields``
and ``global_fields`` return every time step without building the mesh:

* ``read_node_fields``    -> {name: (n_steps, n_points)}
* ``read_element_fields`` -> {name: (n_steps, n_cells)}, columns in the
  original cell order
* ``read_global_fields``  -> {name: (n_steps,)}

The example also selects variables on ``read_exo`` and shows its 1-based
``time_step`` (negative values count from the end, 0 is an error).
"""

import numpy as np
import pyvista as pv

import exovista


def main():
    # A small hex grid split into two element blocks along x.
    grid = pv.ImageData(dimensions=(4, 3, 3)).cast_to_unstructured_grid()
    grid.cell_data["region"] = np.where(grid.cell_centers().points[:, 0] > 1.5, 2, 1)

    times = np.linspace(0.0, 1.0, 5)
    x = grid.points[:, 0]
    temperature = np.array([np.cos(2 * np.pi * t) * x for t in times])
    energy = np.outer(times, np.arange(grid.n_cells, dtype=float))
    flux = np.stack([energy, -energy, 2 * energy], axis=-1)  # a vector field

    filename = "example_read_field_history.exo"
    exovista.write_exo(
        filename, grid, times=times,
        node_fields={"temperature": temperature},
        element_fields={"energy": energy, "flux": flux},
        global_fields={"total_energy": energy.sum(axis=1)},
    )

    # Full histories, no mesh.
    node_fields = exovista.read_node_fields(filename)
    elem_fields = exovista.read_element_fields(filename, combine_components=True)
    global_fields = exovista.read_global_fields(filename)
    print("temperature:", node_fields["temperature"].shape,
          np.array_equal(node_fields["temperature"], temperature))
    print("energy     :", elem_fields["energy"].shape,
          np.array_equal(elem_fields["energy"], energy))
    print("flux       :", elem_fields["flux"].shape,
          np.array_equal(elem_fields["flux"], flux))
    print("total      :", global_fields["total_energy"])
    # Without combine_components the components are separate variables.
    print("components :", sorted(exovista.read_element_fields(filename)))

    # Select variables when reading the mesh.
    mesh = exovista.read_exo(filename, read_node_variables="temperature",
                             read_element_variables=False)
    print("point arrays:", list(mesh.point_data.keys()))
    print("cell arrays :", list(mesh.cell_data.keys()))

    # time_step is 1-based; negative values count from the end.
    first = exovista.read_exo(filename, time_step=1)
    last = exovista.read_exo(filename, time_step=-1)
    print("step 1 is t =", first.field_data["time"][0],
          np.array_equal(first.point_data["temperature"], temperature[0]))
    print("step -1 is t =", last.field_data["time"][0],
          np.array_equal(last.point_data["temperature"], temperature[-1]))
    try:
        exovista.read_exo(filename, time_step=0)
    except IndexError as exc:
        print("time_step=0 raises IndexError:", exc)


if __name__ == "__main__":
    main()
