"""Save PyVista meshes to ExodusII with ``mesh.save``.

Installing exovista registers a PyVista writer plugin for the ``.exo``,
``.e``, ``.ex2``, ``.exii`` and ``.gen`` extensions:
``mesh.save("model.exo", **kwargs)`` is ``exovista.write_exo("model.exo",
mesh, **kwargs)``. (PyVista's own ``compression`` option is not exovista's;
use ``compression_level``.)
"""

import numpy as np
import pyvista as pv

import exovista


def main():
    # Any PyVista dataset: here a tetrahedral ball.
    ball = pv.Sphere(theta_resolution=12, phi_resolution=12).delaunay_3d()
    ball.save("example_pyvista_save.exo")
    print("plain save   :", exovista.read_exo("example_pyvista_save.exo"))

    # Keyword arguments go to write_exo: two element blocks (north/south),
    # the outer surface as one side set and a time-varying node field.
    halves = ball.copy()
    halves.cell_data["region"] = np.where(halves.cell_centers().points[:, 2] > 0, 1, 2)
    surface = halves.extract_surface(algorithm=None)
    del surface.cell_data["region"]  # (copied from the volume) one side set
    times = np.linspace(0.0, 1.0, 4)
    height = np.outer(times, halves.points[:, 2])
    halves.save(
        "example_pyvista_save.exo",
        surface=surface,
        block_names={1: "north", 2: "south"},
        side_set_names="skin",
        times=times,
        node_fields={"height": height},
        compression_level=4,
    )
    mesh, side_sets = exovista.read_exo("example_pyvista_save.exo",
                                        read_side_sets=True)
    print("blocks       :", np.unique(mesh.cell_data["exo_block_name"]))
    print("side sets    :", side_sets.keys())
    print("times        :", mesh.field_data["times"])

    # A MultiBlock saves too: one element block per leaf, named after it.
    box = pv.ImageData(dimensions=(3, 3, 3), origin=(2.0, 0.0, 0.0))
    parts = pv.MultiBlock({"ball": ball, "box": box.cast_to_unstructured_grid()})
    parts.save("example_pyvista_save_parts.e")
    mesh = exovista.read_exo("example_pyvista_save_parts.e")
    print("MultiBlock   :", np.unique(mesh.cell_data["exo_block_name"]))

    # VTK's reader (pv.read) opens the same files.
    blocks = pv.read("example_pyvista_save_parts.e")["Element Blocks"]
    print("pv.read      :", blocks.keys())


if __name__ == "__main__":
    main()
