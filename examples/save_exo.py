import logging
import exovista
import numpy as np
import pyvista as pv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def export_tetra_mesh():
    volume = pv.examples.download_letter_a()
    volume.points -= volume.center
    volume["tag"] = 1 * (volume.cell_centers().points[:, 0] > 0)
    volume.plot(show_edges=True, categories=True, parallel_projection=True, text="element blocks")

    surface = volume.extract_surface()
    surface["tag"] = 1 * (surface.cell_centers().points[:, 2] > 0)
    surface.plot(show_edges=True, categories=True, parallel_projection=True, text="side sets")

    exovista.write_exo("tetra.exo", volume, surface, "tag")
    return None


def export_hex_mesh():
    volume = pv.CylinderStructured(radius=np.linspace(1, 2, 5)).cast_to_unstructured_grid()
    surface = volume.extract_surface()

    volume["region"] = 1*(volume.cell_centers().points[:, 1] > 0) + 2*(volume.cell_centers().points[:, 2] > 0)
    volume.plot(show_edges=True, categories=True, parallel_projection=True, text="element blocks")


    surface["region"] = 1*(surface.cell_centers().points[:, 0] > 0)
    surface.plot(show_edges=True, categories=True, parallel_projection=True, text="side sets")

    exovista.write_exo("hex.exo", volume, surface)
    exovista.write_exo("hex_no_sides.exo", volume, None)
    return None


if __name__ == '__main__':
    export_tetra_mesh()
    export_hex_mesh()