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


def export_hybrid():
    volume = pv.MultiBlock([
        pv.examples.download_letter_a(),
        pv.CylinderStructured(radius=np.linspace(1, 2, 5)).cast_to_unstructured_grid()
    ]).combine()
    surface = volume.extract_surface()

    exovista.write_exo("hybrid.exo", volume, surface)

    return None


def export_with_node_arrays():
    volume = pv.examples.download_letter_a()
    volume.points -= volume.center
    volume["tag"] = 1 * (volume.cell_centers().points[:, 0] > 0)

    # tool will export volume mesh node arrays to the exo file
    volume["x"] = volume.points[:, 0]
    volume["fx"] = volume.points[:, 0]**2

    surface = volume.extract_surface()
    surface["tag"] = 1 * (surface.cell_centers().points[:, 2] > 0)
    exovista.write_exo("node_array_test.exo", volume, surface, "tag")
    return None


if __name__ == '__main__':
    # export_tetra_mesh()
    # export_hex_mesh()
    # export_hybrid()
    export_with_node_arrays()