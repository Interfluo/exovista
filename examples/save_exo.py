#!/usr/bin/env python
"""
Example script demonstrating various ExoVista export capabilities.

This script provides examples of exporting different mesh types to ExodusII format:
- Tetrahedral mesh with element blocks and side sets
- Hexahedral mesh from structured grids
- Hybrid mesh combining different element types
- Mesh with node arrays (point data)

Usage:
    python save_exo.py --tetra    # Export tetrahedral mesh
    python save_exo.py --hex      # Export hexahedral mesh
    python save_exo.py --hybrid   # Export hybrid mesh
    python save_exo.py --all      # Export all examples
"""

import argparse
import logging
import exovista
import numpy as np
import pyvista as pv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def export_tetra_mesh() -> None:
    """
    Export a tetrahedral mesh with element blocks and side sets.
    
    Uses the PyVista 'letter_a' example mesh. Assigns regions based on
    x-coordinate for element blocks and z-coordinate for side sets.
    """
    logging.info("Exporting tetrahedral mesh...")
    
    volume = pv.examples.download_letter_a()
    volume.points -= volume.center
    volume["tag"] = 1 * (volume.cell_centers().points[:, 0] > 0)

    surface = volume.extract_surface()
    surface["tag"] = 1 * (surface.cell_centers().points[:, 2] > 0)

    exovista.write_exo("tetra.exo", volume, surface, "tag")
    logging.info("Saved tetra.exo")


def export_hex_mesh() -> None:
    """
    Export a hexahedral mesh from a structured cylinder grid.
    
    Creates a structured cylindrical mesh, assigns regions for element
    blocks and side sets, and exports with and without side sets.
    """
    logging.info("Exporting hexahedral mesh...")
    
    volume = pv.CylinderStructured(
        radius=np.linspace(1, 2, 5)
    ).cast_to_unstructured_grid()
    surface = volume.extract_surface()

    # Assign regions based on y and z coordinates
    volume["region"] = (
        1 * (volume.cell_centers().points[:, 1] > 0) +
        2 * (volume.cell_centers().points[:, 2] > 0)
    )
    surface["region"] = 1 * (surface.cell_centers().points[:, 0] > 0)

    exovista.write_exo("hex.exo", volume, surface)
    exovista.write_exo("hex_no_sides.exo", volume, None)
    logging.info("Saved hex.exo and hex_no_sides.exo")


def export_hybrid() -> None:
    """
    Export a hybrid mesh combining tetrahedral and hexahedral elements.
    
    Combines the letter_a tetrahedral mesh with a structured cylinder
    mesh to demonstrate mixed element type support.
    """
    logging.info("Exporting hybrid mesh...")
    
    volume = pv.MultiBlock([
        pv.examples.download_letter_a(),
        pv.CylinderStructured(
            radius=np.linspace(1, 2, 5)
        ).cast_to_unstructured_grid()
    ]).combine()
    surface = volume.extract_surface()

    exovista.write_exo("hybrid.exo", volume, surface)
    logging.info("Saved hybrid.exo")


def export_with_node_arrays() -> None:
    """
    Export a mesh with node arrays (point data).
    
    Creates a simple mesh and adds a sinusoidal function as point data,
    demonstrating how to save node arrays to the Exodus file.
    """
    logging.info("Exporting mesh with node arrays...")
    
    # Use a simple structured grid instead of tetgen for portability
    volume = pv.CylinderStructured(
        radius=np.linspace(1, 2, 10),
        theta_resolution=32,
        z_resolution=10
    ).cast_to_unstructured_grid()
    surface = volume.extract_surface()

    # Assign regions
    volume["region"] = (
        1 * (volume.cell_centers().points[:, 1] > 0) +
        2 * (volume.cell_centers().points[:, 2] > 0)
    )
    surface["region"] = 1 * (surface.cell_centers().points[:, 0] > 0)

    # Add node array
    volume["fx"] = np.sin(4 * np.pi * volume.points[:, 0])

    exovista.write_exo(
        "node_array_test.exo",
        volume,
        surface,
        save_node_arrays=True
    )
    logging.info("Saved node_array_test.exo")


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="ExoVista example script for exporting meshes to ExodusII format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--tetra", action="store_true",
        help="Export tetrahedral mesh example"
    )
    parser.add_argument(
        "--hex", action="store_true",
        help="Export hexahedral mesh example"
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Export hybrid mesh example"
    )
    parser.add_argument(
        "--node-arrays", action="store_true",
        help="Export mesh with node arrays example"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Export all examples"
    )

    args = parser.parse_args()

    # Default to node-arrays if no args provided (backward compatible)
    if not any([args.tetra, args.hex, args.hybrid, getattr(args, 'node_arrays', False), args.all]):
        setattr(args, 'node_arrays', True)

    if args.all or args.tetra:
        export_tetra_mesh()
    if args.all or args.hex:
        export_hex_mesh()
    if args.all or args.hybrid:
        export_hybrid()
    if args.all or args.node_arrays:
        export_with_node_arrays()


if __name__ == '__main__':
    main()