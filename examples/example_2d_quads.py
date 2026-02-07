
import pyvista as pv
import numpy as np
import exovista
import logging

logging.basicConfig(level=logging.INFO)

def main():
    """
    Example of creating and saving a 2D mesh with generic Quad elements.
    """
    # Create points for a simple square made of 4 quads
    # 0 -- 1 -- 2
    # |    |    |
    # 3 -- 4 -- 5
    # |    |    |
    # 6 -- 7 -- 8
    points = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0],
        [0, 2, 0], [1, 2, 0], [2, 2, 0]
    ], dtype=float)

    # 4 Quad cells
    # Cell 0: 0, 1, 4, 3
    # Cell 1: 1, 2, 5, 4
    # Cell 2: 3, 4, 7, 6
    # Cell 3: 4, 5, 8, 7
    cells = np.concatenate([
        [4, 0, 1, 4, 3],
        [4, 1, 2, 5, 4],
        [4, 3, 4, 7, 6],
        [4, 4, 5, 8, 7]
    ])
    cell_types = np.array([pv.CellType.QUAD] * 4)

    grid = pv.UnstructuredGrid(cells, cell_types, points)
    
    # Assign region data - let's make two materials
    # Cells 0 and 3 are material 1, Cells 1 and 2 are material 2
    grid.cell_data["region"] = np.array([1, 2, 2, 1])

    # Save to EXO
    output_filename = "example_2d_quads.exo"
    logging.info(f"Saving 2D Quad mesh to {output_filename}...")
    exovista.write_exo(output_filename, grid)
    logging.info("Done.")

if __name__ == "__main__":
    main()
