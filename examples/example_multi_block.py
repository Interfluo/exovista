
import pyvista as pv
import numpy as np
import exovista
import logging

logging.basicConfig(level=logging.INFO)

def main():
    """
    Example of saving a Multi-Block mesh defined by regions.
    Also demonstrates saving Hex elements.
    """
    # Create two cubes side-by-side
    # We can create a larger grid and split it via regions
    # 4x2x2 grid of cells
    # Use ImageData cast to UnstructuredGrid for Hex cells
    grid = pv.ImageData(dimensions=(5, 3, 3)).cast_to_unstructured_grid()
    
    # Define regions based on x-coordinate
    centers = grid.cell_centers().points
    regions = np.ones(grid.n_cells, dtype=int)
    
    # Left half is region 1, Right half is region 2
    # x goes from 0 to 4
    regions[centers[:, 0] > 2] = 2
    
    grid.cell_data["region"] = regions
    
    # Add some node data
    grid.point_data["temperature"] = grid.points[:, 0] * 10
    
    # Save to EXO
    output_filename = "example_multi_block_v2.exo"
    logging.info(f"Saving Multi-Block Hex mesh to {output_filename}...")
    # block_names argument is optional, currently write_exo generates names like block_0, block_1
    # Check if we can pass names?
    # Yes: block_names
    exovista.write_exo(output_filename, grid, block_names=["left_material", "right_material"])
    logging.info("Done.")

if __name__ == "__main__":
    main()
