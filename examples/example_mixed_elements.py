
import pyvista as pv
import numpy as np
import exovista
import logging

logging.basicConfig(level=logging.INFO)

def main():
    """
    Example of saving a mesh with mixed element types (Hex and Tet).
    """
    # Create a Hex mesh (ImageData)
    hex_grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    
    # Create a Tet mesh
    # We can tetrahedralize a copy of the hex grid
    # But let's make it distinct geometry
    # Shift it to the right
    tet_grid = hex_grid.copy()
    tet_grid.points[:, 0] += 5.0
    tet_grid = tet_grid.triangulate()
    
    # Combine them into one mesh
    # simple concatenation of points and cells
    combined = hex_grid.merge(tet_grid)
    
    # Assign regions
    # Let's say hexes are region 1, tets are region 2
    # But wait, write_exo splits by cell type AND region.
    # So even if they have the same region, they will be different blocks because different cell types.
    # Let's verify this behavior.
    combined.cell_data["region"] = np.ones(combined.n_cells, dtype=int)
    
    # Save to EXO
    output_filename = "example_mixed_elements.exo"
    logging.info(f"Saving Mixed Element mesh to {output_filename}...")
    # Expect 2 blocks: one for Hex, one for Tet (since region is 1 for all)
    # block_names=["hex_block", "tet_block"] might work if the order is deterministic.
    # write_pyvista.py iterates: `for cell_type in pv.CellType:`
    # The order of pv.CellType iteration determines block order if regions are same.
    # Usually: VERTEX, POLY_VERTEX, LINE, POLY_LINE, TRIANGLE, TRIANGLE_STRIP, POLYGON, PIXEL, QUAD, TETRA, VOXEL, HEXAHEDRON...
    # So TETRA (10) comes before HEXAHEDRON (12)?
    # Let's check pv.CellType values if we care about names.
    # If we don't pass names, it defaults to block_0, block_1.
    exovista.write_exo(output_filename, combined, block_names=["tet_block", "hex_block"])
    logging.info("Done.")

if __name__ == "__main__":
    main()
