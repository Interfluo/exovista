
import pyvista as pv
import numpy as np
import exovista
import logging

logging.basicConfig(level=logging.INFO)

def main():
    """
    Example of saving Side Sets on a 3D mesh.
    """
    # Create a simple Tet mesh from a box
    # Create a box and tetrahedralize it
    box = pv.Box(bounds=(0, 10, 0, 10, 0, 10))
    # Triangulate to get surface triangles
    box_tris = box.triangulate()
    # This is a surface mesh. To get a volume mesh, we need volume cells.
    # PyVista's `tetrahedralize` works on PolyData to create UnstructuredGrid
    # but we need points inside. Let's make a grid and tetrahedralize it.
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    # This splits hexes into tets (5 per hex usually or 6)
    tet_grid = grid.triangulate()
    
    tet_grid.cell_data["region"] = np.ones(tet_grid.n_cells, dtype=int)
    
    # Extract surface for side sets
    # Let's define the bottom (z=0) and top (z=10) surfaces as side sets
    surface = tet_grid.extract_surface()
    surface_centers = surface.cell_centers().points
    
    # Currently `process_meshes` expects "region" on surface to split side sets?
    # Or does it use connectivity? 
    # write_exo calls process_meshes(..., surface, region_key)
    # process_meshes splits surface by region_key.
    
    surface_regions = np.zeros(surface.n_cells, dtype=int)
    
    # Top surface (z approx 2 - dimensions were 3x3x3 so z goes 0, 1, 2)
    # Bounds are default (0, 0, 0) to (2, 2, 2)
    top_indices = surface_centers[:, 2] > 1.9
    bottom_indices = surface_centers[:, 2] < 0.1
    
    surface_regions[top_indices] = 1 # Set 1
    surface_regions[bottom_indices] = 2 # Set 2
    # The rest remains 0 (or can be ignored if we want to filter them out beforehand, 
    # but process_meshes will create a side set for region 0 too if present)
    
    surface.cell_data["region"] = surface_regions
    
    # If we want to ONLY save top and bottom, we should probably threshold the surface
    # But let's save all sides, partitioned by these regions.
    
    output_filename = "example_side_sets.exo"
    logging.info(f"Saving Tet mesh with Side Sets to {output_filename}...")
    exovista.write_exo(output_filename, tet_grid, surface=surface, side_set_names=["sides_other", "sides_top", "sides_bottom"])
    logging.info("Done.")

if __name__ == "__main__":
    main()
