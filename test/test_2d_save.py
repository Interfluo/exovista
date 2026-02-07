
import unittest
import pyvista as pv
import numpy as np
import exovista
import os
import tempfile
import shutil

class Test2DSave(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_save_quad_mesh_with_sideset(self):
        # Create a simple 2D mesh with Quads
        # 0 -- 1
        # |    |
        # 3 -- 2
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ], dtype=float)
        
        # Cell connectivity
        cells = np.array([4, 0, 1, 2, 3])
        cell_types = np.array([pv.CellType.QUAD])
        
        volume = pv.UnstructuredGrid(cells, cell_types, points)
        volume.cell_data["region"] = np.array([1])
        
        # Create a side set (lines)
        # Edge 1-2
        side_cells = np.array([2, 1, 2])
        
        # Create PolyData explicitly
        surface = pv.PolyData()
        surface.points = points
        surface.lines = side_cells
        
        # Needs region data for the surface too
        surface.cell_data["region"] = np.ones(surface.n_cells, dtype=int)

        output_file = os.path.join(self.test_dir, "test_quad_sideset.exo")
        
        # Try to save it
        # This should not raise an exception
        exovista.write_exo(output_file, volume, surface=surface)
        
        self.assertTrue(os.path.exists(output_file))
        
        # Verify we can read it back
        mesh = pv.read(output_file)
        while isinstance(mesh, pv.MultiBlock) and len(mesh) > 0:
             mesh = mesh[0]
        
        self.assertEqual(mesh.n_cells, 1)
        self.assertEqual(mesh.n_points, 4)

if __name__ == "__main__":
    unittest.main()
