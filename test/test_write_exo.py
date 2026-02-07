
import unittest
import pyvista as pv
import numpy as np
import exovista
import os
import tempfile
import shutil

class TestWriteExo(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_hex_mesh(self):
        # Create a simple 3D mesh with Hex manually to avoid VOXEL type
        # 8 points, 1 hex
        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float)
        cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])
        cell_types = np.array([pv.CellType.HEXAHEDRON])
        grid = pv.UnstructuredGrid(cells, cell_types, points)
        grid.cell_data["region"] = np.ones(grid.n_cells, dtype=int)
        
        output_file = os.path.join(self.test_dir, "test_hex.exo")
        
        # Save
        exovista.write_exo(output_file, grid)
        
        self.assertTrue(os.path.exists(output_file))
        
        # Read back to verify
        mesh = pv.read(output_file)
        while isinstance(mesh, pv.MultiBlock) and len(mesh) > 0:
             mesh = mesh[0]
        self.assertEqual(mesh.n_cells, 1)
        self.assertEqual(mesh.n_points, 8)

    def test_save_tet_mesh(self):
        # Create a simple Tet mesh
        # 4 points, 1 tet
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        cells = np.array([4, 0, 1, 2, 3])
        cell_types = np.array([pv.CellType.TETRA])
        grid = pv.UnstructuredGrid(cells, cell_types, points)
        grid.cell_data["region"] = np.array([1])
        
        output_file = os.path.join(self.test_dir, "test_tet.exo")
        
        # Save
        exovista.write_exo(output_file, grid)
        
        self.assertTrue(os.path.exists(output_file))
        
        # Read back
        mesh = pv.read(output_file)
        while isinstance(mesh, pv.MultiBlock) and len(mesh) > 0:
             mesh = mesh[0]
        self.assertEqual(mesh.n_cells, 1)
        self.assertEqual(mesh.n_points, 4)

    def test_save_hex_with_sideset(self):
        # Hex with side set
        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float)
        cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])
        cell_types = np.array([pv.CellType.HEXAHEDRON])
        grid = pv.UnstructuredGrid(cells, cell_types, points)
        grid.cell_data["region"] = np.ones(grid.n_cells, dtype=int)
        
        # Surface (one face)
        surface = grid.extract_surface()
        # keep only one face for simplicity? Or just use full surface
        # Let's just use the full surface as side set
        surface.cell_data["region"] = np.ones(surface.n_cells, dtype=int)
        
        output_file = os.path.join(self.test_dir, "test_hex_sideset.exo")
        
        exovista.write_exo(output_file, grid, surface=surface)
        
        self.assertTrue(os.path.exists(output_file))
        
        # Read back
        mesh = pv.read(output_file)
        while isinstance(mesh, pv.MultiBlock) and len(mesh) > 0:
             mesh = mesh[0]
        self.assertEqual(mesh.n_cells, 1)

if __name__ == "__main__":
    unittest.main()
