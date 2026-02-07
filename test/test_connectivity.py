
import unittest
import pyvista as pv
import numpy as np
import os
import shutil
import tempfile
import netCDF4
import exovista

class TestConnectivity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_voxel_connectivity(self):
        # Create a single Voxel
        # Points: 0,0,0; 1,0,0; 0,1,0; 1,1,0; ...
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=float)
        # Voxel connectivity: 0, 1, 2, 3, 4, 5, 6, 7
        cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])
        cell_types = np.array([pv.CellType.VOXEL])
        grid = pv.UnstructuredGrid(cells, cell_types, points)
        grid.cell_data["region"] = np.array([1])

        output_file = os.path.join(self.test_dir, "voxel_test.exo")
        exovista.write_exo(output_file, grid)

        # Read back with netCDF4 to check connectivity
        # Note: Exodus uses 1-based indexing for connectivity
        with netCDF4.Dataset(output_file, 'r') as nc:
            # Connect1 should be the first block
            conn = nc.variables['connect1'][:]
            # Expected Hex order: 0, 1, 3, 2, 4, 5, 7, 6 (0-based)
            # So 1-based: 1, 2, 4, 3, 5, 6, 8, 7
            expected = np.array([1, 2, 4, 3, 5, 6, 8, 7])
            print(f"Read connectivity: {conn}")
            np.testing.assert_array_equal(conn[0], expected)

    def test_pixel_connectivity(self):
        # Create a single Pixel
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
        ], dtype=float)
        # Pixel connectivity: 0, 1, 2, 3
        cells = np.array([4, 0, 1, 2, 3])
        cell_types = np.array([pv.CellType.PIXEL])
        grid = pv.UnstructuredGrid(cells, cell_types, points)
        grid.cell_data["region"] = np.array([1])

        output_file = os.path.join(self.test_dir, "pixel_test.exo")
        exovista.write_exo(output_file, grid)

        with netCDF4.Dataset(output_file, 'r') as nc:
            conn = nc.variables['connect1'][:]
            # Expected Quad order: 0, 1, 3, 2 (0-based)
            # So 1-based: 1, 2, 4, 3
            expected = np.array([1, 2, 4, 3])
            print(f"Read connectivity: {conn}")
            np.testing.assert_array_equal(conn[0], expected)

if __name__ == "__main__":
    unittest.main()
