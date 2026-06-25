
import unittest
import pyvista as pv
import numpy as np
import exovista
import os
import tempfile
import shutil


class TestTimeFields(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Two hex cells sharing a face, split into two regions/blocks.
        points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1],
        ], dtype=float)
        cells = np.concatenate([
            [8, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 1, 8, 9, 2, 5, 10, 11, 6],
        ])
        cell_types = np.array([pv.CellType.HEXAHEDRON] * 2)
        self.grid = pv.UnstructuredGrid(cells, cell_types, points)
        self.grid.cell_data["region"] = np.array([1, 2])
        self.points = points

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_node_and_element_time_fields(self):
        times = np.array([0.0, 0.5, 1.0])
        temperature = np.array([t * self.points[:, 0] for t in times])  # (3, 12)
        pressure = np.array([[t, 2 * t] for t in times])                # (3, 2)

        out = os.path.join(self.test_dir, "time_fields.exo")
        exovista.write_exo(
            out, self.grid,
            times=times,
            node_fields={"temperature": temperature},
            element_fields={"pressure": pressure},
        )
        self.assertTrue(os.path.exists(out))

        f = exovista.File(out, mode="r")
        try:
            self.assertEqual(f.num_times(), 3)
            np.testing.assert_allclose(f.get_times(), times)
            self.assertIn("temperature", list(f.get_node_variable_names()))
            self.assertIn("pressure", list(f.get_element_variable_names()))

            for step in range(1, 4):
                vals = np.asarray(
                    f.get_node_variable_values("temperature", time_step=step)
                )
                np.testing.assert_allclose(vals, temperature[step - 1])

            # Block 1 holds cell 0, block 2 holds cell 1.
            block_ids = list(f.get_element_block_ids())
            for step in range(1, 4):
                b1 = np.asarray(
                    f.get_element_variable_values(block_ids[0], "pressure", time_step=step)
                )
                b2 = np.asarray(
                    f.get_element_variable_values(block_ids[1], "pressure", time_step=step)
                )
                np.testing.assert_allclose(b1, [pressure[step - 1, 0]])
                np.testing.assert_allclose(b2, [pressure[step - 1, 1]])
        finally:
            f.close()

    def test_single_step_1d_field(self):
        # A 1D field is accepted as a single time step.
        out = os.path.join(self.test_dir, "single_step.exo")
        node_vals = self.points[:, 1].copy()
        exovista.write_exo(
            out, self.grid,
            node_fields={"height": node_vals},
        )
        f = exovista.File(out, mode="r")
        try:
            self.assertEqual(f.num_times(), 1)
            vals = np.asarray(f.get_node_variable_values("height", time_step=1))
            np.testing.assert_allclose(vals, node_vals)
        finally:
            f.close()

    def test_shape_mismatch_raises(self):
        out = os.path.join(self.test_dir, "bad.exo")
        with self.assertRaises(ValueError):
            exovista.write_exo(
                out, self.grid,
                times=np.array([0.0, 1.0]),
                node_fields={"temperature": np.zeros((3, self.grid.n_points))},
            )


if __name__ == "__main__":
    unittest.main()
