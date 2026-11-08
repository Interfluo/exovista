
"""Time-varying fields written by write_exo, read back through the legacy
``exovista.File`` API (1-based / negative ``time_step``) and the 2.0 readers.

exovista 2.0 stores everything as float64, so values and times round-trip
exactly (1.x stored float32), block ids are the region values, and a single
static step gets the time 0.0 (1.x left it unwritten: 9.97e36).
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import pyvista as pv

import exovista


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
            np.testing.assert_array_equal(f.get_times(), times)
            self.assertIn("temperature", list(f.get_node_variable_names()))
            self.assertIn("pressure", list(f.get_element_variable_names()))

            for step in range(1, 4):
                vals = np.asarray(
                    f.get_node_variable_values("temperature", time_step=step)
                )
                np.testing.assert_array_equal(vals, temperature[step - 1])

            # The block ids are the region values: block 1 holds cell 0,
            # block 2 holds cell 1.
            block_ids = list(f.get_element_block_ids())
            self.assertEqual(block_ids, [1, 2])
            for step in range(1, 4):
                b1, b2 = (
                    np.asarray(f.get_element_variable_values(
                        block_id, "pressure", time_step=step))
                    for block_id in block_ids
                )
                np.testing.assert_array_equal(b1, [pressure[step - 1, 0]])
                np.testing.assert_array_equal(b2, [pressure[step - 1, 1]])
        finally:
            f.close()

        # the 2.0 readers see the same history
        grid = exovista.read_exo(out, time_step=2)
        np.testing.assert_array_equal(grid.point_data["temperature"], temperature[1])
        np.testing.assert_array_equal(grid.cell_data["pressure"], pressure[1])
        np.testing.assert_array_equal(grid.field_data["times"], times)

    def test_time_step_indexing(self):
        # 1-based positive, Python-style negative-from-end, and 0 is invalid.
        times = np.array([0.0, 0.5, 1.0])
        temperature = np.array([t * self.points[:, 0] for t in times])
        pressure = np.array([[t, 2 * t] for t in times])
        out = os.path.join(self.test_dir, "indexing.exo")
        exovista.write_exo(
            out, self.grid, times=times,
            node_fields={"temperature": temperature},
            element_fields={"pressure": pressure},
        )
        f = exovista.File(out, mode="r")
        try:
            # Positive: 1 -> first step.
            np.testing.assert_allclose(
                f.get_node_variable_values("temperature", time_step=1),
                temperature[0],
            )
            # Negative indexes from the end: -1 last, -2 second-to-last.
            np.testing.assert_allclose(
                f.get_node_variable_values("temperature", time_step=-1),
                temperature[2],
            )
            np.testing.assert_allclose(
                f.get_node_variable_values("temperature", time_step=-2),
                temperature[1],
            )
            block_ids = list(f.get_element_block_ids())
            np.testing.assert_allclose(
                f.get_element_variable_values(
                    block_ids[0], "pressure", time_step=-2),
                [pressure[1, 0]],
            )
            # 0 is not a valid 1-based step: it must raise, not return the last.
            with self.assertRaises(IndexError):
                f.get_node_variable_values("temperature", time_step=0)
            with self.assertRaises(IndexError):
                f.get_element_variable_values(block_ids[0], "pressure", time_step=0)
        finally:
            f.close()
        # read_exo has the same time_step semantics
        np.testing.assert_array_equal(
            exovista.read_exo(out, time_step=-2).point_data["temperature"],
            temperature[1],
        )
        with self.assertRaises(IndexError):
            exovista.read_exo(out, time_step=0)

    def test_read_field_mirror_functions(self):
        times = np.array([0.0, 0.5, 1.0])
        temperature = np.array([t * self.points[:, 0] for t in times])
        pressure = np.array([[t, 2 * t] for t in times])
        out = os.path.join(self.test_dir, "mirror.exo")
        exovista.write_exo(
            out, self.grid, times=times,
            node_fields={"temperature": temperature},
            element_fields={"pressure": pressure},
        )

        # Full time history, returned in the same layout write_exo accepted.
        node_fields = exovista.read_node_fields(out)
        np.testing.assert_allclose(node_fields["temperature"], temperature)
        self.assertEqual(node_fields["temperature"].shape, temperature.shape)

        elem_fields = exovista.read_element_fields(out)
        np.testing.assert_allclose(elem_fields["pressure"], pressure)
        self.assertEqual(elem_fields["pressure"].shape, pressure.shape)

        # Name selection (str and list) and unknown-name errors.
        self.assertEqual(
            list(exovista.read_node_fields(out, names="temperature")),
            ["temperature"],
        )
        self.assertEqual(
            list(exovista.read_element_fields(out, names=["pressure"])),
            ["pressure"],
        )
        with self.assertRaises(KeyError):
            exovista.read_node_fields(out, names="missing")

    def test_read_exo_variable_selection(self):
        times = np.array([0.0, 1.0])
        temperature = np.array([t * self.points[:, 0] for t in times])
        pressure = np.array([[t, 2 * t] for t in times])
        out = os.path.join(self.test_dir, "select.exo")
        exovista.write_exo(
            out, self.grid, times=times,
            node_fields={"temperature": temperature},
            element_fields={"pressure": pressure},
        )
        grid = exovista.read_exo(
            out, read_node_variables="temperature", read_element_variables=False
        )
        self.assertIn("temperature", grid.point_data)
        self.assertNotIn("pressure", grid.cell_data)
        with self.assertRaises(KeyError):
            exovista.read_exo(out, read_node_variables="missing")

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
            np.testing.assert_array_equal(f.get_times(), [0.0])
            vals = np.asarray(f.get_node_variable_values("height", time_step=1))
            np.testing.assert_array_equal(vals, node_vals)
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
        self.assertFalse(os.path.exists(out))  # nothing written on error


if __name__ == "__main__":
    unittest.main()
