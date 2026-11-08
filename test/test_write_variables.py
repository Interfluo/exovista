"""Variables written by exovista.write_exo: static arrays, histories, globals.

Values are checked with VTK's ExodusII reader at every time step (cell values
mapped back to the input cells through VTK's GlobalElementId = original cell
index + 1) and with exovista._backend.open_reader / raw netCDF4.
"""

import inspect
import warnings

import netCDF4
import numpy as np
import pytest
import pyvista as pv

from _pvcompat import exodus_reader
from exovista._backend import open_reader
from exovista.write_pyvista import write_exo

CT = pv.CellType


def vtk_read(path, time_point=0):
    reader = exodus_reader(str(path))
    reader.enable_all_point_arrays()
    reader.enable_all_cell_arrays()
    reader.reader.SetAllArrayStatus(reader.reader.GLOBAL, 1)
    reader.reader.GenerateGlobalElementIdArrayOn()
    if reader.number_time_points:
        reader.set_active_time_point(time_point)
    return reader, reader.read()["Element Blocks"]


def vtk_cell_values(blocks, name, n_cells):
    """Cell array ``name`` over all VTK blocks, in ORIGINAL cell order."""
    out = None
    for block in blocks:
        values = np.asarray(block.cell_data[name])
        if out is None:
            out = np.full((n_cells,) + values.shape[1:], np.nan)
        out[np.asarray(block.cell_data["GlobalElementId"]) - 1] = values
    return out


def vtk_point_values(blocks, grid, name):
    """Point array ``name`` at the input points.

    VTK keeps only the points each block uses, so look every input point up
    (by coordinates) in the blocks.
    """
    found = {}
    for block in blocks:
        values = np.asarray(block.point_data[name])
        for p, v in zip(np.asarray(block.points), values):
            found[tuple(p)] = v
    return np.array([found[tuple(p)] for p in np.asarray(grid.points, np.float32)])


def var_names(path):
    with open_reader(path) as reader:
        return (
            reader.node_var_names(), reader.elem_var_names(),
            reader.global_var_names(), reader.times().tolist(),
        )


def hex_grid(nx=2, ny=1, nz=1):
    axes = [np.arange(n + 1, dtype=float) for n in (nx, ny, nz)]
    grid = pv.StructuredGrid(*np.meshgrid(*axes, indexing="ij"))
    return grid.cast_to_unstructured_grid()


def X(suffixes, name):
    return [name + s for s in suffixes]


VEC3 = ["_X", "_Y", "_Z"]
SYM6 = ["_XX", "_YY", "_ZZ", "_XY", "_YZ", "_XZ"]
FULL9 = ["_XX", "_XY", "_XZ", "_YX", "_YY", "_YZ", "_ZX", "_ZY", "_ZZ"]


# --------------------------------------------------------------------------- #
# static point / cell data
# --------------------------------------------------------------------------- #
def test_static_arrays_split_into_components(tmp_path):
    grid = hex_grid(3, 1, 1)
    n, m = grid.n_points, grid.n_cells
    rng = np.random.default_rng(1)
    grid.point_data["T"] = rng.integers(0, 64, n) / 4.0
    grid.point_data["I"] = np.arange(n, dtype=np.int32)
    grid.point_data["B"] = np.arange(n) % 2 == 0
    grid.point_data["V"] = rng.integers(0, 64, (n, 3)) / 4.0
    grid.point_data["W"] = rng.integers(0, 64, (n, 2)) / 4.0
    grid.point_data["S"] = rng.integers(0, 64, (n, 6)) / 4.0
    grid.point_data["F"] = rng.integers(0, 64, (n, 9)) / 4.0
    grid.point_data["Q"] = rng.integers(0, 64, (n, 4)) / 4.0
    grid.cell_data["c"] = [1.5, 2.5, 3.5]
    grid.cell_data["cv"] = np.arange(9.0).reshape(3, 3)
    grid.cell_data["region"] = [2, 1, 2]
    path = tmp_path / "static.exo"
    write_exo(path, grid)

    nodes, elems, globals_, times = var_names(path)
    assert nodes == (
        ["T", "I", "B"] + X(VEC3, "V") + X(["_X", "_Y"], "W") + X(SYM6, "S")
        + X(FULL9, "F") + X(["_0", "_1", "_2", "_3"], "Q")
    )
    assert elems == ["c"] + X(VEC3, "cv")
    assert (globals_, times) == ([], [0.0])
    with open_reader(path) as reader:
        for name, suffixes in (("V", VEC3), ("S", SYM6), ("F", FULL9)):
            for k, suffix in enumerate(suffixes):
                np.testing.assert_array_equal(
                    reader.node_var(name + suffix, 0), grid.point_data[name][:, k]
                )
        np.testing.assert_array_equal(reader.node_var("B", 0), grid["B"] * 1.0)
        np.testing.assert_array_equal(reader.node_var("I", 0), grid["I"])

    _, blocks = vtk_read(path)
    # VTK regroups _X/_Y[/_Z] into one array named "<name>_" ...
    np.testing.assert_array_equal(vtk_point_values(blocks, grid, "V_"), grid["V"])
    np.testing.assert_array_equal(vtk_point_values(blocks, grid, "W_"), grid["W"])
    np.testing.assert_array_equal(vtk_point_values(blocks, grid, "T"), grid["T"])
    # ... a 9-tensor into its three rows, and keeps the 6-tensor as scalars
    for row, axis in enumerate("XYZ"):
        np.testing.assert_array_equal(
            vtk_point_values(blocks, grid, f"F_{axis}"),
            grid["F"][:, 3 * row: 3 * row + 3],
        )
    for k, suffix in enumerate(SYM6):
        np.testing.assert_array_equal(
            vtk_point_values(blocks, grid, "S" + suffix), grid["S"][:, k]
        )
    np.testing.assert_array_equal(vtk_cell_values(blocks, "c", m), grid["c"])
    np.testing.assert_array_equal(vtk_cell_values(blocks, "cv_", m), grid["cv"])


def test_n_cells_equal_to_n_points_is_not_misclassified(tmp_path):
    mesh = pv.Tetrahedron()  # 4 points, 4 triangles
    assert mesh.n_points == mesh.n_cells == 4
    mesh.point_data.clear()
    mesh.cell_data.clear()
    mesh.point_data["on_points"] = [1.0, 2.0, 3.0, 4.0]
    mesh.cell_data["on_cells"] = [10.0, 20.0, 30.0, 40.0]
    mesh.field_data["meta"] = [7.0, 8.0, 9.0, 10.0]  # never written
    write_exo(tmp_path / "eq.exo", mesh)
    assert var_names(tmp_path / "eq.exo") == (["on_points"], ["on_cells"], [], [0.0])
    _, blocks = vtk_read(tmp_path / "eq.exo")
    np.testing.assert_array_equal(
        vtk_cell_values(blocks, "on_cells", 4), mesh["on_cells"]
    )


def test_bare_mesh_has_no_variables_and_no_steps(tmp_path):
    write_exo(tmp_path / "bare.exo", hex_grid())
    assert var_names(tmp_path / "bare.exo") == ([], [], [], [])
    reader, _ = vtk_read(tmp_path / "bare.exo")
    assert reader.number_time_points == 0


def test_helper_and_region_arrays_are_not_written(tmp_path):
    grid = hex_grid()
    grid.cell_data["region"] = [1, 2]
    grid.point_data["vtkOriginalPointIds"] = np.arange(grid.n_points)
    grid.cell_data["vtkOriginalCellIds"] = [0, 1]
    grid.cell_data["vtkGhostType"] = np.zeros(2, np.uint8)
    grid.point_data["vtkGhostType"] = np.zeros(grid.n_points, np.uint8)
    grid.point_data["kept"] = np.zeros(grid.n_points)
    surface = grid.extract_surface(  # carries vtkOriginal* arrays too
        **({"algorithm": "dataset_surface"} if "algorithm" in inspect.signature(
            grid.extract_surface).parameters else {})
    )
    write_exo(tmp_path / "h.exo", grid, surface)
    assert var_names(tmp_path / "h.exo") == (["kept"], [], [], [0.0])
    with netCDF4.Dataset(tmp_path / "h.exo") as nc:
        text = str(nc.variables["name_nod_var"][:].tobytes())
    for leaked in ("orig_pts", "orig_cell_ids", "cell_ids", "face_ids", "vtkOrig"):
        assert leaked not in text
    # region is written when it is not the region key
    write_exo(tmp_path / "r.exo", grid, region_key=None)
    assert "region" in var_names(tmp_path / "r.exo")[1]


def test_non_numeric_and_complex_arrays_are_skipped_with_warning(tmp_path):
    grid = hex_grid()
    grid.point_data["label"] = [f"p{i}" for i in range(grid.n_points)]
    grid.point_data["z"] = np.ones(grid.n_points) * (1 + 2j)
    grid.cell_data["name"] = ["a", "b"]
    grid.point_data["ok"] = np.ones(grid.n_points)
    with pytest.warns(UserWarning) as record:
        write_exo(tmp_path / "s.exo", grid)
    messages = " | ".join(str(w.message) for w in record)
    assert "'label'" in messages and "non-numeric" in messages
    assert "'z'" in messages and "complex" in messages
    assert "'name'" in messages
    assert var_names(tmp_path / "s.exo")[:2] == (["ok"], [])


def test_save_flags(tmp_path):
    grid = hex_grid()
    grid.point_data["p"] = np.zeros(grid.n_points)
    grid.cell_data["c"] = [1.0, 2.0]
    write_exo(tmp_path / "a.exo", grid, save_node_arrays=False)
    assert var_names(tmp_path / "a.exo")[:2] == ([], ["c"])
    write_exo(tmp_path / "b.exo", grid, save_cell_arrays=False)
    assert var_names(tmp_path / "b.exo")[:2] == (["p"], [])
    write_exo(tmp_path / "c.exo", grid, save_node_arrays=False, save_cell_arrays=False)
    assert var_names(tmp_path / "c.exo") == ([], [], [], [])


def test_static_names_up_to_256_bytes(tmp_path):
    grid = hex_grid()
    grid.point_data["a" * 256] = np.ones(grid.n_points)  # exactly the limit
    grid.point_data["v" * 254] = np.ones((grid.n_points, 3))  # "_X" -> 256
    grid.point_data["w" * 255] = np.ones((grid.n_points, 3))  # "_X" -> 257
    with pytest.warns(UserWarning, match=r"'w+'\.\.\.: the variable name is longer"):
        write_exo(tmp_path / "n.exo", grid)
    assert var_names(tmp_path / "n.exo")[0] == ["a" * 256] + X(VEC3, "v" * 254)


def test_static_name_clash_is_skipped_with_warning(tmp_path):
    grid = hex_grid()
    grid.point_data["p_X"] = np.zeros(grid.n_points)
    grid.point_data["p"] = np.ones((grid.n_points, 3))
    with pytest.warns(UserWarning, match=r"skipped point array 'p'.*p_X"):
        write_exo(tmp_path / "c.exo", grid)
    assert var_names(tmp_path / "c.exo")[0] == ["p_X"]


# --------------------------------------------------------------------------- #
# time histories
# --------------------------------------------------------------------------- #
def interleaved():
    grid = hex_grid(4, 1, 1)
    grid.cell_data["region"] = [2, 1, 2, 1]
    return grid


def test_time_varying_fields_at_every_step(tmp_path):
    grid = interleaved()
    m = grid.n_cells
    times = np.array([0.0, 0.25, 0.5, 1.0])
    x = np.asarray(grid.points)[:, 0]
    temperature = np.array([t * x for t in times])  # (4, n)
    displacement = np.array([t * np.asarray(grid.points) for t in times])  # (4, n, 3)
    pressure = np.array([t * np.arange(m) for t in times])  # original cell order
    stress = np.array([t * np.arange(2 * m).reshape(m, 2) for t in times])
    energy = 2 * times
    path = tmp_path / "hist.exo"
    write_exo(
        path, grid, times=times,
        node_fields={"T": temperature, "U": displacement},
        element_fields={"p": pressure, "s": stress},
        global_fields={"energy": energy, "mom": np.c_[times, -times]},
    )
    assert var_names(path) == (
        ["T"] + X(VEC3, "U"), ["p", "s_X", "s_Y"], ["energy", "mom_X", "mom_Y"],
        times.tolist(),
    )
    for k in range(len(times)):
        reader, blocks = vtk_read(path, k)
        assert reader.time_values == pytest.approx(times)
        T, U = (vtk_point_values(blocks, grid, a) for a in ("T", "U_"))
        np.testing.assert_allclose(T, temperature[k])
        np.testing.assert_allclose(U, displacement[k])
        np.testing.assert_allclose(vtk_cell_values(blocks, "p", m), pressure[k])
        np.testing.assert_allclose(vtk_cell_values(blocks, "s_", m), stress[k])
    np.testing.assert_allclose(blocks[0].field_data["energy"], energy)
    np.testing.assert_allclose(blocks[0].field_data["mom_"], np.c_[times, -times])
    with open_reader(path) as reader:
        np.testing.assert_array_equal(reader.global_var("energy"), energy)
        for block in reader.blocks():
            cells = reader.elem_id_map()  # 1-based original ids, file order
            assert block.n_elem == 2
        np.testing.assert_array_equal(cells, [2, 4, 1, 3])
        np.testing.assert_array_equal(reader.elem_var("p", 1), pressure[:, [1, 3]])
        np.testing.assert_array_equal(reader.elem_var("p", 2), pressure[:, [0, 2]])


def test_static_arrays_are_constant_over_steps(tmp_path):
    grid = interleaved()
    grid.point_data["P"] = np.arange(grid.n_points, dtype=float)
    grid.cell_data["C"] = [5.0, 6.0, 7.0, 8.0]
    path = tmp_path / "const.exo"
    write_exo(path, grid, times=[1.0, 2.0, 3.0])
    with open_reader(path) as reader:
        np.testing.assert_array_equal(
            reader.node_var("P"), np.tile(grid["P"], (3, 1))
        )
        np.testing.assert_array_equal(reader.elem_var("C", 1), [[6.0, 8.0]] * 3)
    for k in range(3):
        _, blocks = vtk_read(path, k)
        np.testing.assert_array_equal(vtk_cell_values(blocks, "C", 4), grid["C"])


def test_time_varying_names_win_over_static_arrays(tmp_path):
    grid = hex_grid()
    grid.point_data["T"] = np.full(grid.n_points, -1.0)
    grid.point_data["U"] = np.full((grid.n_points, 3), -1.0)
    grid.cell_data["E"] = [-1.0, -1.0]
    T = np.ones((2, grid.n_points))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(
            tmp_path / "w.exo", grid, times=[0, 1],
            node_fields={"T": T, "U": np.zeros((2, grid.n_points))},
            element_fields={"E": [[1.0, 2.0], [3.0, 4.0]]},
        )
    assert var_names(tmp_path / "w.exo")[:2] == (["T", "U"], ["E"])
    with open_reader(tmp_path / "w.exo") as reader:
        np.testing.assert_array_equal(reader.node_var("T"), T)
        np.testing.assert_array_equal(reader.elem_var("E", 1), [[1, 2], [3, 4]])


def test_times_none_gives_one_step_at_zero(tmp_path):
    grid = hex_grid()
    n, m = grid.n_points, grid.n_cells
    write_exo(
        tmp_path / "one.exo", grid,
        node_fields={"a": np.arange(n), "v": np.ones((n, 3))},  # (n,), (n, 3)
        element_fields={"e": np.ones((1, m)), "ev": np.ones((1, m, 2))},
        global_fields={"g": 4.0},
    )
    assert var_names(tmp_path / "one.exo") == (
        ["a"] + X(VEC3, "v"), ["e", "ev_X", "ev_Y"], ["g"], [0.0]
    )
    reader, _ = vtk_read(tmp_path / "one.exo")
    assert reader.time_values == [0.0]
    # times without variables are written too
    write_exo(tmp_path / "t.exo", grid, times=[0.5, 1.5])
    assert var_names(tmp_path / "t.exo") == ([], [], [], [0.5, 1.5])


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"node_fields": {"a": np.ones((2, 12))}}, ValueError, "one step"),
        ({"times": [0, 1], "node_fields": {"a": np.ones(12)}}, ValueError,
         r"\(2, 12\)"),
        ({"times": [0, 1], "node_fields": {"a": np.ones((2, 11))}}, ValueError,
         "shape"),
        ({"element_fields": {"e": np.ones(3)}}, ValueError, r"\(1, 2\)"),
        ({"times": [0], "element_fields": {"e": np.ones((1, 2, 3, 4))}}, ValueError,
         "shape"),
        ({"global_fields": {"g": [1.0, 2.0]}}, ValueError, r"\(1,\)"),
        ({"times": []}, ValueError, "empty"),
        ({"times": [0.0, np.nan]}, ValueError, "NaN"),
        ({"times": [[0.0, 1.0]]}, ValueError, "1D"),
        ({"node_fields": [("a", 1)]}, TypeError, "dict"),
        ({"node_fields": {1: np.ones(12)}}, TypeError, "strings"),
        ({"node_fields": {"s": np.array(["x"] * 12)}}, TypeError, "numeric"),
        ({"node_fields": {"c": np.ones(12) * 1j}}, TypeError, "numeric"),
        ({"node_fields": {"v": np.ones((1, 12, 3)), "v_X": np.ones(12)}}, ValueError,
         "used twice"),
        ({"node_fields": {"x" * 257: np.ones(12)}}, ValueError, "256"),
    ],
)
def test_field_errors(tmp_path, kwargs, error, match):
    with pytest.raises(error, match=match):
        write_exo(tmp_path / "x.exo", hex_grid(), **kwargs)
    assert not (tmp_path / "x.exo").exists()


def test_long_variable_names(tmp_path):
    grid = hex_grid()
    long_node = "node_variable_with_a_name_longer_than_32_characters"
    long_elem = "e" * 200
    write_exo(
        tmp_path / "long.exo", grid,
        node_fields={long_node: np.arange(grid.n_points, dtype=float)},
        element_fields={long_elem: [1.0, 2.0]},
        global_fields={"g" * 100: 1.0},
    )
    assert var_names(tmp_path / "long.exo")[:3] == (
        [long_node], [long_elem], ["g" * 100]
    )
    _, blocks = vtk_read(tmp_path / "long.exo")
    np.testing.assert_array_equal(
        vtk_point_values(blocks, grid, long_node), np.arange(grid.n_points)
    )
    np.testing.assert_array_equal(vtk_cell_values(blocks, long_elem, 2), [1.0, 2.0])


def test_multiblock_element_fields_follow_leaf_order(tmp_path):
    a = hex_grid(2, 1, 1)
    b = hex_grid(1, 1, 1).translate((5, 0, 0))
    mb = pv.MultiBlock({"a": a, "b": b})
    values = np.array([[10.0, 11.0, 20.0]])  # a's two cells, then b's
    write_exo(tmp_path / "mb.exo", mb, element_fields={"e": values})
    with open_reader(tmp_path / "mb.exo") as reader:
        assert [(blk.id, blk.name) for blk in reader.blocks()] == [(1, "a"), (2, "b")]
        np.testing.assert_array_equal(reader.elem_var("e", 1), [[10.0, 11.0]])
        np.testing.assert_array_equal(reader.elem_var("e", 2), [[20.0]])


def test_single_precision_variables(tmp_path):
    grid = hex_grid()
    write_exo(
        tmp_path / "s.exo", grid, precision="single", times=[0.5],
        node_fields={"a": np.ones(grid.n_points)}, element_fields={"e": [1, 2]},
        global_fields={"g": 3.0},
    )
    with netCDF4.Dataset(tmp_path / "s.exo") as nc:
        for name in ("vals_nod_var1", "vals_elem_var1eb1", "vals_glo_var",
                     "time_whole", "coordx"):
            assert nc.variables[name].dtype == np.float32, name
