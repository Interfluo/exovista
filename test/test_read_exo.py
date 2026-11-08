"""Tests for exovista.read_exo: points, cell data, element order, variables
and time steps.

Input files are written with ``exovista._backend.write_model``; VTK's reader
(``pv.ExodusIIReader``) and netCDF4 serve as independent checks.
"""

import os
import sys
import warnings

import netCDF4
import numpy as np
import pytest
import pyvista as pv

import exovista
from exovista import _celltypes as ct
from exovista import read_pyvista as rp
from exovista._backend import write_model
from exovista._model import Block, ExodusModel, SideSet

VTK = ct.VTK


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cell_rows(grid):
    """Point ids of every cell (VTK order) as tuples."""
    return [tuple(grid.get_cell(i).point_ids) for i in range(grid.n_cells)]


def _model_from_grid(grid, region, **kwargs):
    """What write_exo stores: blocks by region (sorted), element ids =
    original cell index + 1 in file order."""
    rows = _cell_rows(grid)
    num_dim = kwargs.pop("num_dim", 3)
    blocks, file_order = [], []
    for k, value in enumerate(np.unique(region)):
        cells = np.flatnonzero(region == value)
        vtk_type = int(grid.celltypes[cells[0]])
        assert np.all(grid.celltypes[cells] == vtk_type)
        conn = np.array([rows[c] for c in cells])
        info = ct.exo_info_for(vtk_type, num_dim)
        exo_conn = ct.to_exodus_conn(vtk_type, conn)
        blocks.append(Block(k + 1, f"region {value}", info.name, exo_conn))
        file_order.extend(cells)
    file_order = np.array(file_order)
    model = ExodusModel(
        "test", num_dim, np.asarray(grid.points)[:, :num_dim], blocks,
        elem_id_map=file_order + 1, **kwargs
    )
    return model, file_order


def _interleaved_hexes():
    """3x2x2 hexes whose two regions interleave along x."""
    grid = pv.ImageData(dimensions=(4, 3, 3), spacing=(0.5, 0.25, 0.125))
    grid = grid.cast_to_unstructured_grid()
    # VOXEL -> HEXAHEDRON so the cells compare directly with what is read
    rows = grid.cells.reshape(-1, 9)[:, 1:][:, [0, 1, 3, 2, 4, 5, 7, 6]]
    grid = pv.UnstructuredGrid({VTK.HEXAHEDRON: rows}, grid.points)
    region = np.arange(grid.n_cells) % 2 + 1
    return grid, region


def _steps_model(**overrides):
    rng = np.random.default_rng(5)
    grid, region = _interleaved_hexes()
    n_steps = 4
    times = np.array([0.0, 0.5, 1.5, 3.0])
    model, file_order = _model_from_grid(grid, region)
    by_block = {}
    elem_hist = rng.random((n_steps, grid.n_cells))  # in original cell order
    start = 0
    for block in model.blocks:
        cells = file_order[start:start + block.n_elem]
        by_block[block.id] = elem_hist[:, cells]
        start += block.n_elem
    kwargs = dict(
        title=model.title, num_dim=3, coords=model.coords, blocks=model.blocks,
        elem_id_map=model.elem_id_map, times=times,
        node_vars={"u": rng.random((n_steps, grid.n_points))},
        elem_vars={"e": by_block},
        global_vars={"g": rng.random(n_steps)},
    )
    kwargs.update(overrides)
    return ExodusModel(**kwargs), grid, elem_hist


# --------------------------------------------------------------------------- #
# points
# --------------------------------------------------------------------------- #
def test_points_float64_never_downcast(tmp_path):
    rng = np.random.default_rng(0)
    coords = 1e6 + np.arange(12.0).reshape(4, 3) * 1e-3
    coords += rng.normal(size=(4, 3)) * 1e-9
    assert np.any(coords.astype(np.float32).astype(float) != coords)
    write_model(tmp_path / "p.exo",
                ExodusModel("p", 3, coords, [Block(1, "t", "TETRA4", [[0, 1, 2, 3]])]))
    grid = rp.read_exo(tmp_path / "p.exo")
    assert grid.points.dtype == np.float64
    np.testing.assert_array_equal(grid.points, coords)
    # also through the side-set path, which shares the points
    _, side_sets = rp.read_exo(tmp_path / "p.exo", read_side_sets=True)
    assert len(side_sets) == 0


def test_2d_and_1d_points_are_embedded(tmp_path):
    coords2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float) + 1e5 + 1e-7
    write_model(tmp_path / "q.exo",
                ExodusModel("q", 2, coords2, [Block(1, "q", "QUAD4", [[0, 1, 2, 3]])]))
    grid = rp.read_exo(tmp_path / "q.exo")
    assert grid.points.dtype == np.float64
    np.testing.assert_array_equal(grid.points[:, :2], coords2)
    np.testing.assert_array_equal(grid.points[:, 2], 0.0)
    assert list(grid.celltypes) == [VTK.QUAD]
    np.testing.assert_array_equal(grid.field_data["num_dim"], [2])

    coords1 = np.array([0.0, 0.5, 2.0])
    write_model(tmp_path / "b.exo",
                ExodusModel("b", 1, coords1, [Block(1, "b", "BAR2", [[0, 1], [1, 2]])]))
    grid = rp.read_exo(tmp_path / "b.exo")
    np.testing.assert_array_equal(grid.points[:, 0], coords1)
    np.testing.assert_array_equal(grid.points[:, 1:], 0.0)
    assert list(grid.celltypes) == [VTK.LINE, VTK.LINE]
    np.testing.assert_array_equal(grid.field_data["num_dim"], [1])


def test_single_precision_file_reads_as_float64(tmp_path):
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 0.1
    model = ExodusModel("f4", 3, coords, [Block(1, "t", "TETRA4", [[0, 1, 2, 3]])],
                        times=[0.1], node_vars={"u": [[0.1, 0.2, 0.3, 0.4]]})
    write_model(tmp_path / "f4.exo", model, float_word_size=4)
    grid = rp.read_exo(tmp_path / "f4.exo")
    assert grid.points.dtype == np.float64
    np.testing.assert_array_equal(grid.points, coords.astype(np.float32))
    assert grid.point_data["u"].dtype == np.float64
    np.testing.assert_array_equal(grid.field_data["times"], np.float32(0.1))


def test_points_only_file(tmp_path):
    write_model(tmp_path / "pts.exo", ExodusModel("pts", 3, np.eye(3)))
    grid = rp.read_exo(tmp_path / "pts.exo")
    assert (grid.n_points, grid.n_cells) == (3, 0)
    for name in ("exo_block_id", "exo_block_name", "exo_elem_id"):
        assert len(grid.cell_data[name]) == 0
    assert grid.field_data["times"].size == 0
    assert "time" not in grid.field_data


# --------------------------------------------------------------------------- #
# cell data and element order
# --------------------------------------------------------------------------- #
def test_block_cell_data(tmp_path):
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1],
                       [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    model = ExodusModel(
        "names", 3, coords,
        [Block(7, "", "TETRA4", [[0, 1, 2, 4]]),
         Block(-3, "bloc thermique é", "TETRA4", [[1, 2, 5, 6], [2, 3, 6, 7]]),
         Block(2**31 - 1, "x" * 256, "HEX8", [range(8)])],
    )
    write_model(tmp_path / "n.exo", model)
    grid = rp.read_exo(tmp_path / "n.exo")
    ids = grid.cell_data["exo_block_id"]
    assert ids.dtype == np.int64
    np.testing.assert_array_equal(ids, [7, -3, -3, 2**31 - 1])
    names = grid.cell_data["exo_block_name"]
    assert list(names) == ["", "bloc thermique é", "bloc thermique é", "x" * 256]
    assert all(type(str(n)) is str for n in names)
    np.testing.assert_array_equal(grid.cell_data["exo_elem_id"], [1, 2, 3, 4])
    assert grid.cell_data["exo_elem_id"].dtype == np.int64
    # the VTK string array survives a save/load round trip
    grid.save(tmp_path / "n.vtu")
    back = pv.read(tmp_path / "n.vtu")
    assert list(back.cell_data["exo_block_name"]) == list(names)


def test_original_order_restores_written_cell_order(tmp_path):
    model, grid_in, elem_hist = _steps_model()
    path = tmp_path / "order.exo"
    write_model(path, model)
    # the file groups cells by block: not the input order
    with netCDF4.Dataset(path) as ds:
        file_ids = np.ma.getdata(ds["elem_num_map"][:])
    assert list(file_ids) != sorted(file_ids)

    grid = rp.read_exo(path, time_step=2)
    assert _cell_rows(grid) == _cell_rows(grid_in)
    np.testing.assert_array_equal(grid.cell_data["exo_elem_id"],
                                  np.arange(1, grid.n_cells + 1))
    np.testing.assert_array_equal(grid.cell_data["exo_block_id"],
                                  np.arange(grid.n_cells) % 2 + 1)
    names = [f"region {k}" for k in np.arange(grid.n_cells) % 2 + 1]
    assert list(grid.cell_data["exo_block_name"]) == names
    np.testing.assert_array_equal(grid.cell_data["e"], elem_hist[1])
    np.testing.assert_array_equal(
        rp.read_element_fields(path)["e"], elem_hist
    )

    # file order on request
    raw = rp.read_exo(path, time_step=2, original_order=False)
    np.testing.assert_array_equal(raw.cell_data["exo_elem_id"], file_ids)
    assert _cell_rows(raw) == [_cell_rows(grid_in)[i - 1] for i in file_ids]
    np.testing.assert_array_equal(raw.cell_data["e"], elem_hist[1][file_ids - 1])
    np.testing.assert_array_equal(
        rp.read_element_fields(path, original_order=False)["e"],
        elem_hist[:, file_ids - 1],
    )


@pytest.mark.parametrize(
    "elem_id_map,expected_order",
    [
        (None, [0, 1, 2, 3]),  # no map: ids 1..N in file order
        ([1, 2, 3, 4], [0, 1, 2, 3]),  # identity
        ([10, 20, 30, 40], [0, 1, 2, 3]),  # not a permutation of 1..N
        ([4, 1, 7, 2], [0, 1, 2, 3]),  # gaps: not a permutation
        ([2, 3, 4, 5], [0, 1, 2, 3]),  # shifted: not a permutation
        ([3, 1, 4, 2], [1, 3, 0, 2]),  # permutation: sorted by id
    ],
)
def test_original_order_only_for_permutation_maps(tmp_path, elem_id_map,
                                                  expected_order):
    coords = np.vstack([np.eye(4, 3), np.eye(4, 3) + 2])
    blocks = [Block(1, "a", "TETRA4", [[0, 1, 2, 3], [4, 5, 6, 7]]),
              Block(2, "b", "TETRA4", [[1, 2, 3, 0], [5, 6, 7, 4]])]
    model = ExodusModel("m", 3, coords, blocks, elem_id_map=elem_id_map,
                        times=[0.0],
                        elem_vars={"e": {1: [[0.0, 1.0]], 2: [[2.0, 3.0]]}})
    write_model(tmp_path / "m.exo", model)
    ids = np.arange(1, 5) if elem_id_map is None else np.array(elem_id_map)
    grid = rp.read_exo(tmp_path / "m.exo")
    np.testing.assert_array_equal(grid.cell_data["exo_elem_id"], ids[expected_order])
    np.testing.assert_array_equal(grid.cell_data["e"], np.array(expected_order, float))
    file_rows = [tuple(r) for b in blocks for r in b.conn]
    assert _cell_rows(grid) == [file_rows[i] for i in expected_order]
    raw = rp.read_exo(tmp_path / "m.exo", original_order=False)
    np.testing.assert_array_equal(raw.cell_data["exo_elem_id"], ids)


# --------------------------------------------------------------------------- #
# time steps
# --------------------------------------------------------------------------- #
@pytest.fixture
def steps_file(tmp_path):
    model, grid, elem_hist = _steps_model()
    path = tmp_path / "steps.exo"
    write_model(path, model)
    return path, model, elem_hist


@pytest.mark.parametrize("time_step,index", [(1, 0), (2, 1), (4, 3), (-1, 3),
                                             (-4, 0), (np.int64(3), 2)])
def test_time_step_selects_values(steps_file, time_step, index):
    path, model, elem_hist = steps_file
    grid = rp.read_exo(path, time_step=time_step)
    np.testing.assert_array_equal(grid.field_data["times"], model.times)
    np.testing.assert_array_equal(grid.field_data["time"], [model.times[index]])
    np.testing.assert_array_equal(grid.field_data["num_dim"], [3])
    np.testing.assert_array_equal(grid.point_data["u"], model.node_vars["u"][index])
    np.testing.assert_array_equal(grid.cell_data["e"], elem_hist[index])
    np.testing.assert_array_equal(grid.field_data["g"],
                                  [model.global_vars["g"][index]])
    assert grid.point_data["u"].dtype == np.float64


def test_default_time_step_is_last(steps_file):
    path, model, elem_hist = steps_file
    grid = rp.read_exo(path)
    np.testing.assert_array_equal(grid.field_data["time"], [3.0])
    np.testing.assert_array_equal(grid.cell_data["e"], elem_hist[-1])


@pytest.mark.parametrize("time_step", [0, 5, -5, 100])
def test_time_step_out_of_range(steps_file, time_step):
    path = steps_file[0]
    with pytest.raises(IndexError):
        rp.read_exo(path, time_step=time_step)
    with pytest.raises(IndexError):
        rp.read_exo(path, time_step=time_step, read_node_variables=False,
                    read_element_variables=False, read_global_variables=False)


@pytest.mark.parametrize("time_step", [1.0, "1", None, True])
def test_time_step_type(steps_file, time_step):
    with pytest.raises(TypeError):
        rp.read_exo(steps_file[0], time_step=time_step)


def test_static_file_has_no_time(tmp_path):
    coords = np.eye(4, 3)
    write_model(tmp_path / "s.exo",
                ExodusModel("s", 3, coords, [Block(1, "t", "TETRA4", [[0, 1, 2, 3]])]))
    grid = rp.read_exo(tmp_path / "s.exo")
    assert grid.field_data["times"].size == 0
    assert grid.field_data["times"].dtype == np.float64
    assert "time" not in grid.field_data
    # a time step cannot be selected but is not an error, except 0
    grid = rp.read_exo(tmp_path / "s.exo", time_step=3)
    assert "time" not in grid.field_data
    with pytest.raises(IndexError):
        rp.read_exo(tmp_path / "s.exo", time_step=0)


def test_single_step_reads_do_not_load_histories(steps_file, monkeypatch):
    """read_exo reads one step of each variable, never the whole history."""
    from exovista import nc

    path = steps_file[0]
    calls = []
    original = nc.get_variable

    def spy(fh, name, default=None, raw=False, index=None):
        if name.startswith("vals_") and not raw:
            calls.append((name, index))
        return original(fh, name, default=default, raw=raw, index=index)

    monkeypatch.setattr(nc, "get_variable", spy)
    rp.read_exo(path, time_step=2)
    assert calls and all(index is not None for _, index in calls)


# --------------------------------------------------------------------------- #
# variables
# --------------------------------------------------------------------------- #
def _vector_model():
    """Node and element variables named with the writer's suffixes."""
    rng = np.random.default_rng(11)
    coords = np.vstack([np.eye(4, 3), np.eye(4, 3) + 2])
    blocks = [Block(1, "a", "TETRA4", [[0, 1, 2, 3]]),
              Block(2, "b", "TETRA4", [[4, 5, 6, 7], [5, 6, 7, 4]])]
    n_steps, n_nodes = 2, 8
    node = {}
    arrays = {}
    for name, n in [("vel", 3), ("uv", 2), ("sym", 6), ("full", 9), ("four", 4),
                    ("five", 5)]:
        arrays[name] = rng.random((n_steps, n_nodes, n))
        for k, suffix in enumerate(exovista._model.component_suffixes(n)):
            node[name + suffix] = arrays[name][:, :, k]
    # store the 9 components in a scrambled order: combination goes by name
    full = {k: node.pop(k) for k in list(node) if k.startswith("full_")}
    for key in sorted(full, reverse=True):
        node[key] = full[key]
    node.update({
        "partial_X": rng.random((n_steps, n_nodes)),  # no _Y: stays scalar
        "partial_Z": rng.random((n_steps, n_nodes)),
        "taken": rng.random((n_steps, n_nodes)),  # base name exists: no combine
        "taken_X": rng.random((n_steps, n_nodes)),
        "taken_Y": rng.random((n_steps, n_nodes)),
        "num_0": rng.random((n_steps, n_nodes)),  # 2 components use _X/_Y
        "num_1": rng.random((n_steps, n_nodes)),
        # letters and digits are separate conventions: mixed_X/_Y form a
        # 2-vector, mixed_0 stays a scalar
        "mixed_X": rng.random((n_steps, n_nodes)),
        "mixed_Y": rng.random((n_steps, n_nodes)),
        "mixed_0": rng.random((n_steps, n_nodes)),
        "T": rng.random((n_steps, n_nodes)),
    })
    elem = {
        "s_X": {1: rng.random((n_steps, 1)), 2: rng.random((n_steps, 2))},
        "s_Y": {1: rng.random((n_steps, 1)), 2: rng.random((n_steps, 2))},
        "s_Z": {2: rng.random((n_steps, 2))},  # not defined on block 1
        "p": {1: rng.random((n_steps, 1))},
    }
    model = ExodusModel("vec", 3, coords, blocks, times=[0.0, 1.0],
                        node_vars=node, elem_vars=elem,
                        global_vars={"g_X": [1.0, 2.0], "g_Y": [3.0, 4.0]})
    return model, arrays


def test_combine_components(tmp_path):
    model, arrays = _vector_model()
    path = tmp_path / "vec.exo"
    write_model(path, model)
    grid = rp.read_exo(path, time_step=1)
    pd = grid.point_data
    for name, values in arrays.items():
        assert pd[name].shape == (8, values.shape[2]), name
        np.testing.assert_array_equal(pd[name], values[0])
        for suffix in exovista._model.component_suffixes(values.shape[2]):
            assert name + suffix not in pd
    scalars = ["partial_X", "partial_Z", "taken", "taken_X", "taken_Y", "num_0",
               "num_1", "mixed_0", "T"]
    for name in scalars:
        assert pd[name].shape == (8,), name
        np.testing.assert_array_equal(pd[name], model.node_vars[name][0])
    np.testing.assert_array_equal(
        pd["mixed"], np.c_[model.node_vars["mixed_X"][0], model.node_vars["mixed_Y"][0]])
    assert sorted(pd.keys()) == sorted(list(arrays) + scalars + ["mixed"])
    # combined arrays sit where their first component was
    assert list(pd.keys())[:5] == ["vel", "uv", "sym", "four", "five"]

    # element components: NaN where a component is undefined (truth table)
    s = grid.cell_data["s"]
    assert s.shape == (3, 3)
    np.testing.assert_array_equal(s[0, :2], [model.elem_vars["s_X"][1][0, 0],
                                             model.elem_vars["s_Y"][1][0, 0]])
    assert np.isnan(s[0, 2])
    np.testing.assert_array_equal(s[1:, 2], model.elem_vars["s_Z"][2][0])
    np.testing.assert_array_equal(grid.cell_data["p"][1:], [np.nan, np.nan])
    # global variables are never combined
    np.testing.assert_array_equal(grid.field_data["g_X"], [1.0])


def test_combined_vector_matches_vtk_reader(tmp_path):
    """VTK's reader groups _X/_Y/_Z into one array too. (It crashes on a
    vector element variable whose components have different truth tables,
    like "s" above, so this file has none.)"""
    model, arrays = _vector_model()
    node_vars = {k: v for k, v in model.node_vars.items() if k.startswith("vel")}
    model = ExodusModel("vec", 3, model.coords, model.blocks, times=model.times,
                        node_vars=node_vars)
    path = tmp_path / "vel.exo"
    write_model(path, model)
    reader = pv.ExodusIIReader(str(path))
    reader.enable_all_point_arrays()
    reader.reader.SetSqueezePoints(False)
    reader.set_active_time_point(1)
    vtk_block = reader.read()["Element Blocks"][0]
    (vtk_name,) = vtk_block.point_data.keys()
    grid = rp.read_exo(path)
    assert list(grid.point_data.keys()) == ["vel"]
    np.testing.assert_allclose(vtk_block.point_data[vtk_name], grid.point_data["vel"])
    np.testing.assert_array_equal(grid.point_data["vel"], arrays["vel"][1])


def test_combine_components_off(tmp_path):
    model, _ = _vector_model()
    write_model(tmp_path / "vec.exo", model)
    grid = rp.read_exo(tmp_path / "vec.exo", combine_components=False)
    assert list(grid.point_data.keys()) == list(model.node_vars)
    assert all(grid.point_data[k].shape == (8,) for k in grid.point_data.keys())
    assert set(model.elem_vars) <= set(grid.cell_data.keys())


def test_variable_selection(tmp_path):
    model, arrays = _vector_model()
    path = tmp_path / "vec.exo"
    write_model(path, model)
    grid = rp.read_exo(path, read_node_variables=["T", "vel", "vel_X"],
                       read_element_variables="s_Z", read_global_variables=False)
    assert list(grid.point_data.keys()) == ["T", "vel", "vel_X"]
    assert grid.point_data["vel"].shape == (8, 3)
    np.testing.assert_array_equal(grid.point_data["vel_X"], arrays["vel"][1][:, 0])
    assert set(grid.cell_data.keys()) == {"exo_block_id", "exo_block_name",
                                          "exo_elem_id", "s_Z"}
    assert set(grid.field_data.keys()) == {"times", "time", "num_dim"}

    grid = rp.read_exo(path, read_node_variables=False,
                       read_element_variables=None, read_global_variables="g_Y")
    assert len(grid.point_data.keys()) == 0
    assert set(grid.field_data.keys()) == {"times", "time", "num_dim", "g_Y"}

    with pytest.raises(KeyError, match="nope"):
        rp.read_exo(path, read_node_variables=["T", "nope"])
    with pytest.raises(KeyError):
        rp.read_exo(path, read_element_variables="vel")  # a node variable
    with pytest.raises(KeyError):
        rp.read_exo(path, read_global_variables="g")  # globals stay scalar
    with pytest.raises(KeyError):  # combined names exist only when combining
        rp.read_exo(path, read_node_variables="vel", combine_components=False)


def test_global_variables(steps_file):
    path, model, _ = steps_file
    grid = rp.read_exo(path, time_step=-2)
    value = grid.field_data["g"]
    assert value.shape == (1,) and value.dtype == np.float64
    assert value[0] == model.global_vars["g"][2]
    grid = rp.read_exo(path, read_global_variables=False)
    assert "g" not in grid.field_data


def test_reserved_names_are_not_overwritten(tmp_path):
    coords = np.eye(4, 3)
    model = ExodusModel(
        "r", 3, coords, [Block(5, "t", "TETRA4", [[0, 1, 2, 3]])],
        times=[2.0],
        elem_vars={"exo_block_id": {5: [[99.0]]}, "ok": {5: [[1.0]]}},
        global_vars={"time": [123.0], "num_dim": [7.0], "gg": [1.0]},
    )
    write_model(tmp_path / "r.exo", model)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        grid = rp.read_exo(tmp_path / "r.exo")
    messages = [str(w.message) for w in record]
    assert len(messages) == 2
    assert "exo_block_id" in messages[0] and "read_element_fields" in messages[0]
    assert "'time'" in messages[1] and "read_global_fields" in messages[1]
    np.testing.assert_array_equal(grid.cell_data["exo_block_id"], [5])
    np.testing.assert_array_equal(grid.cell_data["ok"], [1.0])
    np.testing.assert_array_equal(grid.field_data["time"], [2.0])
    np.testing.assert_array_equal(grid.field_data["num_dim"], [3])
    np.testing.assert_array_equal(grid.field_data["gg"], [1.0])
    # still available through the history readers
    assert rp.read_element_fields(tmp_path / "r.exo")["exo_block_id"][0, 0] == 99
    assert rp.read_global_fields(tmp_path / "r.exo")["time"][0] == 123.0


def test_read_side_sets_flag(tmp_path):
    coords = np.eye(4, 3)
    model = ExodusModel("s", 3, coords, [Block(1, "t", "TETRA4", [[0, 1, 2, 3]])],
                        side_sets=[SideSet(4, "base", [0], [4])])
    write_model(tmp_path / "s.exo", model)
    assert isinstance(rp.read_exo(tmp_path / "s.exo"), pv.UnstructuredGrid)
    grid, side_sets = rp.read_exo(tmp_path / "s.exo", read_side_sets=True)
    assert isinstance(grid, pv.UnstructuredGrid)
    assert isinstance(side_sets, pv.MultiBlock)
    assert side_sets.keys() == ["base"]
    assert side_sets["base"].n_points == grid.n_points


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses /proc")
def test_files_are_closed(steps_file):
    path = os.path.realpath(steps_file[0])

    def open_files():
        fds = os.listdir("/proc/self/fd")
        out = set()
        for fd in fds:
            try:
                out.add(os.readlink(f"/proc/self/fd/{fd}"))
            except OSError:
                pass
        return out

    before = open_files()
    rp.read_exo(path, read_side_sets=True)
    rp.read_side_sets(path)
    rp.read_node_sets(path)
    rp.read_node_fields(path)
    rp.read_element_fields(path)
    rp.read_global_fields(path)
    with pytest.raises(IndexError):
        rp.read_exo(path, time_step=0)
    assert path not in open_files() - before


def test_public_api():
    for name in rp.__all__:
        assert getattr(exovista.read_pyvista, name)
    assert exovista.read_exo is rp.read_exo
