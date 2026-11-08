"""Time histories: read_node_fields, read_element_fields, read_global_fields."""

import numpy as np
import pytest

from exovista import read_pyvista as rp
from exovista._backend import write_model
from exovista._model import Block, ExodusModel, component_suffixes

N_STEPS = 3


def _model(n_steps=N_STEPS):
    rng = np.random.default_rng(2)
    coords = np.vstack([np.eye(4, 3), np.eye(4, 3) + 2, np.eye(4, 3) + 4])
    blocks = [
        Block(1, "a", "TETRA4", [[0, 1, 2, 3], [4, 5, 6, 7]]),
        Block(2, "b", "TETRA4", [[8, 9, 10, 11]]),
    ]
    times = np.linspace(0.0, 1.0, n_steps)
    vel = rng.random((n_steps, 12, 3))
    node_vars = {"T": rng.random((n_steps, 12))}
    for k, suffix in enumerate(component_suffixes(3)):
        node_vars["vel" + suffix] = vel[..., k]
    stress = rng.random((n_steps, 3, 6))
    elem_vars = {
        "e": {1: rng.random((n_steps, 2)), 2: rng.random((n_steps, 1))},
        "only_b": {2: rng.random((n_steps, 1))},
    }
    for k, s in enumerate(component_suffixes(6)):
        elem_vars["stress" + s] = {1: stress[:, :2, k], 2: stress[:, 2:, k]}
    model = ExodusModel(
        "fields", 3, coords, blocks, elem_id_map=[3, 1, 2], times=times,
        node_vars=node_vars, elem_vars=elem_vars,
        global_vars={"energy": rng.random(n_steps), "dt": rng.random(n_steps)},
    )
    return model, vel, stress


@pytest.fixture
def fields_file(tmp_path):
    model, vel, stress = _model()
    path = tmp_path / "fields.exo"
    write_model(path, model)
    return path, model, vel, stress


def test_read_node_fields(fields_file):
    path, model, vel, _ = fields_file
    fields = rp.read_node_fields(path)
    assert list(fields) == list(model.node_vars)
    for name, values in fields.items():
        assert values.shape == (N_STEPS, 12) and values.dtype == np.float64
        np.testing.assert_array_equal(values, model.node_vars[name])

    combined = rp.read_node_fields(path, combine_components=True)
    assert list(combined) == ["T", "vel"]
    assert combined["vel"].shape == (N_STEPS, 12, 3)
    np.testing.assert_array_equal(combined["vel"], vel)

    assert list(rp.read_node_fields(path, names="vel_Y")) == ["vel_Y"]
    sel = rp.read_node_fields(path, names=["vel", "T"], combine_components=True)
    assert list(sel) == ["vel", "T"]
    with pytest.raises(KeyError):
        rp.read_node_fields(path, names="vel")  # not combining
    with pytest.raises(KeyError):
        rp.read_node_fields(path, names=["T", "missing"])
    assert rp.read_node_fields(path, names=[]) == {}


def test_read_element_fields(fields_file):
    path, model, _, stress = fields_file
    # ids [3, 1, 2] in file order -> original order: file elements 1, 2, 0
    order = [1, 2, 0]

    def history(name):
        per_block = model.elem_vars[name]
        cols = [per_block.get(1, np.full((N_STEPS, 2), np.nan)),
                per_block.get(2, np.full((N_STEPS, 1), np.nan))]
        return np.concatenate(cols, axis=1)

    fields = rp.read_element_fields(path)
    assert list(fields) == list(model.elem_vars)
    for name, values in fields.items():
        assert values.shape == (N_STEPS, 3)
        np.testing.assert_array_equal(values, history(name)[:, order])
    # not defined on block 1 (file elements 0 and 1 = original cells 2 and 0)
    assert np.isnan(fields["only_b"][:, [0, 2]]).all()
    assert not np.isnan(fields["only_b"][:, 1]).any()

    raw = rp.read_element_fields(path, names="e", original_order=False)
    np.testing.assert_array_equal(raw["e"], history("e"))

    combined = rp.read_element_fields(path, combine_components=True)
    assert list(combined) == ["e", "only_b", "stress"]
    assert combined["stress"].shape == (N_STEPS, 3, 6)
    np.testing.assert_array_equal(combined["stress"], stress[:, order, :])

    # the columns are the cells of read_exo with the same original_order
    for original_order in (True, False):
        grid = rp.read_exo(path, time_step=2, original_order=original_order)
        hist = rp.read_element_fields(path, original_order=original_order)
        for name in ("e", "only_b"):
            np.testing.assert_array_equal(grid.cell_data[name], hist[name][1])

    with pytest.raises(KeyError):
        rp.read_element_fields(path, names="T")


def test_read_global_fields(fields_file):
    path, model, _, _ = fields_file
    fields = rp.read_global_fields(path)
    assert list(fields) == ["energy", "dt"]
    for name, values in fields.items():
        assert values.shape == (N_STEPS,) and values.dtype == np.float64
        np.testing.assert_array_equal(values, model.global_vars[name])
    assert list(rp.read_global_fields(path, names="dt")) == ["dt"]
    with pytest.raises(KeyError):
        rp.read_global_fields(path, names="nope")


def test_histories_match_single_steps(fields_file):
    path = fields_file[0]
    nodes = rp.read_node_fields(path, combine_components=True)
    elems = rp.read_element_fields(path, combine_components=True)
    glob = rp.read_global_fields(path)
    for step in range(N_STEPS):
        grid = rp.read_exo(path, time_step=step + 1)
        for name, values in nodes.items():
            np.testing.assert_array_equal(grid.point_data[name], values[step])
        for name, values in elems.items():
            np.testing.assert_array_equal(grid.cell_data[name], values[step])
        for name, values in glob.items():
            np.testing.assert_array_equal(grid.field_data[name], values[step:step + 1])


def test_fields_of_file_without_variables(tmp_path):
    write_model(tmp_path / "s.exo", ExodusModel(
        "s", 3, np.eye(4, 3), [Block(1, "t", "TETRA4", [[0, 1, 2, 3]])]))
    assert rp.read_node_fields(tmp_path / "s.exo") == {}
    assert rp.read_element_fields(tmp_path / "s.exo") == {}
    assert rp.read_global_fields(tmp_path / "s.exo") == {}


def test_fields_of_data_files(datafile):
    path = datafile("noh.exo")
    nodes = rp.read_node_fields(path)
    assert set(nodes) == {"DISPLX", "DISPLY", "VELOCITY_X", "VELOCITY_Y"}
    assert all(v.shape == (31, 22) for v in nodes.values())
    combined = rp.read_node_fields(path, combine_components=True)
    assert list(combined) == ["DISPLX", "DISPLY", "VELOCITY"]
    assert combined["VELOCITY"].shape == (31, 22, 2)
    elems = rp.read_element_fields(path)
    assert all(v.shape == (31, 10) for v in elems.values())
    glob = rp.read_global_fields(path)
    assert len(glob) == 74 and all(v.shape == (31,) for v in glob.values())
