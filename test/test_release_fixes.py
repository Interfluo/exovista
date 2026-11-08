"""Regression tests for the findings of the 2.0 release review.

Covers the writer's storage layout (one time step per chunk), the atomic
write (read-only targets, errors on close), name decoding, "normal model"
files, write-back of reader output (num_dim, VTK's padded vectors, merged
nodes, global variables), recombined global variables, the title limit,
reporting of invalid input and the ``exovista-read`` command.
"""

import errno
import io
import os
import stat
import subprocess
import sys
import warnings

import netCDF4
import numpy as np
import pytest
import pyvista as pv

from exovista import nc as exo_nc
from exovista import (
    read_exo,
    read_global_fields,
    read_node_fields,
    read_node_sets,
    read_side_sets,
    write_exo,
)
from exovista._backend import open_reader, write_model
from exovista._model import Block, ExodusModel
from exovista.exoread import main as exoread_main
from exovista.util import stringify


def hexes(nx, ny, nz, origin=(0.0, 0.0, 0.0)):
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), origin=origin)
    return grid.cast_to_unstructured_grid()


def leftovers(directory):
    return [f for f in os.listdir(directory) if f.startswith(".exovista")]


# --------------------------------------------------------------------------- #
# storage layout: one time step per chunk
# --------------------------------------------------------------------------- #
def test_time_dependent_variables_have_one_step_per_chunk(tmp_path):
    grid = hexes(4, 2, 2)
    grid.cell_data["region"] = np.repeat([1, 2], 8)
    n_steps = 7
    times = np.arange(n_steps, dtype=float)
    rng = np.random.default_rng(1)
    node = rng.random((n_steps, grid.n_points, 3))
    elem = rng.random((n_steps, grid.n_cells))
    path = tmp_path / "c.exo"
    write_exo(path, grid, times=times, node_fields={"u": node},
              element_fields={"e": elem}, global_fields={"k": times**2})
    with netCDF4.Dataset(path) as ds:
        timed = [name for name, var in ds.variables.items()
                 if var.dimensions[:1] == ("time_step",) and var.ndim > 1]
        assert {"vals_glo_var", "vals_nod_var1", "vals_elem_var1eb1",
                "vals_elem_var1eb2"} <= set(timed)
        for name in timed:
            chunks = ds[name].chunking()
            assert chunks[0] == 1, (name, chunks)
            assert list(chunks[1:]) == list(ds[name].shape[1:]), (name, chunks)
    back = read_exo(path, time_step=3)
    np.testing.assert_array_equal(back.point_data["u"], node[2])
    np.testing.assert_array_equal(back.cell_data["e"], elem[2])
    # uncompressed files stay contiguous
    write_exo(tmp_path / "u.exo", grid, times=times, node_fields={"u": node},
              compression_level=0)
    with netCDF4.Dataset(tmp_path / "u.exo") as ds:
        assert ds["vals_nod_var1"].chunking() == "contiguous"


def test_large_entity_dimension_is_split_into_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(exo_nc, "MAX_CHUNK_BYTES", 8 * 100)  # 100 doubles
    grid = hexes(9, 9, 2)  # 300 nodes
    values = np.random.default_rng(2).random((3, grid.n_points))
    write_exo(tmp_path / "s.exo", grid, times=[0.0, 1.0, 2.0],
              node_fields={"u": values})
    with netCDF4.Dataset(tmp_path / "s.exo") as ds:
        assert ds["vals_nod_var1"].chunking() == [1, 100]
    np.testing.assert_array_equal(
        read_node_fields(tmp_path / "s.exo")["u"], values
    )


# --------------------------------------------------------------------------- #
# atomic write: read-only targets and errors on close
# --------------------------------------------------------------------------- #
def test_read_only_target_is_not_replaced(tmp_path, monkeypatch):
    grid = hexes(1, 1, 1)
    path = tmp_path / "ro.exo"
    write_exo(path, grid, title="first")
    real_access = os.access

    def access(target, mode, *args, **kwargs):
        if mode == os.W_OK and os.path.realpath(target) == os.path.realpath(path):
            return False
        return real_access(target, mode, *args, **kwargs)

    monkeypatch.setattr(os, "access", access)
    with pytest.raises(PermissionError, match="not writable"):
        write_exo(path, grid, title="second")
    monkeypatch.undo()
    with netCDF4.Dataset(path) as ds:
        assert ds.title == "first"
    assert leftovers(tmp_path) == []


def test_read_only_target_on_disk(tmp_path):
    grid = hexes(1, 1, 1)
    path = tmp_path / "ro.exo"
    write_exo(path, grid, title="first")
    os.chmod(path, stat.S_IREAD)
    try:
        if os.access(path, os.W_OK):
            pytest.skip("running with privileges that override file permissions")
        with pytest.raises(PermissionError):
            write_exo(path, grid, title="second")
        with netCDF4.Dataset(path) as ds:
            assert ds.title == "first"
        assert leftovers(tmp_path) == []
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def test_error_on_close_is_raised_and_target_kept(tmp_path, monkeypatch):
    grid = hexes(1, 1, 1)
    path = tmp_path / "guard.exo"
    write_exo(path, grid, title="good original")

    class FailingClose:
        """A netCDF4.Dataset whose close() fails after closing (as an HDF5
        flush error at close would). A proxy: subclassing Dataset crashes."""

        def __init__(self, dataset):
            object.__setattr__(self, "_dataset", dataset)

        def __getattr__(self, name):
            return getattr(self._dataset, name)

        def __setattr__(self, name, value):
            setattr(self._dataset, name, value)

        def close(self):
            self._dataset.close()
            raise OSError(errno.EIO, "Input/output error (simulated flush)")

    real_open = exo_nc.open

    def open_(filename, mode="r"):
        handle = real_open(filename, mode)
        return FailingClose(handle) if mode == "w" else handle

    monkeypatch.setattr(exo_nc, "open", open_)
    with pytest.raises(OSError, match="simulated flush"):
        write_exo(path, grid, title="new")
    monkeypatch.undo()
    with netCDF4.Dataset(path) as ds:
        assert ds.title == "good original"
    assert leftovers(tmp_path) == []


# --------------------------------------------------------------------------- #
# names end at their first NUL
# --------------------------------------------------------------------------- #
def _overwrite_name(ds, var, row, name):
    width = ds[var].shape[1]
    raw = np.frombuffer(name.encode() + b"\0", dtype="S1")
    assert raw.size < width
    ds[var][row, : raw.size] = raw


def test_names_end_at_the_first_nul(tmp_path):
    """The Exodus library renames in place: ex_put_name writes only
    strlen(name) + 1 bytes, so the tail of a longer old name follows the NUL."""
    grid = hexes(2, 1, 1)
    surface = pv.PolyData(  # the x = 0 face
        np.array([[0.0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]), [4, 0, 1, 2, 3]
    )
    path = tmp_path / "n.exo"
    write_exo(path, grid, surface, block_names="block_long_name",
              side_set_names="side_set_long_name",
              node_sets={"node_set_long_name": [0, 1]},
              element_fields={"temperature_long": np.ones(2)},
              node_fields={"velocity_long": np.ones(grid.n_points)},
              global_fields={"energy_long": [1.0]})
    with netCDF4.Dataset(path, "a") as ds:
        for var, name in (("eb_names", "abc"), ("ss_names", "ss"),
                          ("ns_names", "ns"), ("name_elem_var", "temp"),
                          ("name_nod_var", "vel"), ("name_glo_var", "en")):
            _overwrite_name(ds, var, 0, name)
    mesh = read_exo(path)
    assert set(mesh.cell_data["exo_block_name"]) == {"abc"}
    assert "temp" in mesh.cell_data.keys() and "vel" in mesh.point_data.keys()
    assert "en" in mesh.field_data.keys()
    assert read_exo(path, read_element_variables="temp").cell_data["temp"].size == 2
    assert read_side_sets(path).keys() == ["ss"]
    assert list(read_node_sets(path)) == ["ns"]
    # the generic (non-S1) path of stringify cuts at the NUL too
    assert stringify(np.array(["a", "b", "", "c"])) == "ab"
    assert stringify(np.array([b"ab", b"c\0d", b"e"], dtype="S3")) == "abc"


# --------------------------------------------------------------------------- #
# "normal model" files: one combined vals_nod_var
# --------------------------------------------------------------------------- #
def _chars(ds, var, names, width):
    out = np.zeros((len(names), width), "S1")
    for i, name in enumerate(names):
        out[i, : len(name)] = np.frombuffer(name.encode(), dtype="S1")
    ds[var][:] = out


def _normal_model_file(path, nodal_values=True):
    """A HEX8 file in the Exodus "normal model" layout (file_size = 0):
    combined ``coord`` and ``vals_nod_var(time_step, num_nod_var, num_nodes)``."""
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    values = np.stack([np.stack([np.arange(8.0) + 100 * s, -np.arange(8.0) - s])
                       for s in range(2)])
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.api_version = np.float32(4.01)
        ds.version = np.float32(3.01)
        ds.floating_point_word_size = np.int32(8)
        ds.file_size = np.int32(0)
        ds.title = "normal model"
        for name, size in (("len_string", 33), ("len_line", 81), ("four", 4),
                           ("len_name", 33), ("num_dim", 3), ("num_nodes", 8),
                           ("num_elem", 1), ("num_el_blk", 1),
                           ("num_el_in_blk1", 1), ("num_nod_per_el1", 8),
                           ("num_nod_var", 2), ("time_step", None)):
            ds.createDimension(name, size)
        ds.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0, 1.0]
        ds.createVariable("coor_names", "S1", ("num_dim", "len_name"))
        _chars(ds, "coor_names", ["x", "y", "z"], 33)
        ds.createVariable("coord", "f8", ("num_dim", "num_nodes"))[:] = coords.T
        prop = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
        prop.setncattr("name", "ID")
        prop[:] = [5]
        ds.createVariable("eb_status", "i4", ("num_el_blk",))[:] = [1]
        conn = ds.createVariable("connect1", "i4",
                                 ("num_el_in_blk1", "num_nod_per_el1"))
        conn.elem_type = "HEX8"
        conn[:] = [np.arange(1, 9)]
        ds.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))
        _chars(ds, "name_nod_var", ["A", "B"], 33)
        if nodal_values:
            ds.createVariable(
                "vals_nod_var", "f8", ("time_step", "num_nod_var", "num_nodes")
            )[:] = values
    return values


def test_normal_model_nodal_variables(tmp_path):
    path = tmp_path / "normal.exo"
    values = _normal_model_file(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mesh = read_exo(path, time_step=1)
        histories = read_node_fields(path)
    np.testing.assert_array_equal(mesh.point_data["A"], values[0, 0])
    np.testing.assert_array_equal(mesh.point_data["B"], values[0, 1])
    np.testing.assert_array_equal(read_exo(path).point_data["A"], values[1, 0])
    np.testing.assert_array_equal(histories["A"], values[:, 0])
    np.testing.assert_array_equal(histories["B"], values[:, 1])


def test_nodal_variable_without_values_warns(tmp_path):
    path = tmp_path / "novalues.exo"
    _normal_model_file(path, nodal_values=False)
    with pytest.warns(UserWarning, match="'[AB]' has no stored values"):
        mesh = read_exo(path)
    assert np.isnan(mesh.point_data["A"]).all()


# --------------------------------------------------------------------------- #
# write-back of reader output
# --------------------------------------------------------------------------- #
def _plate_with_vectors(tmp_path):
    plate = pv.Plane(i_resolution=2, j_resolution=2).cast_to_unstructured_grid()
    plate = plate.copy()
    for name in list(plate.point_data.keys()):
        del plate.point_data[name]
    plate.point_data["vel"] = np.c_[plate.points[:, :2], np.zeros(plate.n_points)]
    path = tmp_path / "shell.exo"
    write_exo(path, plate, num_dim=3)
    return path


def _num_dim_types_vars(path):
    with open_reader(path) as r:
        return r.num_dim, [b.elem_type for b in r.blocks()], r.node_var_names()


def test_read_exo_output_of_a_3d_shell_model_stays_3d(tmp_path):
    path = _plate_with_vectors(tmp_path)
    grid = read_exo(path)
    assert grid.field_data["num_dim"][0] == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(tmp_path / "copy.exo", grid, region_key="exo_block_id")
    assert _num_dim_types_vars(tmp_path / "copy.exo") == (
        3, ["SHELL4"], ["vel_X", "vel_Y", "vel_Z"])


def test_vtk_output_of_a_3d_shell_model_keeps_vector_z_with_num_dim_3(tmp_path):
    path = _plate_with_vectors(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(tmp_path / "s3.exo", pv.read(path), num_dim=3)
    assert _num_dim_types_vars(tmp_path / "s3.exo") == (
        3, ["SHELL4"], ["vel_X", "vel_Y", "vel_Z"])
    # automatic num_dim cannot know: 2D, with a warning that says so
    with pytest.warns(UserWarning, match="pass num_dim=3 to write shells"):
        write_exo(tmp_path / "s2.exo", pv.read(path))
    assert _num_dim_types_vars(tmp_path / "s2.exo") == (
        2, ["QUAD4"], ["vel_X", "vel_Y"])


def test_flat_mesh_away_from_z0_warns(tmp_path):
    cut = pv.Plane(center=(0, 0, 1.5), i_resolution=2, j_resolution=2)
    with pytest.warns(UserWarning, match="flat at z = 1.5.*num_dim=3"):
        write_exo(tmp_path / "cut.exo", cut)
    np.testing.assert_array_equal(read_exo(tmp_path / "cut.exo").points[:, 2], 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(tmp_path / "cut3.exo", cut, num_dim=3)
    np.testing.assert_array_equal(read_exo(tmp_path / "cut3.exo").points[:, 2], 1.5)


def _contact_model(path, duplicate_in_block=False):
    """Two HEX8 blocks touching at x = 1 without sharing nodes; T is 0 on
    block 1 and 1 on block 2."""
    left = hexes(1, 1, 1)
    right = hexes(1, 1, 1, origin=(1.0, 0.0, 0.0))
    coords = np.vstack([left.points, right.points])
    conn1 = left.cell_connectivity.reshape(1, 8)[:, [0, 1, 3, 2, 4, 5, 7, 6]]
    conn2 = conn1 + 8
    temperature = np.r_[np.zeros(8), np.ones(8)]
    blocks = [Block(1, "left", "HEX8", conn1), Block(2, "right", "HEX8", conn2)]
    if duplicate_in_block:  # one block holding both bodies
        blocks = [Block(1, "both", "HEX8", np.vstack([conn1, conn2]))]
    model = ExodusModel("contact", 3, coords, blocks, times=[0.0],
                        node_vars={"T": temperature[None]})
    write_model(path, model)


def test_vtk_write_back_reports_merged_distinct_nodes(tmp_path):
    path = tmp_path / "contact.exo"
    _contact_model(path)
    with pytest.warns(UserWarning, match=r"\['T'\] have different values .*"
                      "merged by coordinates.*GenerateGlobalNodeIdArrayOn"):
        write_exo(tmp_path / "back.exo", pv.read(path))
    assert read_exo(tmp_path / "back.exo").n_points == 12  # merged, as warned

    path = tmp_path / "one_block.exo"
    _contact_model(path, duplicate_in_block=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        write_exo(tmp_path / "back1.exo", pv.read(path))
    messages = " | ".join(str(w.message) for w in record)
    assert "merged 4 point(s)" in messages and "same element block" in messages
    assert "['T'] have different values" in messages


def test_multiblock_write_back_without_conflicts_is_quiet(tmp_path):
    grid = hexes(2, 1, 1)
    grid.cell_data["region"] = [1, 2]
    grid.point_data["T"] = grid.points[:, 0].copy()
    write_exo(tmp_path / "m.exo", grid)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(tmp_path / "back.exo", pv.read(tmp_path / "m.exo"))
    assert read_exo(tmp_path / "back.exo").n_points == grid.n_points


# --------------------------------------------------------------------------- #
# global variables: recombined on read, written back from read_exo output
# --------------------------------------------------------------------------- #
def _globals_file(path):
    grid = hexes(1, 1, 1)
    gl = np.arange(9.0).reshape(3, 3)
    write_exo(path, grid, times=[0.0, 1.0, 2.0],
              global_fields={"g": gl, "e": [1.0, 2.0, 3.0]})
    return gl


def test_multi_component_globals_are_recombined(tmp_path):
    path = tmp_path / "gl.exo"
    gl = _globals_file(path)
    mesh = read_exo(path, time_step=2)
    np.testing.assert_array_equal(mesh.field_data["g"], gl[1:2])
    assert mesh.field_data["g"].shape == (1, 3)
    np.testing.assert_array_equal(mesh.field_data["e"], [2.0])
    picked = read_exo(path, read_global_variables="g").field_data
    assert "g" in picked.keys() and "e" not in picked.keys()
    raw = read_exo(path, combine_components=False).field_data
    assert {"g_X", "g_Y", "g_Z"} <= set(raw.keys()) and "g" not in raw.keys()
    assert sorted(read_global_fields(path)) == ["e", "g_X", "g_Y", "g_Z"]
    combined = read_global_fields(path, combine_components=True)
    np.testing.assert_array_equal(combined["g"], gl)
    np.testing.assert_array_equal(
        read_global_fields(path, names="g", combine_components=True)["g"], gl)


def test_read_exo_globals_are_written_back(tmp_path):
    path = tmp_path / "gl.exo"
    gl = _globals_file(path)
    grid = read_exo(path, time_step=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(tmp_path / "copy.exo", grid, region_key="exo_block_id")
    back = read_global_fields(tmp_path / "copy.exo", combine_components=True)
    np.testing.assert_array_equal(back["g"], gl[1:2])
    np.testing.assert_array_equal(back["e"], [2.0])
    # constant over the steps written; global_fields wins on a name
    write_exo(tmp_path / "copy2.exo", grid, region_key="exo_block_id",
              times=[0.0, 5.0], global_fields={"e": [7.0, 8.0]})
    back = read_global_fields(tmp_path / "copy2.exo", combine_components=True)
    np.testing.assert_array_equal(back["e"], [7.0, 8.0])
    np.testing.assert_array_equal(back["g"], np.repeat(gl[1:2], 2, axis=0))
    # field data that is not one value per variable is reported
    grid.field_data["history"] = np.arange(4.0)
    with pytest.warns(UserWarning, match="skipped field data array 'history'"):
        write_exo(tmp_path / "copy3.exo", grid, region_key="exo_block_id")
    # a plain mesh's field data is not written
    plain = hexes(1, 1, 1)
    plain.field_data["note"] = [1.0]
    write_exo(tmp_path / "plain.exo", plain)
    assert read_global_fields(tmp_path / "plain.exo") == {}


def test_vtk_reader_globals_are_reported(tmp_path):
    path = tmp_path / "gl.exo"
    _globals_file(path)
    data = pv.read(path)
    for leaf in data["Element Blocks"]:
        leaf.field_data["energy"] = [1.0, 2.0, 3.0]  # as VTK stores histories
    with pytest.warns(UserWarning, match=r"global variables \['energy'\].*"
                      "global_fields="):
        write_exo(tmp_path / "back.exo", data)
    assert read_global_fields(tmp_path / "back.exo") == {}


# --------------------------------------------------------------------------- #
# title, version attributes, region_key, non-Exodus files, configuration
# --------------------------------------------------------------------------- #
def test_long_title_is_cut_to_80_bytes(tmp_path):
    grid = hexes(1, 1, 1)
    with pytest.warns(UserWarning, match="truncated the title to 80 bytes"):
        write_exo(tmp_path / "t.exo", grid, title="T" * 200)
    with netCDF4.Dataset(tmp_path / "t.exo") as ds:
        assert ds.title == "T" * 80
    with pytest.warns(UserWarning, match="truncated the title"):
        write_exo(tmp_path / "u.exo", grid, title="é" * 50)  # 100 bytes
    with netCDF4.Dataset(tmp_path / "u.exo") as ds:
        assert ds.title == "é" * 40
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(tmp_path / "v.exo", grid, title="x" * 80)


def test_version_attributes_are_nc_float(tmp_path):
    write_exo(tmp_path / "v.exo", hexes(1, 1, 1))
    with netCDF4.Dataset(tmp_path / "v.exo") as ds:
        for name in ("api_version", "version"):
            value = ds.getncattr(name)
            assert value.dtype == np.float32 and value == np.float32(5.03)


def test_misspelt_region_key_raises(tmp_path):
    grid = hexes(2, 1, 1)
    grid.cell_data["Material"] = [1, 2]
    with pytest.raises(KeyError, match="'material' is not a cell array.*Material"):
        write_exo(tmp_path / "rk.exo", grid, region_key="material")
    assert not (tmp_path / "rk.exo").exists()
    write_exo(tmp_path / "rk.exo", grid, region_key="Material")
    assert sorted(set(read_exo(tmp_path / "rk.exo").cell_data["exo_block_id"])) \
        == [1, 2]


def test_non_exodus_netcdf_file_is_rejected(tmp_path):
    path = tmp_path / "plain.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 3)
        ds.createVariable("v", "f8", ("x",))[:] = 1.0
    for read in (read_exo, read_node_fields, read_side_sets):
        with pytest.raises(ValueError, match="not an ExodusII file"):
            read(path)


@pytest.mark.parametrize("value, expected", [("12", 1), ("abc", 1), ("0", 0),
                                             ("9", 9)])
def test_compression_level_environment_is_validated(value, expected):
    env = dict(os.environ, EXODUSII_COMPRESSION_LEVEL=value)
    result = subprocess.run(
        [sys.executable, "-c",
         "from exovista.config import config; print(config.compression_level)"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-1] == str(expected)
    warned = "ignoring EXODUSII_COMPRESSION_LEVEL" in result.stderr
    assert warned == (value in ("12", "abc"))


# --------------------------------------------------------------------------- #
# exovista-read
# --------------------------------------------------------------------------- #
def _exoread_error(capsys, *argv):
    with pytest.raises(SystemExit) as exc:
        exoread_main(list(argv))
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("exovista-read: error: ") and "Traceback" not in err
    return err


def test_exoread_reports_errors_without_traceback(tmp_path, capsys):
    path = tmp_path / "m.exo"
    grid = hexes(1, 1, 1)
    grid.point_data["temperature"] = np.arange(8.0)
    write_exo(path, grid)
    assert "No such file" in _exoread_error(capsys, str(tmp_path / "none.exo"))
    assert "'bogus' not found" in _exoread_error(capsys, "-n", "bogus", str(path))
    assert "Index 99" in _exoread_error(
        capsys, "-i", "99", "-n", "temperature", str(path))
    plain = tmp_path / "plain.nc"
    with netCDF4.Dataset(plain, "w") as ds:
        ds.createDimension("x", 1)
    assert "not an ExodusII file" in _exoread_error(capsys, str(plain))
    garbage = tmp_path / "garbage.exo"
    garbage.write_bytes(b"not a netCDF file at all")
    _exoread_error(capsys, str(garbage))


def test_exoread_handles_a_closed_pipe(tmp_path):
    path = tmp_path / "m.exo"
    write_exo(path, hexes(1, 1, 1))

    class ClosedPipe(io.StringIO):
        def write(self, text):
            raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    assert exoread_main([str(path)], file=ClosedPipe()) == 1


def test_exoread_help_does_not_offer_coordinates(capsys):
    with pytest.raises(SystemExit):
        exoread_main(["--help"])
    out = capsys.readouterr().out
    assert "COORDINATES" not in out and "DISPLACEMENTS" not in out
