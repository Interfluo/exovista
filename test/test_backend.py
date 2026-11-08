"""Tests for exovista._model / exovista._backend and the core tweaks they need.

``pv.ExodusIIReader`` (VTK) is used as an independent oracle and netCDF4 to
inspect the files directly.
"""

import glob
import os
import subprocess
import sys
import tracemalloc

import meshio
import netCDF4
import numpy as np
import pytest
import pyvista as pv

import exovista
from exovista import _backend, nc
from exovista._backend import open_reader, read_model, write_model
from exovista._model import Block, ExodusModel, NodeSet, SideSet
from exovista.file import exodusii_file
from exovista.parallel_file import parallel_exodusii_file

LONG = "a_block_name_that_is_much_longer_than_thirty_two_characters_" * 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hex_tet_model(**overrides):
    """Two blocks (HEX8 + 2 TETRA4) sharing nodes, with every feature set."""
    rng = np.random.default_rng(42)
    coords = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [2, 0, 0], [2, 1, 0],
        ],
        dtype=float,
    )
    coords += rng.normal(scale=1e-3, size=coords.shape)  # not float32-exact
    blocks = [
        Block(10, LONG, "HEX8", [[0, 1, 2, 3, 4, 5, 6, 7]]),
        Block(20, "tets", "TETRA4", [[1, 8, 9, 5], [1, 9, 2, 5]]),
    ]
    times = np.array([0.0, 0.1, 0.25])
    n_steps = len(times)
    kwargs = dict(
        title="backend test",
        num_dim=3,
        coords=coords,
        blocks=blocks,
        side_sets=[
            SideSet(1, "left" + "_x" * 30, [0], [4]),
            SideSet(7, "", [1, 2, 0], [1, 2, 6]),
        ],
        node_sets=[
            NodeSet(3, "bottom", [0, 1, 2, 3, 8, 9]),
            NodeSet(4, "n" * 100, [9]),
        ],
        elem_id_map=np.array([3, 1, 2]),
        times=times,
        node_vars={
            "temp": rng.normal(size=(n_steps, 10)),
            "pressure_with_a_long_name_" * 3: rng.normal(size=(n_steps, 10)),
        },
        elem_vars={
            "stress": {10: rng.normal(size=(n_steps, 1)), 20: rng.normal(size=(3, 2))},
            "hex_only": {10: rng.normal(size=(n_steps, 1))},
        },
        global_vars={"energy": rng.normal(size=n_steps), "dt": rng.normal(size=3)},
        qa_records=[("exovista", "2.0.0", "2026-11-08", "12:00:00")],
        info=["written by test_backend"],
    )
    kwargs.update(overrides)
    return ExodusModel(**kwargs)


def _assert_models_equal(a, b, exact=True):
    check = np.testing.assert_array_equal if exact else np.testing.assert_allclose
    assert a.title == b.title
    assert a.num_dim == b.num_dim
    check(a.coords, b.coords)
    assert b.coords.dtype == np.float64
    assert [(x.id, x.name, x.elem_type) for x in a.blocks] == [
        (x.id, x.name, x.elem_type) for x in b.blocks
    ]
    for x, y in zip(a.blocks, b.blocks):
        np.testing.assert_array_equal(x.conn, y.conn)
    assert [(s.id, s.name) for s in a.side_sets] == [
        (s.id, s.name) for s in b.side_sets
    ]
    for x, y in zip(a.side_sets, b.side_sets):
        np.testing.assert_array_equal(x.elems, y.elems)
        np.testing.assert_array_equal(x.sides, y.sides)
    assert [(s.id, s.name) for s in a.node_sets] == [
        (s.id, s.name) for s in b.node_sets
    ]
    for x, y in zip(a.node_sets, b.node_sets):
        np.testing.assert_array_equal(x.nodes, y.nodes)
    if a.elem_id_map is None:
        assert b.elem_id_map is None
    else:
        np.testing.assert_array_equal(a.elem_id_map, b.elem_id_map)
    check(a.times, b.times)
    assert list(a.node_vars) == list(b.node_vars)
    for name in a.node_vars:
        check(a.node_vars[name], b.node_vars[name])
    assert list(a.elem_vars) == list(b.elem_vars)
    for name in a.elem_vars:
        assert sorted(a.elem_vars[name]) == sorted(b.elem_vars[name])
        for bid in a.elem_vars[name]:
            check(a.elem_vars[name][bid], b.elem_vars[name][bid])
    assert list(a.global_vars) == list(b.global_vars)
    for name in a.global_vars:
        check(a.global_vars[name], b.global_vars[name])


# --------------------------------------------------------------------------- #
# write_model -> open_reader round trips
# --------------------------------------------------------------------------- #
def test_roundtrip_float64_exact(tmp_path):
    model = _hex_tet_model()
    path = tmp_path / "rt.exo"
    write_model(path, model)
    back = read_model(path)
    _assert_models_equal(model, back, exact=True)
    assert back.qa_records == model.qa_records
    assert back.info == model.info


def test_reader_api(tmp_path):
    model = _hex_tet_model()
    write_model(tmp_path / "rt.exo", model)
    with open_reader(tmp_path / "rt.exo") as r:
        assert r.num_dim == 3
        assert r.title == "backend test"
        assert (r.num_nodes, r.num_elem, r.num_steps) == (10, 3, 3)
        assert r.float_word_size == 8
        assert not r.is_parallel
        assert r.node_var_names() == list(model.node_vars)
        assert r.elem_var_names() == ["stress", "hex_only"]
        assert r.global_var_names() == ["energy", "dt"]
        for step in range(3):
            np.testing.assert_array_equal(
                r.node_var("temp", step), model.node_vars["temp"][step]
            )
            np.testing.assert_array_equal(
                r.elem_var("stress", 20, step), model.elem_vars["stress"][20][step]
            )
            assert r.global_var("energy", step) == model.global_vars["energy"][step]
        # truth table: not defined on block 20
        assert r.elem_var("hex_only", 20) is None
        assert r.elem_var("hex_only", 20, 1) is None
        with pytest.raises(KeyError):
            r.node_var("nope")
        with pytest.raises(KeyError):
            r.elem_var("stress", 99)
        with pytest.raises(IndexError):
            r.node_var("temp", 3)
        with pytest.raises(IndexError):
            r.global_var("dt", -1)


def test_vtk_oracle_reads_backend_file(tmp_path):
    model = _hex_tet_model()
    path = str(tmp_path / "oracle.exo")
    write_model(path, model)
    reader = pv.ExodusIIReader(path)
    assert reader.element_blocks.names == [LONG, "tets"]
    assert reader.side_sets.names[0] == "left" + "_x" * 30
    assert reader.node_sets.names == ["bottom", "n" * 100]
    np.testing.assert_allclose(reader.time_values, model.times)
    reader.enable_all_cell_arrays()
    reader.enable_all_point_arrays()
    reader.reader.SetSqueezePoints(False)  # keep every node in every block
    reader.reader.GenerateGlobalElementIdArrayOn()
    reader.set_active_time_point(2)
    blocks = reader.read()["Element Blocks"]
    hexes, tets = blocks[0], blocks[1]
    np.testing.assert_array_equal(hexes.points, model.coords)
    np.testing.assert_array_equal(hexes.point_data["temp"], model.node_vars["temp"][2])
    np.testing.assert_array_equal(
        tets.cell_data["stress"], model.elem_vars["stress"][20][2]
    )
    assert "hex_only" in hexes.cell_data
    assert "hex_only" not in tets.cell_data  # element variable truth table
    # VTK reports the element id map as GlobalElementId
    np.testing.assert_array_equal(
        np.concatenate([hexes["GlobalElementId"], tets["GlobalElementId"]]), [3, 1, 2]
    )


def test_truth_table_written(tmp_path):
    write_model(tmp_path / "tt.exo", _hex_tet_model())
    with netCDF4.Dataset(tmp_path / "tt.exo") as ds:
        np.testing.assert_array_equal(ds["elem_var_tab"][:], [[1, 1], [1, 0]])
        assert "vals_elem_var2eb2" not in ds.variables
        names = netCDF4.chartostring(ds["coor_names"][:])
        assert [str(n).strip() for n in names] == ["x", "y", "z"]
        assert ds.maximum_name_length >= len(LONG)


def test_float32_option(tmp_path):
    model = _hex_tet_model()
    write_model(tmp_path / "f4.exo", model, float_word_size=4)
    with netCDF4.Dataset(tmp_path / "f4.exo") as ds:
        assert ds.floating_point_word_size == 4
        assert ds["coordx"].dtype == np.float32
        assert ds["vals_nod_var1"].dtype == np.float32
        assert ds["time_whole"].dtype == np.float32
    back = read_model(tmp_path / "f4.exo")
    assert back.coords.dtype == np.float64
    np.testing.assert_array_equal(back.coords, model.coords.astype(np.float32))
    np.testing.assert_array_equal(
        back.node_vars["temp"], model.node_vars["temp"].astype(np.float32)
    )
    with open_reader(tmp_path / "f4.exo") as r:
        assert r.float_word_size == 4
    with pytest.raises(ValueError):
        write_model(tmp_path / "bad.exo", model, float_word_size=2)


@pytest.mark.parametrize("level", [0, 1, 6])
def test_compression_levels(tmp_path, level):
    write_model(tmp_path / "c.exo", _hex_tet_model(), compression_level=level)
    with netCDF4.Dataset(tmp_path / "c.exo") as ds:
        assert ds.data_model == "NETCDF4_CLASSIC"
        for name in ("coordx", "connect1", "vals_nod_var1", "vals_elem_var1eb1"):
            filters = ds[name].filters()
            assert filters["zlib"] is bool(level), name
            if level:
                assert filters["complevel"] == level
    _assert_models_equal(_hex_tet_model(), read_model(tmp_path / "c.exo"))


def test_compression_default_follows_config(tmp_path):
    write_model(tmp_path / "d.exo", _hex_tet_model())
    level = exovista.config.config.compression_level
    with netCDF4.Dataset(tmp_path / "d.exo") as ds:
        assert ds["coordx"].filters()["zlib"] is bool(level)
    with pytest.raises(ValueError):
        write_model(tmp_path / "bad.exo", _hex_tet_model(), compression_level=10)


def test_2d_model(tmp_path):
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0.5]], dtype=float)
    model = ExodusModel(
        "flat",
        2,
        coords,
        [
            Block(1, "quads", "QUAD4", [[0, 1, 2, 3]]),
            Block(2, "tris", "TRI3", [[1, 4, 2]]),
        ],
        side_sets=[SideSet(1, "right", [1], [1])],
        times=[1.5],
        node_vars={"u": [np.arange(5.0)]},
    )
    path = tmp_path / "flat.exo"
    write_model(path, model)
    with netCDF4.Dataset(path) as ds:
        assert ds.dimensions["num_dim"].size == 2
        assert "coordz" not in ds.variables
    back = read_model(path)
    _assert_models_equal(model, back)
    assert back.coords.shape == (5, 2)
    reader = pv.ExodusIIReader(str(path))
    reader.reader.SetSqueezePoints(False)
    reader.enable_all_point_arrays()
    blocks = reader.read()["Element Blocks"]
    assert [pv.CellType(b.celltypes[0]).name for b in blocks] == ["QUAD", "TRIANGLE"]
    np.testing.assert_array_equal(blocks[0].points[:, 2], 0.0)
    np.testing.assert_array_equal(blocks[0].points[:, :2], coords)
    np.testing.assert_array_equal(blocks[1].point_data["u"], np.arange(5.0))


def test_empty_sets_and_null_block(tmp_path):
    model = ExodusModel(
        "empty",
        3,
        np.eye(4, 3),
        [
            Block(1, "tet", "TETRA4", [[0, 1, 2, 3]]),
            Block(2, "null", "HEX8", np.zeros((0, 8), int)),
        ],
        side_sets=[SideSet(5, "none", [], [])],
        node_sets=[NodeSet(6, "none", [])],
        times=[0.0],
        elem_vars={"e": {1: [[1.0]]}},
    )
    path = tmp_path / "empty.exo"
    write_model(path, model)
    with open_reader(path) as r:
        blocks = r.blocks()
        assert [(b.id, b.name, b.n_elem) for b in blocks] == [
            (1, "tet", 1), (2, "null", 0)
        ]
        assert [(s.id, s.name, len(s.elems)) for s in r.side_sets()] == [(5, "none", 0)]
        assert [(s.id, s.name, len(s.nodes)) for s in r.node_sets()] == [(6, "none", 0)]
        assert r.elem_var("e", 2) is None
    grid = pv.ExodusIIReader(str(path)).read()["Element Blocks"]
    assert grid[0].n_cells == 1


def test_block_attributes(tmp_path):
    model = ExodusModel(
        "spheres",
        3,
        np.arange(9.0).reshape(3, 3),
        [Block(1, "balls", "SPHERE", [[0], [1], [2]], attributes=[0.5, 0.25, 1.0])],
    )
    write_model(tmp_path / "s.exo", model)
    (block,) = read_model(tmp_path / "s.exo").blocks
    np.testing.assert_array_equal(block.attributes, [[0.5], [0.25], [1.0]])


def test_write_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "target.exo"
    path.write_bytes(b"previous content")

    def boom(f, model):
        f.put_init("x", 3, 1, 0, 0, 0, 0)
        raise RuntimeError("disk full")

    monkeypatch.setattr(_backend, "_write", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        write_model(path, _hex_tet_model())
    assert path.read_bytes() == b"previous content"
    assert os.listdir(tmp_path) == ["target.exo"]


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(blocks=[Block(1, "b", "HEX8", [[0, 1, 2, 3, 4, 5, 6, 10]])]), "outside"),
        (dict(blocks=[Block(1, "b", "HEX8", [[0, 1, 2, 3]])]), "needs 8 nodes"),
        (dict(side_sets=[SideSet(1, "s", [0], [7])]), "6 sides"),
        (dict(side_sets=[SideSet(1, "s", [3], [1])]), "outside"),
        (dict(node_sets=[NodeSet(1, "n", [10])]), "outside"),
        (dict(elem_id_map=[1, 2]), "elem_id_map"),
        (dict(node_vars={"t": np.zeros((2, 10))}), "shape"),
        (dict(elem_vars={"e": {30: np.zeros((3, 1))}}), "unknown block"),
        (dict(global_vars={"g": np.zeros(4)}), "shape"),
        (dict(times=[]), "times is empty"),
        (
            dict(blocks=[Block(1, "a", "TETRA4", [[0, 1, 2, 3]])] * 2, elem_vars={}),
            "duplicate",
        ),
        (dict(coords=np.zeros((10, 2))), "coords"),
    ],
)
def test_validation(tmp_path, overrides, match):
    if "blocks" in overrides:
        overrides.setdefault("elem_vars", {})
        overrides.setdefault("side_sets", [])
        overrides.setdefault("elem_id_map", None)
    with pytest.raises(ValueError, match=match):
        write_model(tmp_path / "bad.exo", _hex_tet_model(**overrides))
    assert not os.listdir(tmp_path)


def test_name_too_long(tmp_path):
    with pytest.raises(ValueError, match="256"):
        Block(1, "x" * 257, "HEX8", np.zeros((1, 8), int))
    model = _hex_tet_model(node_vars={"v" * 300: np.zeros((3, 10))})
    with pytest.raises(ValueError, match="256"):
        write_model(tmp_path / "never.exo", model)
    assert not os.listdir(tmp_path)


# --------------------------------------------------------------------------- #
# open_reader on existing files
# --------------------------------------------------------------------------- #
SERIAL_FILES = ["edges.base.exo", "mkmesh.gen", "noh.exo"]
PARALLEL = [
    ("edges.exo.4.*", "edges.base.exo", 738, 320),
    ("noh.exo.3.*", "noh.exo", 22, 10),
    ("mkmesh.par.2.*", "mkmesh.gen", 12, 5),
]


def _exercise_reader(r):
    coords = r.coords()
    assert coords.dtype == np.float64 and coords.shape == (r.num_nodes, r.num_dim)
    blocks = r.blocks()
    assert sum(b.n_elem for b in blocks) == r.num_elem
    for b in blocks:
        assert b.conn.size == 0 or 0 <= b.conn.min() <= b.conn.max() < r.num_nodes
    for ss in r.side_sets():
        assert len(ss.elems) == len(ss.sides)
        assert ss.elems.size == 0 or 0 <= ss.elems.min() <= ss.elems.max() < r.num_elem
    for ns in r.node_sets():
        assert ns.nodes.size == 0 or 0 <= ns.nodes.min() <= ns.nodes.max() < r.num_nodes
    n = r.num_steps
    for name in r.node_var_names()[:3]:
        assert r.node_var(name).shape == (n, r.num_nodes)
        if n:
            assert r.node_var(name, n - 1).shape == (r.num_nodes,)
    for name in r.elem_var_names()[:3]:
        for b in blocks:
            values = r.elem_var(name, b.id, 0 if n else None)
            assert values is None or values.shape[-1] == b.n_elem
    for name in r.global_var_names()[:3]:
        assert r.global_var(name).shape == (n,)
    return blocks


@pytest.mark.parametrize("name", SERIAL_FILES)
def test_open_reader_serial_data(datafile, name):
    with open_reader(datafile(name)) as r:
        _exercise_reader(r)
        assert not r.is_parallel


def test_open_reader_every_data_file_alone(datadir):
    """Each file in test/data (decomposition pieces too) reads on its own."""
    paths = sorted(glob.glob(os.path.join(datadir, "*")))
    assert len(paths) >= 12
    for path in paths:
        with open_reader(path) as r:
            assert not r.is_parallel
            _exercise_reader(r)


@pytest.mark.parametrize("pattern,base,n_nodes,n_elem", PARALLEL)
def test_open_reader_parallel_matches_serial(datadir, datafile, pattern, base,
                                             n_nodes, n_elem):
    files = sorted(glob.glob(os.path.join(datadir, pattern)))
    assert len(files) > 1
    with open_reader(*files) as r:
        assert r.is_parallel
        assert (r.num_nodes, r.num_elem) == (n_nodes, n_elem)
        _exercise_reader(r)
    # glob pattern form
    with open_reader(os.path.join(datadir, pattern)) as r:
        assert (r.num_nodes, r.num_elem) == (n_nodes, n_elem)

    joined = read_model(*files)
    serial = read_model(datafile(base))
    assert (joined.num_nodes, joined.num_elem) == (n_nodes, n_elem)
    np.testing.assert_array_equal(joined.coords, serial.coords)
    for a, b in zip(joined.blocks, serial.blocks):
        assert (a.id, a.n_elem) == (b.id, b.n_elem)
        assert a.elem_type.upper() == b.elem_type.upper()
        np.testing.assert_array_equal(a.conn, b.conn)
    for a, b in zip(joined.side_sets, serial.side_sets):
        assert a.id == b.id
        assert sorted(zip(a.elems, a.sides)) == sorted(zip(b.elems, b.sides))
    for a, b in zip(joined.node_sets, serial.node_sets):
        assert a.id == b.id
        np.testing.assert_array_equal(np.sort(a.nodes), np.sort(b.nodes))
    np.testing.assert_array_equal(joined.times, serial.times)
    for name in serial.node_vars:
        np.testing.assert_allclose(joined.node_vars[name], serial.node_vars[name],
                                   rtol=1e-6, atol=1e-6)
    for name in serial.elem_vars:
        for bid in serial.elem_vars[name]:
            np.testing.assert_allclose(joined.elem_vars[name][bid],
                                       serial.elem_vars[name][bid], rtol=1e-6)
    for name in serial.global_vars:
        np.testing.assert_allclose(joined.global_vars[name],
                                   serial.global_vars[name], rtol=1e-6)


def test_open_reader_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        with open_reader(tmp_path / "nope.exo"):
            pass
    with pytest.raises(FileNotFoundError):
        with open_reader(str(tmp_path / "nope.exo.*")):
            pass


def test_open_reader_meshio_file(tmp_path):
    """meshio stores coordinates in a combined 'coord' variable."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=float,
    ) * 0.3
    mesh = meshio.Mesh(
        points,
        [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
         ("tetra", np.array([[0, 1, 2, 4]]))],
        point_data={"T": np.arange(8.0) / 7},
    )
    path = str(tmp_path / "meshio.exo")
    meshio.write(path, mesh, file_format="exodus")
    with netCDF4.Dataset(path) as ds:
        assert "coord" in ds.variables and "coordx" not in ds.variables
    with open_reader(path) as r:
        np.testing.assert_array_equal(r.coords(), points)
        blocks = r.blocks()
        assert [b.elem_type for b in blocks] == ["HEX8", "TETRA"]
        assert [b.name for b in blocks] == ["", ""]
        np.testing.assert_array_equal(blocks[1].conn, [[0, 1, 2, 4]])
        assert r.node_var_names() == ["T"]
        np.testing.assert_array_equal(r.node_var("T", 0), np.arange(8.0) / 7)
    # the legacy API reads it too
    with exodusii_file(path) as f:
        np.testing.assert_array_equal(f.get_coords(), points)
        assert f.get_element_block_name(f.get_element_block_ids()[0]) == ""


# --------------------------------------------------------------------------- #
# single time-step reads
# --------------------------------------------------------------------------- #
def _multi_step_model(n_steps=6, n_nodes=40):
    rng = np.random.default_rng(7)
    coords = rng.random((n_nodes, 3))
    conn = np.arange(n_nodes - n_nodes % 4).reshape(-1, 4)
    return ExodusModel(
        "steps", 3, coords, [Block(1, "tets", "TETRA4", conn)],
        times=np.linspace(0.0, 1.0, n_steps),
        node_vars={
            "a": rng.random((n_steps, n_nodes)),
            "b": rng.random((n_steps, n_nodes)),
        },
        elem_vars={"e": {1: rng.random((n_steps, len(conn)))}},
        global_vars={"g": rng.random(n_steps), "h": rng.random(n_steps)},
    )


def test_single_step_reads_match_history(tmp_path):
    model = _multi_step_model()
    write_model(tmp_path / "ms.exo", model)
    with open_reader(tmp_path / "ms.exo") as r:
        for step in range(model.n_steps):
            np.testing.assert_array_equal(
                r.node_var("b", step), model.node_vars["b"][step]
            )
            np.testing.assert_array_equal(
                r.elem_var("e", 1, step), model.elem_vars["e"][1][step]
            )
            assert r.global_var("h", step) == model.global_vars["h"][step]


def test_single_step_reads_slice_netcdf(tmp_path, monkeypatch):
    model = _multi_step_model()
    write_model(tmp_path / "ms.exo", model)
    calls = []
    original = nc.get_variable

    def spy(fh, name, default=None, raw=False, index=None):
        if not raw and name.startswith("vals_"):
            calls.append((name, index))
        return original(fh, name, default=default, raw=raw, index=index)

    monkeypatch.setattr(nc, "get_variable", spy)
    with open_reader(tmp_path / "ms.exo") as r:
        r.node_var("b", 4)
        r.elem_var("e", 1, 2)
        r.global_var("h", 5)
    assert calls == [
        ("vals_nod_var2", 4),
        ("vals_elem_var1eb1", 2),
        ("vals_glo_var", (5, 1)),
    ]


def test_single_step_read_memory(tmp_path):
    n_steps, n_nodes = 25, 50_000
    coords = np.zeros((n_nodes, 3))
    coords[:, 0] = np.arange(n_nodes)
    history = np.broadcast_to(np.arange(n_nodes, dtype=float), (n_steps, n_nodes))
    model = ExodusModel(
        "big", 3, coords, [Block(1, "pts", "SPHERE", np.arange(n_nodes)[:, None])],
        times=np.arange(n_steps, dtype=float), node_vars={"v": history},
    )
    write_model(tmp_path / "big.exo", model, compression_level=0)
    full_bytes = n_steps * n_nodes * 8
    with open_reader(tmp_path / "big.exo") as r:
        r.node_var_names()
        tracemalloc.start()
        try:
            values = r.node_var("v", n_steps - 1)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    np.testing.assert_array_equal(values, history[-1])
    assert peak < full_bytes / 5, (peak, full_bytes)


@pytest.mark.parametrize("parallel", [False, True])
def test_legacy_time_step_semantics(datadir, datafile, parallel):
    """1-based, negative from the end, 0 and out of range raise IndexError."""
    if parallel:
        f = parallel_exodusii_file(*glob.glob(os.path.join(datadir, "noh.exo.3.*")))
    else:
        f = exodusii_file(datafile("noh.exo"))
    with f:
        n = f.num_times()
        assert n == 31
        node_name = f.get_node_variable_names()[0]
        elem_name = f.get_element_variable_names()[0]
        glo_name = f.get_global_variable_names()[0]
        block_id = f.get_element_block_ids()[0]
        nodes = f.get_node_variable_values(node_name)
        elems = f.get_element_variable_values(block_id, elem_name)
        glo = f.get_global_variable_values(glo_name)
        for ts in (1, 2, n, -1, -2, -n):
            index = ts - 1 if ts > 0 else ts
            np.testing.assert_array_equal(
                f.get_node_variable_values(node_name, time_step=ts), nodes[index]
            )
            np.testing.assert_array_equal(
                f.get_element_variable_values(block_id, elem_name, time_step=ts),
                elems[index],
            )
            assert f.get_global_variable_value(glo_name, ts) == glo[index]
        for bad in (0, n + 1, -n - 1):
            with pytest.raises(IndexError):
                f.get_node_variable_values(node_name, time_step=bad)
            with pytest.raises(IndexError):
                f.get_element_variable_values(block_id, elem_name, time_step=bad)
            with pytest.raises(IndexError):
                f.get_global_variable_value(glo_name, bad)


# --------------------------------------------------------------------------- #
# core (legacy API) tweaks
# --------------------------------------------------------------------------- #
def _write_legacy(path, **kwargs):
    with exodusii_file(str(path), mode="w", **kwargs) as f:
        f.put_init("legacy", 2, 4, 1, 1, 2, 0)
        f.put_coords(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
        f.put_element_block(1, "QUAD4", 1, 4)
        f.put_element_conn(1, np.array([[1, 2, 3, 4]]))
        f.put_element_id_map([42])
        f.put_node_set_param(10, 2)
        f.put_node_set_name(10, "bottom_nodes_with_a_rather_long_name_here")
        f.put_node_set_nodes(10, [1, 2])
        f.put_node_set_param(20, 3)
        f.put_node_set_name(20, "three")
        f.put_node_set_nodes(20, [2, 3, 4])
        f.put_time(1, 0.0)
        f.put_time(2, 1.0)
        f.put_global_variable_params(2)
        f.put_global_variable_names(["g1", "g2"])
        f.put_global_variable_values(1, [1.0, 2.0])
        f.put_global_variable_values(2, [3.0, 4.0])
        f.put_node_set_variable_params(1)
        f.put_node_set_variable_names(["nsv"])
        f.put_node_set_variable_values(1, 10, "nsv", [1.0, 2.0])
        f.put_node_set_variable_values(2, 20, "nsv", [3.0, 4.0, 5.0])


def test_legacy_writer_overrides(tmp_path):
    _write_legacy(tmp_path / "a.exo", float_word_size=4, compression_level=0)
    with netCDF4.Dataset(tmp_path / "a.exo") as ds:
        assert ds.floating_point_word_size == 4
        assert ds["coordx"].dtype == np.float32
        assert not ds["coordx"].filters()["zlib"]
    _write_legacy(tmp_path / "b.exo", float_word_size=8, compression_level=5)
    with netCDF4.Dataset(tmp_path / "b.exo") as ds:
        assert ds["coordx"].dtype == np.float64
        assert ds["coordx"].filters()["complevel"] == 5
    with pytest.raises(ValueError):
        exodusii_file(str(tmp_path / "c.exo"), mode="w", float_word_size=3)
    with pytest.raises(ValueError):
        exodusii_file(str(tmp_path / "c.exo"), mode="w", compression_level=-1)
    with pytest.raises(ValueError):
        exodusii_file(str(tmp_path / "c.exo"), mode="x")


def test_legacy_env_defaults(tmp_path):
    """EXODUSII_FLOAT_WORD_SIZE / EXODUSII_COMPRESSION_LEVEL still apply."""
    code = (
        "import numpy as np, sys\n"
        "from exovista.file import exodusii_file\n"
        "with exodusii_file(sys.argv[1], mode='w') as f:\n"
        "    f.put_init('env', 1, 2, 0, 0, 0, 0)\n"
        "    f.put_coords(np.array([[0.0], [1.0]]))\n"
    )
    env = dict(os.environ, EXODUSII_FLOAT_WORD_SIZE="4", EXODUSII_COMPRESSION_LEVEL="0")
    path = str(tmp_path / "env.exo")
    subprocess.run([sys.executable, "-c", code, path], env=env, check=True)
    with netCDF4.Dataset(path) as ds:
        assert ds.floating_point_word_size == 4
        assert ds["coordx"].dtype == np.float32
        assert not ds["coordx"].filters()["zlib"]


def test_legacy_maps_sets_globals(tmp_path):
    _write_legacy(tmp_path / "l.exo")
    with netCDF4.Dataset(tmp_path / "l.exo") as ds:
        # node set variables live on each set's own node-count dimension
        assert ds["vals_nset_var1ns1"].dimensions == ("time_step", "num_nod_ns1")
        assert ds["vals_nset_var1ns2"].dimensions == ("time_step", "num_nod_ns2")
    with exodusii_file(str(tmp_path / "l.exo")) as f:
        np.testing.assert_array_equal(f.get_element_id_map(), [42])
        ns = f.get_node_set(10)
        assert ns.name == "bottom_nodes_with_a_rather_long_name_here"
        np.testing.assert_array_equal(ns.nodes, [1, 2])
        assert f.get_node_set_name(20) == "three"
        assert f.num_nodes_in_node_set(20) == 3
        assert list(f.get_global_variable_names()) == ["g1", "g2"]
        np.testing.assert_array_equal(f.get_global_variable_values("g2"), [2.0, 4.0])
        assert f.get_global_variable_value("g1", -1) == 3.0
        np.testing.assert_array_equal(f.get_all_global_variable_values(2), [3.0, 4.0])
        np.testing.assert_array_equal(
            f.get_node_set_variable_values(20, "nsv", 2), [3.0, 4.0, 5.0]
        )
    with open_reader(tmp_path / "l.exo") as r:
        np.testing.assert_array_equal(r.elem_id_map(), [42])
        assert [(s.id, s.nodes.tolist()) for s in r.node_sets()] == [
            (10, [0, 1]), (20, [1, 2, 3])
        ]
        np.testing.assert_array_equal(r.global_var("g1"), [1.0, 3.0])


def test_read_handles_are_closed(datadir, datafile, tmp_path):
    with open_reader(datafile("noh.exo")) as r:
        handles = list(r._f.files)
    assert handles and not any(h.isopen() for h in handles)

    files = glob.glob(os.path.join(datadir, "noh.exo.3.*"))
    with open_reader(*files) as r:
        handles = list(r._f.files)
    assert len(handles) == 3 and not any(h.isopen() for h in handles)

    with pytest.raises(RuntimeError):
        with open_reader(datafile("noh.exo")) as r:
            handles = list(r._f.files)
            raise RuntimeError
    assert not any(h.isopen() for h in handles)

    f = exodusii_file(datafile("noh.exo"))
    f.close()
    f.close()  # idempotent
    assert not f.fh.isopen()
    with parallel_exodusii_file(*files) as p:
        handles = list(p.files)
    assert not any(h.isopen() for h in handles)


def test_netcdf_fallback_removed():
    import importlib

    with pytest.raises(ImportError):
        importlib.import_module("exovista.netcdf")
    assert not hasattr(exovista.config.config, "use_netcdf4_if_possible")
    assert nc.library() == "netcdf4"
