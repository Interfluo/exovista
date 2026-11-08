"""read_exo on files from elsewhere: test/data (serial and parallel), meshio,
exovista 1.x, VTK's writer; and performance on a large mesh.

VTK's readers (``pv.ExodusIIReader`` / ``pv.PExodusIIReader``) are the oracle.
"""

import cProfile
import glob
import os
import pstats
import subprocess
import sys
import textwrap
import time
import warnings

import meshio
import netCDF4
import numpy as np
import pytest
import pyvista as pv

from _pvcompat import exodus_reader, p_exodus_reader, validate_mesh
from exovista import _celltypes as ct
from exovista import read_pyvista as rp
from exovista._backend import write_model
from exovista._model import Block, ExodusModel, SideSet

VTK = ct.VTK
DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
SERIAL = ["edges.base.exo", "mkmesh.gen", "noh.exo"]
PIECES = sorted(
    os.path.basename(p) for p in glob.glob(os.path.join(DATA, "*.[0-9]"))
)
PARALLEL = [
    ("edges.exo.4.*", "edges.base.exo", 738, 320),
    ("noh.exo.3.*", "noh.exo", 22, 10),
    ("mkmesh.par.2.*", "mkmesh.gen", 12, 5),
]


def _cell_rows(grid):
    return [tuple(grid.get_cell(i).point_ids) for i in range(grid.n_cells)]


# --------------------------------------------------------------------------- #
# test/data
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", SERIAL + PIECES)
def test_data_file_matches_vtk_reader(datafile, name):
    path = datafile(name)
    grid, side_sets = rp.read_exo(path, read_side_sets=True, original_order=False)
    assert grid.points.dtype == np.float64
    assert validate_mesh(grid, ["wrong_number_of_points", "invalid_point_references",
                                "non_finite_points"]).is_valid
    assert set(grid.celltypes) <= {VTK.QUAD, VTK.HEXAHEDRON}

    reader = exodus_reader(path)
    reader.reader.SetSqueezePoints(False)  # point ids are file node indices
    reader.disable_displacements()
    reader.enable_all_point_arrays()
    reader.enable_all_cell_arrays()
    n_steps = len(grid.field_data["times"])
    if n_steps:
        np.testing.assert_allclose(reader.time_values, grid.field_data["times"])
        reader.set_active_time_point(n_steps - 1)
    blocks = [b for b in reader.read()["Element Blocks"] if b is not None]
    rows = [r for b in blocks for r in _cell_rows(b)]
    assert rows == _cell_rows(grid)
    if blocks:
        np.testing.assert_allclose(blocks[0].points, grid.points, rtol=1e-6)
        # variables both readers keep as scalars
        common = set(blocks[0].point_data.keys()) & set(grid.point_data.keys())
        for key in sorted(common)[:10]:
            np.testing.assert_allclose(blocks[0].point_data[key],
                                       grid.point_data[key], rtol=1e-6)
        common = set(blocks[0].cell_data.keys()) & set(grid.cell_data.keys())
        for key in sorted(common - {"ObjectId"})[:10]:
            np.testing.assert_allclose(
                np.concatenate([b.cell_data[key] for b in blocks]),
                grid.cell_data[key], rtol=1e-6)
    for mesh in side_sets:
        cell_id = mesh.cell_data["cell_id"]
        for i in range(mesh.n_cells):
            cell = set(grid.get_cell(cell_id[i]).point_ids)
            assert set(mesh.get_cell(i).point_ids) <= cell


def _normalized(grid, side_sets):
    """Order-independent description of a grid and its side sets."""
    cells = {
        int(i): (row, int(b))
        for i, row, b in zip(grid.cell_data["exo_elem_id"], _cell_rows(grid),
                             grid.cell_data["exo_block_id"])
    }
    sets = {
        key: sorted(tuple(sorted(c)) for c in _cell_rows(mesh))
        for key, mesh in zip(side_sets.keys(), side_sets)
    }
    return cells, sets


@pytest.mark.parametrize("pattern,base,n_nodes,n_elem", PARALLEL)
def test_parallel_decompositions(datafile, pattern, base, n_nodes, n_elem):
    parts = sorted(glob.glob(os.path.join(DATA, pattern)))
    assert len(parts) > 1
    serial, serial_sets = rp.read_exo(datafile(base), read_side_sets=True)
    joined, joined_sets = rp.read_exo(*parts, read_side_sets=True)
    globbed = rp.read_exo(os.path.join(DATA, pattern), read_node_variables=False,
                          read_element_variables=False, read_global_variables=False)
    assert globbed.n_cells == n_elem
    np.testing.assert_array_equal(globbed.points, serial.points)
    assert _cell_rows(globbed) == _cell_rows(serial)
    for grid in (joined,):
        assert (grid.n_points, grid.n_cells) == (n_nodes, n_elem)
        np.testing.assert_array_equal(grid.points, serial.points)
        assert _cell_rows(grid) == _cell_rows(serial)
        for key in ("exo_block_id", "exo_elem_id"):
            np.testing.assert_array_equal(grid.cell_data[key], serial.cell_data[key])
        np.testing.assert_array_equal(grid.field_data["times"],
                                      serial.field_data["times"])
        assert grid.point_data.keys() == serial.point_data.keys()
        for key in serial.point_data.keys():
            np.testing.assert_allclose(grid.point_data[key], serial.point_data[key],
                                       rtol=1e-6, atol=1e-6)
        assert grid.cell_data.keys() == serial.cell_data.keys()
        for key in serial.cell_data.keys():
            if key != "exo_block_name":
                np.testing.assert_allclose(grid.cell_data[key], serial.cell_data[key],
                                           rtol=1e-6, atol=1e-6)
        for key in serial.field_data.keys():
            np.testing.assert_allclose(grid.field_data[key], serial.field_data[key],
                                       rtol=1e-6, atol=1e-6)
    assert validate_mesh(joined).is_valid
    assert _normalized(joined, joined_sets) == _normalized(serial, serial_sets)

    joined_ns = rp.read_node_sets(*parts)
    serial_ns = rp.read_node_sets(datafile(base))
    assert list(joined_ns) == list(serial_ns)
    for key in serial_ns:
        np.testing.assert_array_equal(np.sort(joined_ns[key]), np.sort(serial_ns[key]))

    names = rp.read_element_fields(datafile(base), names=[])  # nothing read
    assert names == {}
    serial_fields = rp.read_element_fields(datafile(base))
    fields = rp.read_element_fields(*parts, names=list(serial_fields)[:5])
    for key in fields:
        np.testing.assert_allclose(fields[key], serial_fields[key], rtol=1e-6,
                                   atol=1e-6)

    # VTK's parallel reader: the same cells, keyed by global element id
    reader = p_exodus_reader(parts[0])
    reader.disable_displacements()
    reader.reader.GenerateGlobalElementIdArrayOn()
    by_id = {}
    for block in reader.read()["Element Blocks"]:
        if block is None:
            continue
        for i, gid in enumerate(block.cell_data["GlobalElementId"]):
            by_id[int(gid)] = np.asarray(block.points)[block.get_cell(i).point_ids]
    assert sorted(by_id) == sorted(joined.cell_data["exo_elem_id"])
    for cell, gid in enumerate(joined.cell_data["exo_elem_id"]):
        np.testing.assert_allclose(
            np.asarray(joined.points)[joined.get_cell(cell).point_ids],
            by_id[int(gid)], rtol=1e-6, atol=1e-12,
        )


def test_parallel_permuted_element_ids(datafile, tmp_path):
    """Pieces whose global ids put the joined elements out of id order: the
    cells come back sorted by id (original_order) or in joined order."""
    import shutil

    serial = rp.read_exo(datafile("mkmesh.gen"))
    rows = _cell_rows(serial)
    n = serial.n_cells
    swap = {1: n, n: 1}
    copies = []
    for path in sorted(glob.glob(os.path.join(DATA, "mkmesh.par.2.*"))):
        copy = tmp_path / os.path.basename(path)
        shutil.copy(path, copy)
        with netCDF4.Dataset(copy, "a") as ds:
            ids = np.ma.getdata(ds["elem_num_map"][:])
            ds["elem_num_map"][:] = [swap.get(int(i), int(i)) for i in ids]
        copies.append(str(copy))
    expected = [rows[swap.get(i, i) - 1] for i in range(1, n + 1)]

    grid, side_sets = rp.read_exo(*copies, read_side_sets=True)
    np.testing.assert_array_equal(grid.cell_data["exo_elem_id"], np.arange(1, n + 1))
    assert _cell_rows(grid) == expected
    raw = rp.read_exo(*copies, original_order=False)
    assert list(raw.cell_data["exo_elem_id"]) != sorted(raw.cell_data["exo_elem_id"])
    assert sorted(_cell_rows(raw)) == sorted(rows)
    for mesh in side_sets:
        for i, cell in enumerate(mesh.cell_data["cell_id"]):
            assert set(mesh.get_cell(i).point_ids) <= set(grid.get_cell(cell).point_ids)


def test_single_piece_keeps_file_order(datafile):
    """A piece's element ids are global (not 1..n): no reordering."""
    grid = rp.read_exo(datafile("edges.exo.4.0"))
    with netCDF4.Dataset(datafile("edges.exo.4.0")) as ds:
        ids = np.ma.getdata(ds["elem_num_map"][:])
    np.testing.assert_array_equal(grid.cell_data["exo_elem_id"], ids)


# --------------------------------------------------------------------------- #
# meshio
# --------------------------------------------------------------------------- #
def test_meshio_file(tmp_path):
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0.5, 0.5, 2.0]],
    ) * 0.25
    rng = np.random.default_rng(4)
    mesh = meshio.Mesh(
        points,
        [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
         ("pyramid", np.array([[4, 5, 6, 7, 8]]))],
        point_data={"T": np.arange(9.0), "vel": rng.random((9, 3)),
                    "uv": rng.random((9, 2))},
    )
    path = tmp_path / "meshio.exo"
    meshio.write(str(path), mesh, file_format="exodus")
    with netCDF4.Dataset(path) as ds:
        assert "coord" in ds.variables and "coordx" not in ds.variables
    grid = rp.read_exo(path)
    np.testing.assert_array_equal(grid.points, points)
    assert list(grid.celltypes) == [VTK.HEXAHEDRON, VTK.PYRAMID]
    assert _cell_rows(grid) == [tuple(range(8)), (4, 5, 6, 7, 8)]
    assert list(grid.cell_data["exo_block_name"]) == ["", ""]
    assert validate_mesh(grid).is_valid
    assert list(grid.point_data.keys()) == ["T", "vel", "uv"]
    np.testing.assert_array_equal(grid.point_data["vel"], mesh.point_data["vel"])
    np.testing.assert_array_equal(grid.point_data["uv"], mesh.point_data["uv"])
    scalars = rp.read_exo(path, combine_components=False)
    assert list(scalars.point_data.keys()) == ["T", "vel_X", "vel_Y", "vel_Z",
                                               "uv_X", "uv_Y"]
    np.testing.assert_array_equal(rp.read_node_fields(path)["vel_Z"][0],
                                  mesh.point_data["vel"][:, 2])


def test_meshio_2d_file(tmp_path):
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0.5]]) * 0.5
    mesh = meshio.Mesh(points, [("quad", np.array([[0, 1, 2, 3]])),
                                ("triangle", np.array([[1, 4, 2]]))])
    path = tmp_path / "flat.exo"
    meshio.write(str(path), mesh, file_format="exodus")
    grid = rp.read_exo(path)
    np.testing.assert_array_equal(grid.field_data["num_dim"], [2])
    np.testing.assert_array_equal(grid.points[:, :2], points)
    np.testing.assert_array_equal(grid.points[:, 2], 0.0)
    assert list(grid.celltypes) == [VTK.QUAD, VTK.TRIANGLE]


# --------------------------------------------------------------------------- #
# exovista 1.x files
# --------------------------------------------------------------------------- #
def _str_rows(names, width=32):
    """Blank-padded char rows, as exovista 1.x (via the vendored core) wrote."""
    out = np.full((len(names), width), b" ", dtype="S1")
    for i, name in enumerate(names):
        raw = name.encode()[:width]
        out[i, : len(raw)] = np.frombuffer(raw, dtype="S1")
    return out


def _write_exovista_1x(path, coords, blocks, side_sets, node_vars, elem_vars,
                       times):
    """Write a file in the exact layout of exovista 1.3.0's write_exo.

    1.x always wrote float32 data (``floating_point_word_size = 4``), 32-char
    names, the title ``pyvista_mesh``, num_dim 3, no element id map, an
    element variable for every block without a truth table, and left the
    time value unwritten (netCDF fill) when no ``times`` were given. Its
    default output carries a helper node variable ``orig_pts``.

    blocks: [(name, elem_type, conn 1-based)], side_sets: [(name, elems
    1-based, sides)], node_vars: {name: (n_steps, n_nodes)}, elem_vars:
    {name: [(n_steps, n_blk) per block]}, times: array or None (one step).
    """
    n_steps = 1 if times is None else len(times)
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.setncattr("api_version", np.float64(5.0300002))
        ds.setncattr("version", np.float64(5.0300002))
        ds.setncattr("floating_point_word_size", np.int32(4))
        ds.setncattr("file_size", np.int32(1))
        ds.setncattr("title", "pyvista_mesh")
        ds.createDimension("time_step", None)
        for dim, size in [("len_name", 32), ("len_string", 32), ("len_line", 80),
                          ("four", 4), ("num_dim", 3),
                          ("num_nodes", len(coords)),
                          ("num_elem", sum(len(c) for _, _, c in blocks)),
                          ("num_el_blk", len(blocks)),
                          ("num_side_sets", len(side_sets))]:
            ds.createDimension(dim, size)
        kw = dict(zlib=True, complevel=1, shuffle=True)

        def var(name, dtype, dims, values=None):
            v = ds.createVariable(name, dtype, dims, **kw)
            if values is not None:
                v[:] = values
            return v

        time_whole = var("time_whole", "f4", ("time_step",))
        if times is not None:
            time_whole[:] = times
        var("eb_prop1", "i4", ("num_el_blk",), np.arange(1, len(blocks) + 1))
        ds["eb_prop1"].setncattr("name", "eb_prop1")
        var("eb_status", "i4", ("num_el_blk",), np.ones(len(blocks)))
        var("eb_names", "S1", ("num_el_blk", "len_string"),
            _str_rows([b[0] for b in blocks]))
        var("ss_prop1", "i4", ("num_side_sets",), np.arange(1, len(side_sets) + 1))
        ds["ss_prop1"].setncattr("name", "ss_prop1")
        var("ss_status", "i4", ("num_side_sets",), np.ones(len(side_sets)))
        var("ss_names", "S1", ("num_side_sets", "len_string"),
            _str_rows([s[0] for s in side_sets]))
        var("coor_names", "S1", ("num_dim", "len_string"),
            _str_rows(["coordx", "coordy", "coordz"]))
        for k, axis in enumerate("xyz"):
            var(f"coord{axis}", "f4", ("num_nodes",), coords[:, k])
        for i, (_, elem_type, conn) in enumerate(blocks, start=1):
            ds.createDimension(f"num_el_in_blk{i}", len(conn))
            ds.createDimension(f"num_nod_per_el{i}", len(conn[0]))
            v = var(f"connect{i}", "i4", (f"num_el_in_blk{i}", f"num_nod_per_el{i}"),
                    conn)
            v.setncattr("elem_type", elem_type)
        if node_vars:
            ds.createDimension("num_nod_var", len(node_vars))
            var("name_nod_var", "S1", ("num_nod_var", "len_string"),
                _str_rows(list(node_vars)))
            for i, values in enumerate(node_vars.values(), start=1):
                var(f"vals_nod_var{i}", "f4", ("time_step", "num_nodes"), values)
        if elem_vars:
            ds.createDimension("num_elem_var", len(elem_vars))
            var("name_elem_var", "S1", ("num_elem_var", "len_string"),
                _str_rows(list(elem_vars)))
            for i, per_block in enumerate(elem_vars.values(), start=1):
                for b, values in enumerate(per_block, start=1):
                    var(f"vals_elem_var{i}eb{b}", "f4",
                        ("time_step", f"num_el_in_blk{b}"), values)
        for i, (_, elems, sides) in enumerate(side_sets, start=1):
            ds.createDimension(f"num_side_ss{i}", len(elems))
            var(f"side_ss{i}", "i4", (f"num_side_ss{i}",), sides)
            var(f"elem_ss{i}", "i4", (f"num_side_ss{i}",), elems)
        assert ds.dimensions["time_step"].size in (0, n_steps)


def _legacy_hex_file(path, times):
    """What 1.3.0 wrote for a 2x2x1 hex grid with interleaved regions 0/1 and
    a surface split into bottom/top (dumped from a real 1.3.0 file)."""
    xs, ys, zs = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0],
                             indexing="ij")
    coords = np.column_stack([xs.ravel("F"), ys.ravel("F"), zs.ravel("F")])
    # block_names / side_set_names were given with times, defaulted without
    names = ["left", "right", "bottom", "top"] if times is not None else [
        "block_0", "block_1", "side_0", "side_1"]
    blocks = [(names[0], "HEX8", [[1, 2, 5, 4, 10, 11, 14, 13],
                                  [4, 5, 8, 7, 13, 14, 17, 16]]),
              (names[1], "HEX8", [[2, 3, 6, 5, 11, 12, 15, 14],
                                  [5, 6, 9, 8, 14, 15, 18, 17]])]
    side_sets = [(names[2], [1, 1, 1, 3, 3, 3, 2, 2, 4, 4, 2, 4],
                  [1, 5, 4, 1, 5, 2, 5, 4, 5, 2, 3, 3]),
                 (names[3], [1, 3, 2, 4], [6, 6, 6, 6])]
    n_steps = 1 if times is None else len(times)
    steps = np.zeros(n_steps) if times is None else np.asarray(times)
    node_vars = {"temp": np.tile(np.arange(18) * 1.5, (n_steps, 1)),
                 "orig_pts": np.tile(np.arange(18.0), (n_steps, 1))}
    elem_vars = {}
    if times is not None:
        node_vars["nf"] = np.outer(steps, np.arange(18.0))
        ef = np.outer(steps, np.arange(4.0))  # original cell order
        elem_vars["ef"] = [ef[:, [0, 2]], ef[:, [1, 3]]]
    _write_exovista_1x(path, coords, blocks, side_sets, node_vars, elem_vars,
                       times)
    return coords


def test_exovista_1x_file_with_fields(tmp_path):
    path = tmp_path / "old.exo"
    coords = _legacy_hex_file(path, times=[0.0, 0.5, 1.0])
    grid, side_sets = rp.read_exo(path, read_side_sets=True)
    assert grid.points.dtype == np.float64
    np.testing.assert_array_equal(grid.points, coords)
    np.testing.assert_array_equal(grid.field_data["times"], [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(grid.field_data["time"], [1.0])
    # no element id map: file order, ids 1..4
    np.testing.assert_array_equal(grid.cell_data["exo_elem_id"], [1, 2, 3, 4])
    assert list(grid.cell_data["exo_block_name"]) == ["left", "left", "right", "right"]
    np.testing.assert_array_equal(grid.cell_data["ef"], [0.0, 2.0, 1.0, 3.0])
    np.testing.assert_array_equal(grid.point_data["nf"], np.arange(18.0))
    np.testing.assert_array_equal(grid.point_data["temp"], np.arange(18) * 1.5)
    assert "orig_pts" in grid.point_data  # stored in the file: read as is
    assert validate_mesh(grid).is_valid
    assert side_sets.keys() == ["bottom", "top"]
    assert side_sets["bottom"].n_cells == 12 and side_sets["top"].n_cells == 4
    centers = side_sets["top"].cell_centers().points
    np.testing.assert_array_equal(centers[:, 2], 1.0)
    np.testing.assert_array_equal(
        rp.read_element_fields(path)["ef"], np.outer([0, 0.5, 1], [0, 2, 1, 3])
    )
    # VTK reads the same file
    reader = exodus_reader(str(path))
    assert reader.element_blocks.names == ["left", "right"]


def test_exovista_1x_static_file_time_is_zero(tmp_path):
    path = tmp_path / "static.exo"
    _legacy_hex_file(path, times=None)
    with netCDF4.Dataset(path) as ds:
        assert ds["time_whole"][:].mask.all()  # never written
    grid = rp.read_exo(path)
    np.testing.assert_array_equal(grid.field_data["times"], [0.0])
    np.testing.assert_array_equal(grid.field_data["time"], [0.0])
    np.testing.assert_array_equal(grid.point_data["temp"], np.arange(18) * 1.5)
    assert list(grid.point_data.keys()) == ["temp", "orig_pts"]
    assert rp.read_side_sets(path).keys() == ["side_0", "side_1"]


def test_unwritten_time_of_other_files_is_nan(tmp_path):
    path = tmp_path / "masked.exo"
    model = ExodusModel("other", 3, np.eye(4, 3),
                        [Block(1, "t", "TETRA4", [[0, 1, 2, 3]])],
                        times=[1.0], global_vars={"g": [2.0]})
    write_model(path, model)
    with netCDF4.Dataset(path, "a") as ds:
        ds["time_whole"][0] = np.ma.masked
    grid = rp.read_exo(path)
    assert np.isnan(grid.field_data["times"]).all()
    np.testing.assert_array_equal(grid.field_data["g"], [2.0])


def test_exovista_1x_triangle_side_sets_are_edges(tmp_path):
    """1.x wrote 2D triangle meshes as TRI3 in 3D files with *edge* side
    numbers; the Exodus convention would make sides 1-2 the two faces."""
    path = tmp_path / "tri.exo"
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    _write_exovista_1x(
        path, coords, [("block_0", "TRI3", [[1, 2, 3], [1, 3, 4]])],
        [("side_0", [1], [2]), ("side_1", [2], [2])],
        {"orig_pts": np.arange(4.0)[None]}, {}, None,
    )
    grid, side_sets = rp.read_exo(path, read_side_sets=True)
    assert list(grid.celltypes) == [VTK.TRIANGLE, VTK.TRIANGLE]
    assert [tuple(m.get_cell(0).point_ids) for m in side_sets] == [(1, 2), (2, 3)]
    assert all(m.GetNumberOfLines() == 1 for m in side_sets)
    # a TRI3 block of any other 3D file has shell sides: side 2 is a face
    model = ExodusModel("x", 3, coords, [Block(1, "t", "TRI3", [[0, 1, 2]])],
                        side_sets=[SideSet(1, "s", [0], [2])])
    write_model(tmp_path / "shell.exo", model)
    mesh = rp.read_side_sets(tmp_path / "shell.exo")["s"]
    assert mesh.GetNumberOfPolys() == 1


# --------------------------------------------------------------------------- #
# VTK's writer
# --------------------------------------------------------------------------- #
def test_vtk_written_file(tmp_path):
    """vtkExodusIIWriter output (lowercase types, velX-style names). The
    writer is run in a subprocess: it can crash after other VTK imports."""
    path = tmp_path / "vtk.exo"
    script = textwrap.dedent(f"""
        import numpy as np
        from vtkmodules.vtkIOExodus import vtkExodusIIWriter
        import pyvista as pv
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                        [0.5, 0.5, 2]], float)
        grid = pv.UnstructuredGrid(
            {{pv.CellType.HEXAHEDRON: np.arange(8)[None],
              pv.CellType.PYRAMID: np.array([[4, 5, 6, 7, 8]])}}, pts)
        grid.point_data["vel"] = pts * 2
        grid.cell_data["ElementBlockIds"] = np.array([1, 2], dtype=np.int32)
        writer = vtkExodusIIWriter()
        writer.SetFileName({str(path)!r})
        writer.SetInputData(grid)
        writer.WriteAllTimeStepsOff()
        writer.Write()
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            timeout=120)
    if result.returncode != 0 or not path.exists():
        pytest.skip(f"vtkExodusIIWriter failed: {result.stderr[-300:]!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grid = rp.read_exo(path)
    assert list(grid.celltypes) == [VTK.HEXAHEDRON, VTK.PYRAMID]
    np.testing.assert_array_equal(grid.cell_data["exo_block_id"], [1, 2])
    assert validate_mesh(grid).is_valid
    # VTK's suffixes have no underscore: not exovista's convention, not combined
    assert list(grid.point_data.keys()) == ["velX", "velY", "velZ"]
    for k, key in enumerate(["velX", "velY", "velZ"]):
        np.testing.assert_allclose(grid.point_data[key], grid.points[:, k] * 2)


# --------------------------------------------------------------------------- #
# performance
# --------------------------------------------------------------------------- #
def _hex_model(n, interleave):
    """n**3 hexes in two blocks with a bottom side set; ids record a shuffled
    original order when ``interleave``."""
    img = pv.ImageData(dimensions=(n + 1,) * 3, spacing=(1 / 64,) * 3)
    points = np.asarray(img.points)
    vox = img.cast_to_unstructured_grid()
    rows = vox.cells.reshape(-1, 9)[:, 1:][:, [0, 1, 3, 2, 4, 5, 7, 6]]
    cells = len(rows)
    region = (np.arange(cells) % 2) if interleave else (np.arange(cells) * 2 // cells)
    file_order = np.argsort(region, kind="stable")
    blocks = [Block(r + 1, f"region_{r}", "HEX8",
                    rows[file_order[region[file_order] == r]]) for r in (0, 1)]
    file_pos = np.empty(cells, int)
    file_pos[file_order] = np.arange(cells)
    bottom = np.flatnonzero(points[rows[:, 0], 2] == 0.0)
    times = np.array([0.0, 1.0])
    return ExodusModel(
        "big", 3, points, blocks,
        side_sets=[SideSet(1, "bottom", file_pos[bottom], np.full(len(bottom), 5))],
        elem_id_map=file_order + 1, times=times,
        node_vars={"T": np.vstack([points[:, 0], points[:, 1]])},
        elem_vars={"e": {b.id: np.ones((2, b.n_elem)) * b.id for b in blocks}},
        global_vars={"g": times},
    )


def _python_calls(path):
    profile = cProfile.Profile()
    profile.enable()
    rp.read_exo(path, read_side_sets=True)
    profile.disable()
    stats = pstats.Stats(profile)
    return sum(
        n_calls for (filename, _, _), (_, n_calls, *_) in stats.stats.items()
        if filename.endswith(".py")
    )


def test_no_per_cell_python_loops(tmp_path):
    """The number of Python calls does not grow with the number of cells."""
    calls = []
    for n in (6, 24):  # 216 vs 13824 cells
        path = tmp_path / f"h{n}.exo"
        write_model(path, _hex_model(n, interleave=False))
        rp.read_exo(path, read_side_sets=True)  # warm up
        calls.append(_python_calls(path))
    assert calls[1] < calls[0] * 1.2, calls


def test_large_mesh_performance(tmp_path):
    """~1M hexes with a boundary side set read within the 3 s budget."""
    path = tmp_path / "big.exo"
    model = _hex_model(100, interleave=True)
    write_model(path, model)
    start = time.perf_counter()
    grid, side_sets = rp.read_exo(path, read_side_sets=True)
    elapsed = time.perf_counter() - start
    assert grid.n_cells == 1_000_000
    assert side_sets["bottom"].n_cells == 10_000
    np.testing.assert_array_equal(grid.cell_data["exo_elem_id"][:3], [1, 2, 3])
    names = grid.cell_data["exo_block_name"]
    assert names[0] == "region_0" and names[1] == "region_1"
    np.testing.assert_array_equal(grid.cell_data["e"][:4], [1, 2, 1, 2])
    assert elapsed < 3.0, f"read_exo took {elapsed:.2f} s"
