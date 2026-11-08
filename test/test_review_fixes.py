"""Regression tests for the 2.0 verification findings (writer, reader, e2e).

Each test names the problem it guards against. VTK's reader
(``pv.ExodusIIReader``) is the independent oracle; reads that could crash VTK
run in a subprocess.
"""

import glob
import os
import stat
import subprocess
import sys
import time
import warnings

import meshio
import netCDF4
import numpy as np
import pytest
import pyvista as pv

from exovista import (
    read_element_fields,
    read_exo,
    read_global_fields,
    read_node_fields,
    read_node_sets,
    read_side_sets,
    write_exo,
)
from exovista._backend import open_reader, read_model, write_model
from exovista._model import Block, ExodusModel, SideSet, component_groups
from exovista.parallel_file import parallel_exodusii_file

DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
CT = pv.CellType


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def hexes(nx, ny, nz):
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
    return grid.cast_to_unstructured_grid()


def surface_of(mesh):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pv.PyVistaFutureWarning)
        return mesh.extract_surface(algorithm=None)


def face_keys(mesh):
    """Sorted corner coordinates of every cell (geometry, not ids), rounded to
    float32 (VTK's reader returns float32 points)."""
    mesh = mesh.cast_to_unstructured_grid()
    pts = np.asarray(mesh.points).astype(np.float32).astype(float)
    return sorted(
        tuple(sorted(map(tuple, pts[mesh.get_cell(i).point_ids].tolist())))
        for i in range(mesh.n_cells)
    )


def by_point(grid, name):
    """{point coordinates: values} of a point array."""
    values = np.asarray(grid.point_data[name]).reshape(grid.n_points, -1)
    points = np.asarray(grid.points).tolist()
    return {tuple(p): tuple(v) for p, v in zip(points, values.tolist())}


def vtk_reader(path, sets=True):
    reader = pv.ExodusIIReader(str(path))
    reader.enable_all_cell_arrays()
    reader.enable_all_point_arrays()
    if sets:
        reader.side_sets.enable_all()
        reader.node_sets.enable_all()
    return reader


def vtk_reads_in_subprocess(path):
    """Exit code of reading ``path`` with VTK in a child process (-11: crash)."""
    code = (
        "import sys, pyvista as pv\n"
        "r = pv.ExodusIIReader(sys.argv[1])\n"
        "r.enable_all_cell_arrays(); r.enable_all_point_arrays()\n"
        "mb = r.read()\n"
        "pv.read(sys.argv[1])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, timeout=120
    )
    return result.returncode


def truth_table(path):
    with netCDF4.Dataset(path) as ds:
        return np.asarray(ds["elem_var_tab"][:]).tolist()


def raw_exo(path, coords, blocks, side_sets=(), elem_map=None):
    """A minimal Exodus file written with netCDF4 directly.

    ``blocks``: dicts with id, type, name and either ``conn`` (2D, 0-based) or,
    for NSIDED, ``flat`` + ``counts``, or, for NFACED, ``faces`` (node lists),
    ``conn`` (face indices) + ``counts``.
    """
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET")
    ds.title = "raw"
    ds.api_version = np.float32(8.03)
    ds.version = np.float32(8.03)
    ds.floating_point_word_size = np.int32(8)
    ds.file_size = np.int32(1)
    ds.maximum_name_length = np.int32(32)
    for dim, size in [("len_string", 33), ("len_line", 81), ("four", 4),
                      ("len_name", 33), ("time_step", None),
                      ("num_dim", coords.shape[1]), ("num_nodes", len(coords)),
                      ("num_el_blk", len(blocks))]:
        ds.createDimension(dim, size)
    ds.createDimension("num_elem", sum(len(b["counts"]) if "counts" in b
                                       else len(b["conn"]) for b in blocks))
    ds.createVariable("time_whole", "f8", ("time_step",))
    status = ds.createVariable("eb_status", "i4", ("num_el_blk",))
    prop = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
    prop.setncattr("name", "ID")
    names = ds.createVariable("eb_names", "S1", ("num_el_blk", "len_name"))
    for i, b in enumerate(blocks):
        status[i] = 1
        prop[i] = b["id"]
        row = np.zeros(33, "S1")
        row[: len(b["name"])] = np.frombuffer(b["name"].encode(), "S1")
        names[i] = row
    for k, axis in enumerate("xyz"[: coords.shape[1]]):
        ds.createVariable("coord" + axis, "f8", ("num_nodes",))[:] = coords[:, k]
    for i, b in enumerate(blocks, start=1):
        if b["type"] == "NFACED":
            faces = b["faces"]
            flat = np.concatenate(faces)
            ds.createDimension("num_face", len(faces))
            ds.createDimension("num_fa_blk", 1)
            ds.createDimension("num_fa_in_blk1", len(faces))
            ds.createDimension("num_nod_per_fa1", len(flat))
            ds.createVariable("fa_status", "i4", ("num_fa_blk",))[:] = [1]
            fp = ds.createVariable("fa_prop1", "i4", ("num_fa_blk",))
            fp.setncattr("name", "ID")
            fp[:] = [10]
            fb = ds.createVariable("fbconn1", "i4", ("num_nod_per_fa1",))
            fb.elem_type = "nsided"
            fb[:] = flat + 1
            fe = ds.createVariable("fbepecnt1", "i4", ("num_fa_in_blk1",))
            fe.entity_type1, fe.entity_type2 = "NODE", "FACE"
            fe[:] = [len(f) for f in faces]
            ds.createDimension(f"num_el_in_blk{i}", len(b["counts"]))
            ds.createDimension(f"num_fac_per_el{i}", sum(b["counts"]))
            fc = ds.createVariable(f"facconn{i}", "i4", (f"num_fac_per_el{i}",))
            fc.elem_type = "NFACED"
            fc[:] = np.asarray(b["conn"]) + 1
            ee = ds.createVariable(f"ebepecnt{i}", "i4", (f"num_el_in_blk{i}",))
            ee.entity_type1, ee.entity_type2 = "FACE", "ELEM"
            ee[:] = b["counts"]
            continue
        if b["type"] == "NSIDED":
            flat = np.asarray(b["flat"])
            ds.createDimension(f"num_el_in_blk{i}", len(b["counts"]))
            ds.createDimension(f"num_nod_per_el{i}", len(flat))
            v = ds.createVariable(f"connect{i}", "i4", (f"num_nod_per_el{i}",))
            v.elem_type = "NSIDED"
            v[:] = flat + 1
            e = ds.createVariable(f"ebepecnt{i}", "i4", (f"num_el_in_blk{i}",))
            e.entity_type1, e.entity_type2 = "NODE", "ELEM"
            e[:] = b["counts"]
            continue
        conn = np.asarray(b["conn"])
        ds.createDimension(f"num_el_in_blk{i}", conn.shape[0])
        ds.createDimension(f"num_nod_per_el{i}", conn.shape[1])
        v = ds.createVariable(f"connect{i}", "i4",
                              (f"num_el_in_blk{i}", f"num_nod_per_el{i}"))
        v.elem_type = b["type"]
        v[:] = conn + 1
    if elem_map is not None:
        ds.createVariable("elem_num_map", "i4", ("num_elem",))[:] = elem_map
    if side_sets:
        ds.createDimension("num_side_sets", len(side_sets))
        st = ds.createVariable("ss_status", "i4", ("num_side_sets",))
        pr = ds.createVariable("ss_prop1", "i4", ("num_side_sets",))
        pr.setncattr("name", "ID")
        for i, (set_id, elems, sides) in enumerate(side_sets, start=1):
            st[i - 1] = 1
            pr[i - 1] = set_id
            ds.createDimension(f"num_side_ss{i}", len(elems))
            ds.createVariable(f"elem_ss{i}", "i4", (f"num_side_ss{i}",))[:] = (
                np.asarray(elems) + 1)
            ds.createVariable(f"side_ss{i}", "i4", (f"num_side_ss{i}",))[:] = sides
    ds.close()


# --------------------------------------------------------------------------- #
# writer: side-set matching tolerance
# --------------------------------------------------------------------------- #
def test_graded_mesh_side_sets_with_default_tolerance(tmp_path):
    """Default tolerance (1e-6 * diag) larger than the thinnest cells: every
    surface point maps to its nearest node, not to the lowest node in reach."""
    xs = np.linspace(0, 100, 5)
    ys = np.concatenate([[0.0, 1e-6, 3e-6, 1e-5], np.linspace(1, 100, 4)])
    vol = pv.RectilinearGrid(xs, ys, [0.0, 1.0]).cast_to_unstructured_grid()
    surf = surface_of(vol)
    path = tmp_path / "bl.exo"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(path, vol, surf)
    vtk_set = vtk_reader(path).read()["Side Sets"][0]
    assert face_keys(vtk_set) == face_keys(surf)
    _, sets = read_exo(path, read_side_sets=True)
    assert sets[0].n_cells == surf.n_cells == 78


def test_tolerance_larger_than_an_element(tmp_path):
    vol = hexes(3, 3, 3)
    surf = surface_of(vol)
    path = tmp_path / "big_tol.exo"
    write_exo(path, vol, surf, side_set_tolerance=1.2)
    assert face_keys(vtk_reader(path).read()["Side Sets"][0]) == face_keys(surf)


def test_surface_cell_collapsing_onto_one_node_raises(tmp_path):
    vol = hexes(2, 2, 2)
    # a sliver triangle whose corners are all near node (0, 0, 0)
    sliver = pv.PolyData(np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]),
                         faces=[3, 0, 1, 2])
    with pytest.raises(ValueError, match="same volume node"):
        write_exo(tmp_path / "x.exo", vol, sliver, side_set_tolerance=0.2)
    assert not (tmp_path / "x.exo").exists()


def test_quadratic_surface_error_suggests_unsplit_faces(tmp_path):
    tet10 = pv.UnstructuredGrid(
        {CT.QUADRATIC_TETRA: np.arange(10)[None]},
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0],
                  [.5, .5, 0], [0, .5, 0], [0, 0, .5], [.5, 0, .5], [0, .5, .5]],
                 dtype=float),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pv.PyVistaFutureWarning)  # `algorithm`
        split = tet10.extract_surface()  # the default splits quadratic faces
        unsplit = tet10.extract_surface(nonlinear_subdivision=0)
    with pytest.raises(ValueError, match=r"nonlinear_subdivision=0"):
        write_exo(tmp_path / "q.exo", tet10, split)
    write_exo(tmp_path / "q.exo", tet10, unsplit)
    _, sets = read_exo(tmp_path / "q.exo", read_side_sets=True)
    assert sets[0].n_cells == 4


# --------------------------------------------------------------------------- #
# writer: VTK reader output written back
# --------------------------------------------------------------------------- #
@pytest.fixture
def vtk_source(tmp_path):
    """Two hex blocks (ids 10, 20), a side set, a node set, vector variables."""
    grid = hexes(4, 1, 1)
    grid.cell_data["region"] = np.array([10, 10, 20, 20])
    grid.point_data["vel"] = np.c_[grid.points[:, 0], 2 * grid.points[:, 1],
                                   3 * grid.points[:, 2]]
    grid.point_data["T"] = grid.points[:, 0] + 0.5
    grid.cell_data["stress"] = np.arange(4 * 6, dtype=float).reshape(4, 6)
    grid.cell_data["e"] = np.arange(4.0)
    surf = surface_of(grid)
    surf.cell_data["region"] = np.where(surf.cell_centers().points[:, 0] < 0.5, 3, 4)
    path = tmp_path / "src.exo"
    write_exo(path, grid, surf, block_names={10: "left", 20: "right"},
              side_set_names={3: "inlet", 4: "walls"},
              node_sets={"origin": [0], "corners": [0, 4]})
    return path


@pytest.mark.parametrize("how", ["pv.read", "ExodusIIReader"])
def test_vtk_reader_output_written_back(tmp_path, vtk_source, how):
    src = vtk_source
    mb = pv.read(src) if how == "pv.read" else vtk_reader(src).read()
    out = tmp_path / "out.exo"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(out, mb)
    with open_reader(src) as a, open_reader(out) as b:
        assert [(x.id, x.name, x.elem_type, x.n_elem) for x in b.blocks()] == [
            (x.id, x.name, x.elem_type, x.n_elem) for x in a.blocks()]
        assert b.num_elem == 4
        # VTK's glued vectors get their original component names back
        assert sorted(b.node_var_names()) == sorted(a.node_var_names())
        assert sorted(b.elem_var_names()) == sorted(a.elem_var_names())
        assert [(s.id, s.name) for s in b.side_sets()] == [
            (s.id, s.name) for s in a.side_sets()]
        n_node_sets = len(b.node_sets())
    assert n_node_sets == (2 if how == "ExodusIIReader" else 0)
    g_src, s_src = read_exo(src, read_side_sets=True)
    g_out, s_out = read_exo(out, read_side_sets=True)
    # VTK's output has no node numbering (no GlobalNodeId): compare by point
    for name in ("vel", "T"):
        assert by_point(g_out, name) == by_point(g_src, name)
    for name in ("stress", "e", "exo_block_id", "exo_elem_id"):
        np.testing.assert_array_equal(g_out.cell_data[name], g_src.cell_data[name])
    for key in s_src.keys():
        assert face_keys(s_out[key]) == face_keys(s_src[key])
    if how == "ExodusIIReader":
        def node_set_points(path, grid):
            return {k: sorted(map(tuple, np.asarray(grid.points)[v].tolist()))
                    for k, v in read_node_sets(path).items()}

        assert node_set_points(out, g_out) == node_set_points(src, g_src)


def test_vtk_element_blocks_alone_keep_ids_and_names(tmp_path, vtk_source):
    blocks = pv.read(vtk_source)["Element Blocks"]
    out = tmp_path / "eb.exo"
    write_exo(out, blocks)
    with open_reader(out) as r:
        assert [(b.id, b.name) for b in r.blocks()] == [(10, "left"), (20, "right")]
        assert "vel_X" in r.node_var_names() and "vel__X" not in r.node_var_names()
        assert not r.side_sets()


def test_vtk_output_of_unnamed_2d_file(tmp_path):
    """VTK names unnamed entities "Unnamed block ID: 1" and pads the vectors of
    2D files to 3 components; neither leaks into the rewritten file."""
    coords = np.array([[0.0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    model = ExodusModel(
        "2d", 2, coords, [Block(7, "", "QUAD4", [[0, 1, 4, 3], [1, 2, 5, 4]])],
        side_sets=[SideSet(5, "", [0], [4])], times=[0.0],
        node_vars={"v_X": coords[None, :, 0], "v_Y": coords[None, :, 1] + 1},
        elem_vars={"q_X": {7: [[1.0, 2.0]]}, "q_Y": {7: [[3.0, 4.0]]},
                   "region": {7: [[5.0, 6.0]]}},  # an ordinary variable here
    )
    src = tmp_path / "unnamed.exo"
    write_model(src, model)
    out = tmp_path / "out.exo"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(out, vtk_reader(src).read())
    with open_reader(out) as r:
        assert r.num_dim == 2
        assert [(b.id, b.name) for b in r.blocks()] == [(7, "block_7")]
        assert [(s.id, s.name) for s in r.side_sets()] == [(5, "side_set_5")]
        assert sorted(r.node_var_names()) == ["v_X", "v_Y"]
        assert sorted(r.elem_var_names()) == ["q_X", "q_Y", "region"]
    g_src, g_out = read_exo(src), read_exo(out)
    assert by_point(g_out, "v") == by_point(g_src, "v")
    np.testing.assert_array_equal(g_out.cell_data["q"], g_src.cell_data["q"])


def test_vtk_global_node_ids_keep_coincident_nodes(tmp_path):
    """Two blocks with coincident but distinct interface nodes: with VTK's
    GlobalNodeId they stay apart (and keep the file's node order)."""
    left, right = hexes(1, 1, 1), hexes(1, 1, 1).translate((1, 0, 0))
    grid = left.merge(right, merge_points=False)
    grid.cell_data["region"] = [1, 2]
    src = tmp_path / "contact.exo"
    write_exo(src, grid)
    reader = vtk_reader(src)
    reader.reader.GenerateGlobalNodeIdArrayOn()
    write_exo(tmp_path / "out.exo", reader.read())
    back = read_exo(tmp_path / "out.exo")
    assert back.n_points == 16
    np.testing.assert_array_equal(back.points, read_exo(src).points)
    # by coordinates (no GlobalNodeId) the interface nodes are merged
    write_exo(tmp_path / "merged.exo", pv.read(src))
    assert read_exo(tmp_path / "merged.exo").n_points == 12


def test_vtk_implicit_id_arrays_not_written(tmp_path, vtk_source):
    reader = vtk_reader(vtk_source, sets=False)
    reader.reader.SetGenerateImplicitElementIdArray(True)
    reader.reader.SetGenerateImplicitNodeIdArray(True)
    blocks = reader.read()["Element Blocks"]
    assert "ImplicitNodeId" in blocks[0].point_data.keys()
    write_exo(tmp_path / "i.exo", blocks)
    with open_reader(tmp_path / "i.exo") as r:
        names = r.node_var_names() + r.elem_var_names()
    assert not [n for n in names if "Implicit" in n]


# --------------------------------------------------------------------------- #
# writer: NaN blocks and VTK's glued vectors
# --------------------------------------------------------------------------- #
def two_blocks():
    grid = hexes(2, 1, 1)
    grid.cell_data["region"] = np.array([1, 2])
    return grid


@pytest.mark.parametrize(
    "arrays,table",
    [
        # separate scalars VTK glues into one vector: one NaN component
        ({"U_X": [1.0, 2.0], "U_Y": [np.nan, 3.0], "U_Z": [4.0, 5.0]},
         [[1, 1, 1], [1, 1, 1]]),
        ({"u_X": [1.0, 2.0], "u_Y": [3.0, np.nan]}, [[1, 1], [1, 1]]),
        # VTK also glues names without "_" (aX, aY)
        ({"aX": [np.nan, 1.0], "aY": [2.0, 3.0]}, [[1, 1], [1, 1]]),
        # a whole family NaN on block 1 is left out there
        ({"U_X": [np.nan, 2.0], "U_Y": [np.nan, 3.0], "U_Z": [np.nan, 5.0]},
         [[0, 0, 0], [1, 1, 1]]),
        # unrelated variables are still decided on their own
        ({"p": [np.nan, 1.0], "q": [1.0, 2.0]}, [[0, 1], [1, 1]]),
    ],
)
def test_nan_blocks_decided_per_vtk_family(tmp_path, arrays, table):
    grid = two_blocks()
    for name, values in arrays.items():
        grid.cell_data[name] = np.array(values)
    path = tmp_path / "f.exo"
    write_exo(path, grid)
    assert truth_table(path) == table
    assert vtk_reads_in_subprocess(path) == 0
    back = read_exo(path, combine_components=False)
    for name, values in arrays.items():
        np.testing.assert_array_equal(back.cell_data[name], values)


def test_nan_vector_plus_completing_scalar(tmp_path):
    grid = two_blocks()
    grid.cell_data["U"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    grid.cell_data["U_Z"] = np.array([np.nan, 5.0])
    path = tmp_path / "f.exo"
    with pytest.warns(UserWarning, match=r"\['U'\] will not be recombined"):
        write_exo(path, grid)
    assert truth_table(path) == [[1, 1, 1], [1, 1, 1]]
    assert vtk_reads_in_subprocess(path) == 0


# --------------------------------------------------------------------------- #
# writer: MultiBlock arrays
# --------------------------------------------------------------------------- #
def test_multiblock_keeps_arrays_of_some_leaves(tmp_path):
    steel = hexes(1, 1, 1)
    water = hexes(1, 1, 1).translate((1, 0, 0))  # shares 4 nodes with steel
    steel.cell_data["plastic_strain"] = [0.25]
    steel.cell_data["temperature"] = [300.0]
    water.cell_data["temperature"] = [290.0]
    water.point_data["pressure"] = np.arange(8, dtype=float) + 1
    steel.point_data["vec"] = np.ones((8, 3))
    water.point_data["vec"] = np.ones((8, 2))  # component counts disagree
    path = tmp_path / "mb.exo"
    with pytest.warns(UserWarning, match="'vec'.*different numbers"):
        write_exo(path, pv.MultiBlock({"steel": steel, "water": water}))
    grid = read_exo(path)
    assert grid.n_points == 12
    np.testing.assert_array_equal(grid.cell_data["plastic_strain"], [0.25, np.nan])
    np.testing.assert_array_equal(grid.cell_data["temperature"], [300.0, 290.0])
    assert truth_table(path) == [[1, 1], [0, 1]] or truth_table(path) == [
        [1, 1], [1, 0]]
    # pressure: water's values on all of water's nodes (shared ones too), NaN
    # on steel's own nodes
    pressure = grid.point_data["pressure"]
    on_water = np.asarray(grid.points)[:, 0] >= 1
    assert np.isnan(pressure[~on_water]).all()
    expected = {tuple(p): v for p, v in zip(np.asarray(water.points),
                                            water.point_data["pressure"])}
    got = {tuple(p): v for p, v in zip(np.asarray(grid.points)[on_water],
                                       pressure[on_water])}
    assert got == expected
    assert "vec" not in grid.point_data.keys()


def test_multiblock_leaf_region_array_warns(tmp_path):
    leaf = hexes(2, 1, 1)
    leaf.cell_data["region"] = [1, 2]
    with pytest.warns(UserWarning, match="ignored the 'region' cell array"):
        write_exo(tmp_path / "one.exo", pv.MultiBlock({"only": leaf}))
    # a leaf whose region array matches the leaf (one value) is not reported
    leaf.cell_data["region"] = [5, 5]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(tmp_path / "one.exo", pv.MultiBlock({"only": leaf}))


# --------------------------------------------------------------------------- #
# writer: names, target file, precision, suffixes
# --------------------------------------------------------------------------- #
def test_names_are_nul_padded(tmp_path):
    grid = hexes(1, 1, 1)
    grid.point_data["T"] = np.arange(8.0)
    path = tmp_path / "n.exo"
    write_exo(path, grid, surface_of(grid), node_sets={"pins": [0, 1]})
    with netCDF4.Dataset(path) as ds:
        for var in ("name_nod_var", "eb_names", "ss_names", "ns_names",
                    "coor_names"):
            raw = np.ma.getdata(ds[var][0]).tobytes()
            assert b" " not in raw and raw.rstrip(b"\0") and raw.endswith(b"\0"), var
        assert netCDF4.chartostring(ds["name_nod_var"][:])[0] == "T"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = meshio.read(path, file_format="exodus")
    assert "T" in mesh.point_data and "pins" in mesh.point_sets


def test_target_file_long_name_symlink_and_permissions(tmp_path):
    grid = hexes(1, 1, 1)
    long_name = tmp_path / ("x" * 240 + ".exo")  # 244 bytes: a valid name
    write_exo(long_name, grid)
    assert long_name.exists()

    real = tmp_path / "real.exo"
    write_exo(real, grid)
    os.chmod(real, 0o600)
    link = tmp_path / "link.exo"
    link.symlink_to(real)
    grid2 = hexes(2, 1, 1)
    write_exo(link, grid2)
    assert link.is_symlink()
    assert read_exo(real).n_cells == 2
    assert stat.S_IMODE(os.stat(real).st_mode) == 0o600
    assert sorted(os.listdir(tmp_path)) == sorted(
        [long_name.name, "real.exo", "link.exo"])  # no temporary files left


def test_single_precision_overflow(tmp_path):
    grid = hexes(1, 1, 1)
    far = grid.copy()
    far.points[:, 0] *= 1e39
    with pytest.raises(ValueError, match="coordinates reach"):
        write_exo(tmp_path / "s.exo", far, precision="single")
    with pytest.raises(ValueError, match="times reach"):
        write_exo(tmp_path / "s.exo", grid, precision="single", times=[0.0, 1e39],
                  global_fields={"g": [0.0, 1.0]})
    assert not (tmp_path / "s.exo").exists()
    grid.point_data["big"] = np.full(8, 1e39)
    with pytest.warns(UserWarning, match="stored as \\+-inf.*'big'"):
        write_exo(tmp_path / "s.exo", grid, precision="single")
    assert np.isinf(read_exo(tmp_path / "s.exo").point_data["big"]).all()


def test_integers_beyond_float_precision_warn(tmp_path):
    grid = hexes(1, 1, 1)
    grid.cell_data["id"] = np.array([2**53 + 1], dtype=np.int64)
    with pytest.warns(UserWarning, match=r"'id' holds integers.*2\*\*53"):
        write_exo(tmp_path / "d.exo", grid)
    grid.cell_data["id"] = np.array([2**24 + 1], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_exo(tmp_path / "d.exo", grid)  # exact in double precision
    with pytest.warns(UserWarning, match=r"2\*\*24"):
        write_exo(tmp_path / "s.exo", grid, precision="single")
    del grid.cell_data["id"]
    with pytest.warns(UserWarning, match=r"node field 'n' holds integers"):
        write_exo(tmp_path / "s.exo", grid, precision="single",
                  node_fields={"n": np.full(8, 2**25 + 1)})


@pytest.mark.parametrize(
    "arrays,warns,expected",
    [
        ({"B": 3, "B_0": 1}, False, {"B": 3, "B_0": 1}),
        ({"Q": 4, "Q_X": 1}, False, {"Q": 4, "Q_X": 1}),
        ({"stress": 6, "stress_X": 1}, False, {"stress": 6, "stress_X": 1}),
        ({"T": 9, "T_X": 1}, False, {"T": 9, "T_X": 1}),
        ({"u": 3, "u_0": 1}, False, {"u": 3, "u_0": 1}),
        # ambiguous by construction: warned
        ({"V": 2, "V_Z": 1}, True, {"V": 3}),
        ({"q": 4, "q_4": 1}, True, {"q": 5}),
    ],
)
def test_component_suffix_collisions(tmp_path, arrays, warns, expected):
    grid = hexes(1, 1, 1)
    rng = np.random.default_rng(0)
    for name, n in arrays.items():
        grid.point_data[name] = rng.random((8, n)) if n > 1 else rng.random(8)
    path = tmp_path / "c.exo"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_exo(path, grid)
    assert any("will not be recombined" in str(w.message) for w in caught) == warns
    back = read_exo(path).point_data
    got = {k: (back[k].shape[1] if back[k].ndim == 2 else 1) for k in back.keys()}
    assert got == expected
    if not warns:
        for name in arrays:
            np.testing.assert_array_equal(back[name], grid.point_data[name])


def test_component_groups_rules():
    g = component_groups
    assert g(["B_X", "B_Y", "B_Z", "B_0"]) == {"B": ["B_X", "B_Y", "B_Z"]}
    assert g(["V_X", "V_Y", "V_Z"]) == {"V": ["V_X", "V_Y", "V_Z"]}
    assert g(["Q_0", "Q_1", "Q_2", "Q_3", "Q_X"]) == {
        "Q": ["Q_0", "Q_1", "Q_2", "Q_3"]}
    sym = ["s_XX", "s_YY", "s_ZZ", "s_XY", "s_YZ", "s_XZ"]
    assert g(sym + ["s_X"]) == {"s": sym}
    assert g(["u_X", "u_Y", "u_XX"]) == {"u": ["u_X", "u_Y"]}
    assert g(["d_0", "d_1"]) == {}  # 2 components are named _X/_Y
    assert g(["d_0", "d_1", "d_2", "d_3", "d_4", "d_5"]) == {}  # 6 -> letters
    assert g(["t", "t_X", "t_Y"]) == {}  # the base itself is a variable


# --------------------------------------------------------------------------- #
# writer: triangles in 3D files
# --------------------------------------------------------------------------- #
def test_triangles_in_3d_readable_by_meshio(tmp_path):
    path = tmp_path / "sphere.exo"
    sphere = pv.Sphere(theta_resolution=6, phi_resolution=6)
    write_exo(path, sphere)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = meshio.read(path, file_format="exodus")
    assert [c.type for c in mesh.cells] == ["triangle"]
    blocks = vtk_reader(path, sets=False).read()["Element Blocks"]
    assert blocks[0].n_cells == sphere.n_cells
    assert set(blocks[0].celltypes) == {CT.TRIANGLE}


def test_single_precision_2_0_file_not_taken_for_1x(tmp_path):
    """Title "pyvista_mesh" + float32 + TRI3 in 3D used to trigger the 1.x
    side renumbering; a 2.0 file always has an element id map."""
    coords = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    model = ExodusModel(
        "pyvista_mesh", 3, coords, [Block(1, "t", "TRI3", [[0, 1, 2]])],
        side_sets=[SideSet(1, "edge", [0], [3])], elem_id_map=[1],
    )
    path = tmp_path / "t.exo"
    write_model(path, model, float_word_size=4)
    sets = read_side_sets(path)
    # shell numbering: side 3 is the first edge, nodes 0-1 (as VTK reads it)
    assert sets[0].get_cell(0).point_ids == [0, 1]
    vtk_set = vtk_reader(path).read()["Side Sets"][0]
    np.testing.assert_array_equal(
        vtk_set.points[vtk_set.get_cell(0).point_ids], coords[[0, 1]])


# --------------------------------------------------------------------------- #
# reader: polygon / polyhedron blocks
# --------------------------------------------------------------------------- #
def test_nsided_block_skipped_with_warning(tmp_path):
    coords = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1], [3, 0.5]])
    path = tmp_path / "nsided.exo"
    raw_exo(path, coords, [
        dict(id=1, type="QUAD4", name="q", conn=[[0, 1, 2, 3]]),
        dict(id=2, type="NSIDED", name="poly", flat=[1, 4, 5, 2, 4, 6, 5],
             counts=[4, 3]),
    ])
    with pytest.warns(UserWarning, match=r"2 \('NSIDED', 2 elements\)"):
        grid = read_exo(path)
    assert grid.n_cells == 1 and list(grid.celltypes) == [CT.QUAD]
    with pytest.warns(UserWarning, match="NSIDED"):
        assert read_element_fields(path) == {}
    with pytest.warns(UserWarning, match="NSIDED"):
        assert read_side_sets(path).n_blocks == 0
    model = read_model(path)  # keeps the polygon block, without connectivity
    assert [(b.elem_type, b.conn.shape) for b in model.blocks] == [
        ("QUAD4", (1, 4)), ("NSIDED", (2, 0))]
    with pytest.raises(ValueError, match="NSIDED / NFACED"):
        write_model(tmp_path / "copy.exo", model)


def test_nfaced_block_keeps_element_numbering(tmp_path):
    xs = np.arange(4.0)
    pts = np.array([[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in xs])

    def nid(i, j, k):
        return k * 8 + j * 4 + i

    def hexc(i):
        return [nid(i, 0, 0), nid(i + 1, 0, 0), nid(i + 1, 1, 0), nid(i, 1, 0),
                nid(i, 0, 1), nid(i + 1, 0, 1), nid(i + 1, 1, 1), nid(i, 1, 1)]

    h = hexc(0)
    faces = [[h[0], h[3], h[2], h[1]], [h[4], h[5], h[6], h[7]],
             [h[0], h[1], h[5], h[4]], [h[1], h[2], h[6], h[5]],
             [h[2], h[3], h[7], h[6]], [h[3], h[0], h[4], h[7]]]
    path = tmp_path / "nfaced.exo"
    # side set: file element 2 (the first hex, x in [1, 2]), side 2 (x = 2)
    raw_exo(path, pts, [
        dict(id=1, type="NFACED", name="poly", faces=faces, conn=np.arange(6),
             counts=[6]),
        dict(id=2, type="HEX8", name="hexes", conn=[hexc(1), hexc(2)]),
    ], side_sets=[(1, [1], [2])])
    with pytest.warns(UserWarning, match="NFACED"):
        grid, sets = read_exo(path, read_side_sets=True)
    assert grid.n_cells == 2
    face = sets[0]
    x = face.points[face.get_cell(0).point_ids][:, 0]
    np.testing.assert_array_equal(np.unique(x), [2.0])
    assert face.cell_data["exo_elem_id"].tolist() == [2]
    assert face.cell_data["cell_id"].tolist() == [0]
    vtk_face = vtk_reader(path).read()["Side Sets"][0]
    np.testing.assert_array_equal(
        np.unique(vtk_face.points[vtk_face.get_cell(0).point_ids][:, 0]), [2.0])


@pytest.mark.parametrize("name", ["HEX_8", "HEX8_ALT", "HEX-8", "HEX8S", "Hex 8"])
def test_element_type_spellings(tmp_path, name):
    grid = hexes(1, 1, 1)
    path = tmp_path / "h.exo"
    write_exo(path, grid)
    with netCDF4.Dataset(path, "a") as ds:
        ds["connect1"].elem_type = name
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_exo(path)
    assert list(back.celltypes) == [CT.HEXAHEDRON]
    vtk_blocks = vtk_reader(path, sets=False).read()["Element Blocks"]
    assert list(vtk_blocks[0].celltypes) == [CT.HEXAHEDRON]


# --------------------------------------------------------------------------- #
# reader: many blocks x variables, fixed time dimension
# --------------------------------------------------------------------------- #
def test_many_blocks_and_variables_read_fast(tmp_path):
    nb, nv, per = 200, 5, 10
    grid = hexes(nb * per, 1, 1)
    grid.cell_data["region"] = np.repeat(np.arange(1, nb + 1), per)
    fields = {f"v{j}": np.full((2, grid.n_cells), float(j)) for j in range(nv)}
    path = tmp_path / "many.exo"
    write_exo(path, grid, times=[0.0, 1.0], element_fields=fields)
    with netCDF4.Dataset(path) as ds:
        assert not ds.dimensions["time_step"].isunlimited()
        assert len(ds.dimensions["time_step"]) == 2
    start = time.perf_counter()
    back = read_exo(path)
    histories = read_element_fields(path)
    elapsed = time.perf_counter() - start
    for j in range(nv):
        np.testing.assert_array_equal(back.cell_data[f"v{j}"], j)
        np.testing.assert_array_equal(histories[f"v{j}"], fields[f"v{j}"])
    assert elapsed < 5.0, f"reading took {elapsed:.2f} s"  # ~0.8 s; was ~9 s


def test_time_dimension_unlimited_without_steps(tmp_path):
    write_exo(tmp_path / "static.exo", hexes(1, 1, 1), save_node_arrays=False)
    with netCDF4.Dataset(tmp_path / "static.exo") as ds:
        assert ds.dimensions["time_step"].isunlimited()
        assert len(ds.dimensions["time_step"]) == 0


# --------------------------------------------------------------------------- #
# reader: parallel decompositions
# --------------------------------------------------------------------------- #
def _two_part_decomposition(tmp_path, n):
    """A Nemesis split of an n^3 hex mesh in two parts, plus the serial file."""
    grid = hexes(n, n, n)
    pts = np.asarray(grid.points, dtype=float)
    conn = grid.cell_connectivity.reshape(-1, 8)[:, [0, 1, 3, 2, 4, 5, 7, 6]]
    u = pts[:, 0] + 2 * pts[:, 1]
    write_model(tmp_path / "serial.exo", ExodusModel(
        "t", 3, pts, [Block(1, "b", "HEX8", conn)], times=[0.0],
        node_vars={"u": u[None]}))
    center = pts[conn].mean(axis=1)[:, 0]
    for part, sel in enumerate((center < n / 2, center >= n / 2)):
        nodes = np.unique(conn[sel])
        path = tmp_path / f"par.exo.2.{part}"
        write_model(path, ExodusModel(
            "t", 3, pts[nodes], [Block(1, "b", "HEX8",
                                       np.searchsorted(nodes, conn[sel]))],
            elem_id_map=np.flatnonzero(sel) + 1, times=[0.0],
            node_vars={"u": u[nodes][None]}))
        with netCDF4.Dataset(path, "a") as ds:
            ds.createDimension("num_nodes_global", len(pts))
            ds.createDimension("num_elems_global", len(conn))
            ds.createDimension("num_el_blk_global", 1)
            ds.createDimension("num_processors", 2)
            ds.createDimension("num_procs_file", 1)
            ds.createVariable("el_blk_ids_global", "i4", ("num_el_blk_global",))[:] = 1
            ds.createVariable("el_blk_cnt_global", "i4",
                              ("num_el_blk_global",))[:] = len(conn)
            ds.createVariable("node_num_map", "i4", ("num_nodes",))[:] = nodes + 1
    return str(tmp_path / "par.exo.2.*"), str(tmp_path / "serial.exo")


def test_parallel_read_is_vectorized(tmp_path):
    pattern, serial = _two_part_decomposition(tmp_path, 64)  # 262,144 hexes
    start = time.perf_counter()
    grid = read_exo(pattern)
    fields = read_node_fields(pattern)
    elapsed = time.perf_counter() - start
    ref = read_exo(serial)
    assert grid.n_cells == ref.n_cells == 64**3
    np.testing.assert_array_equal(grid.points, ref.points)
    np.testing.assert_array_equal(grid.cell_connectivity, ref.cell_connectivity)
    np.testing.assert_array_equal(grid.point_data["u"], ref.point_data["u"])
    np.testing.assert_array_equal(fields["u"], read_node_fields(serial)["u"])
    # ~0.3 s with array maps; the per-node dict loops took ~2 s
    assert elapsed < 1.5, f"parallel read took {elapsed:.2f} s"


@pytest.mark.parametrize(
    "reader",
    [read_exo, read_element_fields, read_node_fields, read_node_sets,
     read_global_fields, read_side_sets],
)
def test_incomplete_decomposition_raises(reader):
    parts = sorted(glob.glob(os.path.join(DATA, "edges.exo.4.*")))
    if len(parts) != 4:
        pytest.skip("edges.exo.4.* not available")
    with pytest.raises(ValueError, match="Incomplete parallel decomposition"):
        reader(*parts[:3])
    with pytest.raises(ValueError, match="more than once"):
        reader(*parts, parts[0])


def test_incomplete_decomposition_raises_in_legacy_api():
    parts = sorted(glob.glob(os.path.join(DATA, "edges.exo.4.*")))
    if len(parts) != 4:
        pytest.skip("edges.exo.4.* not available")
    with pytest.raises(ValueError, match="4 part"):
        parallel_exodusii_file(*parts[:3])


# --------------------------------------------------------------------------- #
# reader: duplicate / empty variable names
# --------------------------------------------------------------------------- #
@pytest.fixture
def renamed_vars(tmp_path):
    grid = hexes(2, 1, 1)
    path = tmp_path / "names.exo"
    write_exo(
        path, grid, times=[0.0, 1.0],
        node_fields={"n1": np.ones((2, 12)), "n2": np.full((2, 12), 2.0)},
        element_fields={"e1": [[10.0, 10.0], [11.0, 11.0]],
                        "e2": [[20.0, 20.0], [21.0, 21.0]]},
        global_fields={"g1": [100.0, 101.0], "g2": [200.0, 201.0]},
    )

    def rename(new):
        with netCDF4.Dataset(path, "a") as ds:
            for var, names in new.items():
                for i, name in enumerate(names):
                    row = np.zeros(ds[var].shape[1], "S1")
                    row[: len(name)] = np.frombuffer(name.encode(), "S1")
                    ds[var][i] = row
        return path

    return rename


def test_duplicate_variable_names_stay_readable(renamed_vars):
    path = renamed_vars({"name_nod_var": ["n1", "n1"], "name_elem_var": ["e1", "e1"],
                         "name_glo_var": ["g1", "g1"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid = read_exo(path)
    messages = " ".join(str(w.message) for w in caught)
    for old_new in ("'n1' -> 'n1_'", "'e1' -> 'e1_'", "'g1' -> 'g1_'"):
        assert old_new in messages
    np.testing.assert_array_equal(grid.point_data["n1"], 1.0)
    np.testing.assert_array_equal(grid.point_data["n1_"], 2.0)
    np.testing.assert_array_equal(grid.cell_data["e1_"], [21.0, 21.0])
    assert grid.field_data["g1_"][0] == 201.0
    with pytest.warns(UserWarning):
        assert read_node_fields(path)["n1_"][0, 0] == 2.0
    with pytest.warns(UserWarning):
        assert read_element_fields(path)["e1_"][0, 0] == 20.0
    with pytest.warns(UserWarning):
        np.testing.assert_array_equal(read_global_fields(path)["g1_"], [200, 201])
    # VTK names them the same way
    vtk = vtk_reader(path, sets=False).read()["Element Blocks"][0]
    assert {"n1", "n1_"} <= set(vtk.point_data.keys())


def test_empty_variable_names_get_defaults(renamed_vars):
    path = renamed_vars({"name_nod_var": ["", ""], "name_elem_var": [""],
                         "name_glo_var": [""]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid = read_exo(path)
    messages = " ".join(str(w.message) for w in caught)
    for old_new in ("'' -> 'node_var_1'", "'' -> 'node_var_2'",
                    "'' -> 'elem_var_1'", "'' -> 'global_var_1'"):
        assert old_new in messages
    np.testing.assert_array_equal(grid.point_data["node_var_1"], 1.0)
    np.testing.assert_array_equal(grid.point_data["node_var_2"], 2.0)
    np.testing.assert_array_equal(grid.cell_data["elem_var_1"], [11.0, 11.0])
    np.testing.assert_array_equal(grid.cell_data["e2"], [21.0, 21.0])
    assert grid.field_data["global_var_1"][0] == 101.0
    with pytest.warns(UserWarning):
        assert set(read_node_fields(path)) == {"node_var_1", "node_var_2"}
