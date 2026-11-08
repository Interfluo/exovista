"""The PyVista writer plugin: ``mesh.save("model.exo")`` (exovista._pyvista_plugin).

PyVista >= 0.48 dispatches ``DataObject.save`` to writers registered in the
``pyvista.writers`` entry-point group; exovista registers ``save_exodus`` for
.exo, .e, .ex2, .exii and .gen.  The direct-call tests of the handler run with
any PyVista; the ``mesh.save`` tests need the registry and an installed
exovista (``pip install -e .``) so that the entry points exist.
"""
import os
from importlib import metadata

import netCDF4
import numpy as np
import pytest
import pyvista as pv

import exovista
from _pvcompat import exodus_reader, extract_surface
from exovista._pyvista_plugin import EXTENSIONS, save_exodus

HANDLER = "exovista._pyvista_plugin:save_exodus"


def _registered_entry_points():
    try:
        dist = metadata.distribution("exovista")
    except metadata.PackageNotFoundError:
        return None
    return {
        ep.name: ep.value for ep in dist.entry_points if ep.group == "pyvista.writers"
    }


@pytest.fixture(scope="module")
def registry():
    """Skip unless ``mesh.save`` will dispatch .exo files to exovista."""
    if not hasattr(pv, "registered_writers"):
        pytest.skip(f"PyVista {pv.__version__} has no writer registry (needs >= 0.48)")
    if _registered_entry_points() is None:
        pytest.skip("exovista is not installed, so its entry points are not registered")
    return {r.extension: r for r in pv.registered_writers()}


def tet_sphere():
    return pv.Sphere(theta_resolution=8, phi_resolution=8).delaunay_3d()


def hexes(n=2):
    return pv.ImageData(dimensions=(n + 1, n + 1, n + 1)).cast_to_unstructured_grid()


def surface_of(mesh):
    return extract_surface(mesh)  # algorithm=None: outward faces


def vtk_read(path):
    """Read with VTK's own Exodus reader (independent of exovista)."""
    reader = exodus_reader(str(path))
    reader.enable_all_point_arrays()
    reader.enable_all_cell_arrays()
    return reader.read()


def vtk_block_names(path):
    return list(vtk_read(path)["Element Blocks"].keys())


def zlib_level(path, var="coordx"):
    with netCDF4.Dataset(path) as ds:
        filters = ds[var].filters()
        return filters["complevel"] if filters["zlib"] else 0


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def test_entry_points_declared():
    entry_points = _registered_entry_points()
    if entry_points is None:
        pytest.skip("exovista is not installed")
    assert entry_points == dict.fromkeys(EXTENSIONS, HANDLER)
    # .g is BYU geometry in PyVista; exovista must not claim it
    assert ".g" not in entry_points and "g" not in entry_points


def test_extensions_match_reader_conventions():
    assert EXTENSIONS == (".exo", ".e", ".ex2", ".exii", ".gen")


def test_registered_writers(registry):
    for ext in EXTENSIONS:
        assert ext in registry, ext
        assert registry[ext].handler is save_exodus
        assert registry[ext].source == HANDLER
    assert ".g" not in registry or registry[".g"].handler is not save_exodus


def test_plugin_import_is_cheap_and_safe():
    import exovista._pyvista_plugin as plugin

    assert plugin.__all__ == ["save_exodus"]
    assert plugin.save_exodus.__module__ == "exovista._pyvista_plugin"


# --------------------------------------------------------------------------- #
# mesh.save(...)
# --------------------------------------------------------------------------- #
def test_delaunay_sphere_save(registry, tmp_path):
    mesh = tet_sphere()
    path = tmp_path / "x.exo"
    mesh.save(path)

    vtk = pv.read(path)  # VTK's reader, dispatched on .exo
    blocks = vtk["Element Blocks"]
    assert sum(b.n_cells for b in blocks if b is not None) == mesh.n_cells

    back = exovista.read_exo(path)
    assert back.n_cells == mesh.n_cells
    assert np.array_equal(back.points, mesh.points)
    assert np.array_equal(back.celltypes, mesh.celltypes)
    assert np.array_equal(back.cell_connectivity, mesh.cell_connectivity)


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_every_extension(registry, tmp_path, ext):
    mesh = hexes()
    mesh.point_data["T"] = mesh.points[:, 0].copy()
    path = tmp_path / f"m{ext}"
    mesh.save(str(path))

    back = exovista.read_exo(path)
    assert back.n_cells == mesh.n_cells
    assert np.array_equal(back.point_data["T"], mesh.point_data["T"])

    vtk = vtk_read(path)  # .gen has no pv.read mapping; use the reader directly
    block = vtk["Element Blocks"][0]
    assert block.n_cells == mesh.n_cells
    assert np.array_equal(block.point_data["T"], block.points[:, 0])


def test_save_forwards_write_exo_keywords(registry, tmp_path):
    mesh = hexes(2)
    centers = mesh.cell_centers().points
    mesh.cell_data["region"] = np.where(centers[:, 0] < 1.0, 1, 2)
    surface = surface_of(mesh)
    surface.cell_data["region"] = np.ones(surface.n_cells, dtype=int)
    times = [0.0, 0.5, 1.0]
    temp = np.stack([mesh.points[:, 0] * t for t in times])
    path = tmp_path / "kw.e"

    mesh.save(
        path,
        surface=surface,
        block_names={1: "left", 2: "right"},
        side_set_names=["outer"],
        times=times,
        node_fields={"temp": temp},
        global_fields={"energy": np.array(times) * 2},
        compression_level=0,
    )

    assert vtk_block_names(path) == ["left", "right"]
    assert exodus_reader(str(path)).number_time_points == 3
    assert zlib_level(path) == 0

    back = exovista.read_exo(path, time_step=2)
    assert back.field_data["time"][0] == 0.5
    assert np.array_equal(back.point_data["temp"], temp[1])
    assert set(back.cell_data["exo_block_name"]) == {"left", "right"}
    side_sets = exovista.read_side_sets(path)
    assert list(side_sets.keys()) == ["outer"]
    assert side_sets["outer"].n_cells == surface.n_cells
    fields = exovista.read_global_fields(path)
    assert np.array_equal(fields["energy"], np.array(times) * 2)


def test_save_unknown_keyword_raises(registry, tmp_path):
    with pytest.raises(TypeError, match="no_such_option"):
        hexes().save(tmp_path / "bad.exo", no_such_option=1)
    assert not (tmp_path / "bad.exo").exists()


def test_save_invalid_mesh_leaves_no_file(registry, tmp_path):
    with pytest.raises(ValueError):
        pv.UnstructuredGrid().save(tmp_path / "empty.exo")
    assert not (tmp_path / "empty.exo").exists()


def test_multiblock_save(registry, tmp_path):
    left = hexes(1)
    right = hexes(1).translate((2.0, 0.0, 0.0))
    blocks = pv.MultiBlock({"left": left, "right": right})
    path = tmp_path / "mb.exo"
    blocks.save(path)

    assert vtk_block_names(path) == ["left", "right"]
    back = exovista.read_exo(path)
    assert back.n_cells == left.n_cells + right.n_cells
    assert list(dict.fromkeys(back.cell_data["exo_block_name"])) == ["left", "right"]


def test_polydata_shell_save(registry, tmp_path):
    sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)
    path = tmp_path / "shell.exo"
    sphere.save(path)  # PolyData.save has its own recompute_normals parameter

    with netCDF4.Dataset(path) as ds:
        assert ds.dimensions["num_dim"].size == 3
        assert ds["connect1"].getncattr("elem_type") == "TRI3"
    back = exovista.read_exo(path)
    assert back.n_cells == sphere.n_cells
    assert np.all(back.celltypes == pv.CellType.TRIANGLE)
    assert np.array_equal(back.points, sphere.points)
    vtk = pv.read(path)["Element Blocks"][0]
    assert vtk.n_cells == sphere.n_cells


def test_pyvista_save_options_do_not_break_exodus(registry, tmp_path):
    """binary/texture/compression are consumed by DataObject.save and never
    reach the plugin (PyVista 0.48/0.49); the file is written as usual."""
    mesh = hexes()
    path = tmp_path / "opts.exo"
    mesh.save(path, binary=False, compression=None)
    assert exovista.read_exo(path).n_cells == mesh.n_cells


def test_save_into_missing_directory(registry, tmp_path):
    with pytest.raises(FileNotFoundError):
        hexes().save(tmp_path / "missing" / "x.exo")


# --------------------------------------------------------------------------- #
# the handler itself (works with any PyVista)
# --------------------------------------------------------------------------- #
def test_handler_is_write_exo(tmp_path):
    mesh = tet_sphere()
    save_exodus(mesh, str(tmp_path / "direct.exo"), title="direct")
    back = exovista.read_exo(tmp_path / "direct.exo")
    assert back.n_cells == mesh.n_cells
    with netCDF4.Dataset(tmp_path / "direct.exo") as ds:
        assert ds.title == "direct"


def test_handler_accepts_pathlike(tmp_path):
    save_exodus(hexes(), tmp_path / "p.exo")
    assert (tmp_path / "p.exo").exists()


def test_handler_ignores_binary_and_recompute_normals(tmp_path):
    save_exodus(hexes(), str(tmp_path / "b.exo"), binary=False, recompute_normals=True)
    assert exovista.read_exo(tmp_path / "b.exo").n_cells == 8


def test_handler_rejects_texture(tmp_path):
    with pytest.raises(ValueError, match="texture"):
        save_exodus(hexes(), str(tmp_path / "t.exo"), texture="rgb")
    assert not (tmp_path / "t.exo").exists()
    save_exodus(hexes(), str(tmp_path / "t.exo"), texture=None)
    assert (tmp_path / "t.exo").exists()


@pytest.mark.parametrize(
    "compression, level, expected",
    [
        (None, None, 0),
        (False, None, 0),
        (None, 0, 0),
        ("zlib", None, "default"),
        ("ZLIB", 4, 4),
        ("zlib", 0, 0),
    ],
)
def test_handler_maps_pyvista_compression(tmp_path, compression, level, expected):
    path = str(tmp_path / "c.exo")
    kwargs = {} if level is None else {"compression_level": level}
    save_exodus(hexes(), path, compression=compression, **kwargs)
    if expected == "default":
        save_exodus(hexes(), str(tmp_path / "ref.exo"))
        expected = zlib_level(str(tmp_path / "ref.exo"))
    assert zlib_level(path) == expected


@pytest.mark.parametrize("compression", ["lz4", "lzma", 1])
def test_handler_rejects_other_compression(tmp_path, compression):
    with pytest.raises(ValueError, match="compression"):
        save_exodus(hexes(), str(tmp_path / "c.exo"), compression=compression)
    assert not os.path.exists(tmp_path / "c.exo")


def test_handler_rejects_conflicting_compression(tmp_path):
    with pytest.raises(ValueError, match="conflicts"):
        save_exodus(
            hexes(), str(tmp_path / "c.exo"), compression=None, compression_level=5
        )
