"""PyVista version guards for test oracles and helpers.

exovista itself supports PyVista >= 0.42 (see pyproject.toml), but parts of
the test suite check its output, or build their inputs, with newer PyVista
features: VTK's Exodus reader as ``pv.ExodusIIReader`` (0.45) and
``pv.PExodusIIReader`` (0.49), ``PolyData.flip_faces`` and the quadratic
cells of ``pyvista.examples.cells`` (0.45), ``extract_surface(algorithm=...)``
(0.47), ``validate_mesh`` with its validation fields and the remaining
example cells (0.48), the ``pyvista.writers`` registry (0.48). Such tests skip
on older PyVista with a reason that names the missing feature; tests of
exovista's own behaviour keep running.

Use the wrappers below, :func:`requires_pyvista` inside a test or helper
(skips at that point), or the ``requires_pyvista(version, what)`` marker on a
test or module (skips before the test starts).
"""

import os
import re

import pytest
import pyvista as pv


def version_tuple(text):
    """``"0.49.2"`` -> ``(0, 49, 2)`` (non-numeric parts count as 0)."""
    parts = []
    for part in str(text).split(".")[:3]:
        match = re.match(r"\d+", part)
        parts.append(int(match.group()) if match else 0)
    return tuple(parts + [0] * (3 - len(parts)))


PYVISTA_VERSION = version_tuple(pv.__version__)
VTK_VERSION = tuple(pv.vtk_version_info)[:2]

#: Cell types whose size VTK's compute_cell_sizes (and so validate_mesh) gets
#: wrong, negative for valid cells, by VTK (major, minor) version.
_VTK_SIZE_BUGS = {
    (9, 6): {pv.CellType.QUADRATIC_WEDGE, pv.CellType.TRIQUADRATIC_HEXAHEDRON},
}


def vtk_measures(celltype):
    """False where this VTK computes a wrong (negative) size for ``celltype``."""
    return int(celltype) not in {int(t) for t in _VTK_SIZE_BUGS.get(VTK_VERSION, ())}


def pyvista_at_least(version):
    return PYVISTA_VERSION >= version_tuple(version)


def skip_reason(version, what):
    return (
        f"needs PyVista >= {version} for {what} (installed {pv.__version__}); "
        "a test oracle/helper requirement, not one of exovista"
    )


def requires_pyvista(version, what):
    """Skip the running test when PyVista is older than ``version``."""
    if not pyvista_at_least(version):
        pytest.skip(skip_reason(version, what))


def extract_surface(mesh, **kwargs):
    """``mesh.extract_surface(algorithm=None, **kwargs)`` (PyVista >= 0.47).

    The ``algorithm`` argument orients the faces of linear wedges outward;
    older PyVista only has the ``dataset_surface`` filter, so tests that need
    the outward faces skip there.
    """
    requires_pyvista("0.47", "extract_surface(algorithm=None)")
    return mesh.extract_surface(algorithm=None, **kwargs)


def exodus_reader(path):
    """``pv.ExodusIIReader(path)``: VTK's Exodus reader, the test oracle."""
    requires_pyvista("0.45", "pv.ExodusIIReader (VTK's Exodus reader oracle)")
    return pv.ExodusIIReader(os.fspath(path))


def p_exodus_reader(path):
    """``pv.PExodusIIReader(path)``: VTK's parallel Exodus reader."""
    requires_pyvista("0.49", "pv.PExodusIIReader (VTK's parallel reader oracle)")
    return pv.PExodusIIReader(os.fspath(path))


def validate_mesh(mesh, *args, **kwargs):
    """``mesh.validate_mesh(...)``: PyVista's cell validity checks."""
    requires_pyvista("0.48", "DataSet.validate_mesh (cell validity oracle)")
    wrong = sorted(pv.CellType(t).name for t in set(map(int, mesh.celltypes))
                   if not vtk_measures(t))
    if wrong:
        pytest.skip(
            f"VTK {'.'.join(map(str, VTK_VERSION))} computes negative sizes for "
            f"valid {', '.join(wrong)} cells (an oracle bug, not exovista's)"
        )
    return mesh.validate_mesh(*args, **kwargs)


#: Example cells of pyvista.examples.cells that need PyVista 0.48.
_LATE_EXAMPLE_CELLS = {"LagrangeHexahedron", "ConvexPointSet", "BezierTriangle"}


def example_cell(name):
    """``pyvista.examples.cells.<name>()``: a test input."""
    from pyvista.examples import cells

    if not hasattr(cells, name):
        version = "0.48" if name in _LATE_EXAMPLE_CELLS else "0.45"
        pytest.skip(skip_reason(version, f"pyvista.examples.cells.{name}"))
    return getattr(cells, name)()


def flip_faces(poly):
    """``poly.flip_faces()``: reverse the orientation of every face."""
    requires_pyvista("0.45", "PolyData.flip_faces")
    return poly.flip_faces()
