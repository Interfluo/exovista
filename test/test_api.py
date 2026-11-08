"""The public API of exovista 2.0 and the deprecated 1.x top-level names."""
import ast
import glob
import os
import subprocess
import sys
import warnings
from importlib import metadata

import pytest

import exovista
import exovista.legacy

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

PUBLIC = [
    "write_exo",
    "read_exo",
    "read_side_sets",
    "read_node_sets",
    "read_node_fields",
    "read_element_fields",
    "read_global_fields",
]

#: The low-level API exovista 1.x exported from the top-level package.
LEGACY = [
    "File",
    "exo_file",
    "exodusii_file",
    "ExodusIIFile",
    "write_globals",
    "parallel_exodusii_file",
    "MFExodusIIFile",
    "allclose",
    "similar",
    "lineout",
    "find_element_data_in_region",
    "find_node_data_in_region",
    "exoread",
    "compute_element_centers",
    "compute_edge_centers",
    "compute_face_centers",
    "compute_element_length",
    "compute_element_volumes",
    "compute_volume_averaged_elem_variable",
    "compute_node_volumes",
    "compute_node_variable_values_at_element_center",
]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def test_all_is_the_2_0_api():
    assert sorted(exovista.__all__) == sorted(PUBLIC + ["__version__"])
    for name in PUBLIC:
        assert callable(getattr(exovista, name)), name


def test_star_import_exports_only_the_public_api():
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exec("from exovista import *", namespace)
    namespace.pop("__builtins__")
    assert sorted(namespace) == sorted(PUBLIC + ["__version__"])


def test_public_functions_live_in_their_modules():
    from exovista import read_pyvista, write_pyvista

    assert exovista.write_exo is write_pyvista.write_exo
    for name in PUBLIC[1:]:
        assert getattr(exovista, name) is getattr(read_pyvista, name)


def test_version():
    assert isinstance(exovista.__version__, str) and exovista.__version__
    try:
        installed = metadata.version("exovista")
    except metadata.PackageNotFoundError:
        assert exovista.__version__ == "0+unknown"
    else:
        assert exovista.__version__ == installed


def test_version_matches_pyproject():
    pyproject = os.path.join(ROOT, "pyproject.toml")
    if not os.path.exists(pyproject):
        pytest.skip("not running from a source checkout")
    tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
    with open(pyproject, "rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    try:
        metadata.version("exovista")
    except metadata.PackageNotFoundError:
        pytest.skip("exovista is not installed")
    assert exovista.__version__ == declared, "reinstall exovista (pip install -e .)"


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
        exovista.no_such_name  # noqa: B018
    assert not hasattr(exovista, "no_such_name")


# --------------------------------------------------------------------------- #
# exovista.legacy and the deprecated top-level names
# --------------------------------------------------------------------------- #
def test_legacy_module_exports():
    assert sorted(exovista.legacy.__all__) == sorted(LEGACY)
    for name in LEGACY:
        assert hasattr(exovista.legacy, name), name


def test_legacy_module_access_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in LEGACY:
            getattr(exovista.legacy, name)
        exec(f"from exovista.legacy import {', '.join(LEGACY)}", {})


@pytest.mark.parametrize("name", LEGACY)
def test_top_level_legacy_access_is_deprecated(name):
    with pytest.warns(DeprecationWarning) as record:
        obj = getattr(exovista, name)
    assert obj is getattr(exovista.legacy, name)
    assert len(record) == 1
    message = str(record[0].message)
    assert f"exovista.{name} is deprecated" in message
    assert "exovista.legacy" in message
    assert "3.0" in message
    # stacklevel points at the code that used the deprecated name
    assert os.path.samefile(record[0].filename, __file__)


def test_from_import_of_legacy_name_is_deprecated():
    with pytest.warns(DeprecationWarning, match="exovista.legacy") as record:
        from exovista import File  # noqa: F401
    assert os.path.samefile(record[0].filename, __file__)
    assert File is exovista.legacy.File


@pytest.mark.parametrize("name", ["allclose", "similar", "lineout", "exoread"])
def test_legacy_names_shadowing_submodules(name):
    """allclose/similar/lineout/exoread are also submodule names: the
    deprecated top-level name is the function/class, as in 1.x, even after
    the submodule was imported explicitly."""
    __import__(f"exovista.{name}")
    assert name not in vars(exovista)
    with pytest.warns(DeprecationWarning):
        obj = getattr(exovista, name)
    assert obj is getattr(exovista.legacy, name)
    assert callable(obj)


def test_deprecated_aliases_still_work(datadir):
    noh = os.path.join(datadir, "noh.exo")
    with pytest.warns(DeprecationWarning):
        exo_file = exovista.File
    with exo_file(noh) as exof:
        assert exof.num_dimensions() == 2
    with pytest.warns(DeprecationWarning):
        assert exovista.allclose(noh, noh)


def test_legacy_file_modes(tmp_path):
    with pytest.raises(ValueError, match="Invalid Exodus file mode"):
        exovista.legacy.File(str(tmp_path / "x.exo"), mode="rw")
    with pytest.raises(TypeError, match="takes 1 file"):
        exovista.legacy.File("a.exo", "b.exo", mode="w")
    with pytest.raises(FileNotFoundError):
        exovista.legacy.File(str(tmp_path / "missing.exo"))
    assert exovista.legacy.exo_file is exovista.legacy.File


def _legacy_attribute_uses(path):
    """(lineno, text) of uses of deprecated top-level names in a source file."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), path)
    uses = []
    legacy = set(LEGACY)
    in_package = os.path.dirname(os.path.realpath(path)) == os.path.realpath(
        os.path.dirname(exovista.__file__)
    )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in legacy
            and isinstance(node.value, ast.Name)
            and node.value.id == "exovista"
        ):
            uses.append((node.lineno, f"exovista.{node.attr}"))
        elif isinstance(node, ast.ImportFrom):
            package_import = node.module == "exovista" or (
                in_package and node.level == 1 and node.module is None
            )
            if package_import:
                for alias in node.names:
                    if alias.name in legacy:
                        uses.append((node.lineno, f"from exovista import {alias.name}"))
    return uses


def test_package_and_examples_do_not_use_deprecated_names():
    files = sorted(glob.glob(os.path.join(os.path.dirname(exovista.__file__), "*.py")))
    files += sorted(glob.glob(os.path.join(ROOT, "examples", "*.py")))
    offenders = {}
    for path in files:
        uses = _legacy_attribute_uses(path)
        if uses:
            offenders[os.path.relpath(path, ROOT)] = uses
    assert not offenders


# --------------------------------------------------------------------------- #
# import hygiene
# --------------------------------------------------------------------------- #
_IMPORT_CHECK = r"""
import importlib.util, logging, os, sys, warnings

filters = list(warnings.filters)
root_handlers = list(logging.root.handlers)
root_level = logging.root.level
package = os.path.dirname(importlib.util.find_spec("exovista").origin)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import exovista
ours = [str(w.message) for w in caught
        if os.path.abspath(w.filename).startswith(package + os.sep)
        or str(w.message).startswith("exovista")]
assert not ours, ours
assert list(warnings.filters) == filters, "import changed the warning filters"
assert list(logging.root.handlers) == root_handlers, "import configured logging"
assert logging.root.level == root_level
logger = logging.getLogger("exovista")
assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
assert logger.level == logging.NOTSET and logger.propagate
print("ok")
"""


def test_import_is_side_effect_free():
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_CHECK],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_package_is_typed():
    assert os.path.exists(os.path.join(os.path.dirname(exovista.__file__), "py.typed"))
