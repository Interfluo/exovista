"""Legacy low-level ExodusII API (exovista 1.x).

exovista 1.x exposed the low-level file API of the vendored Sandia
``exodusii`` package from the top-level ``exovista`` namespace.  In 2.0 it
lives here::

    from exovista.legacy import File, allclose

    with File("model.exo") as exof:
        coords = exof.get_coords()

The names below are re-exported unchanged and without warnings.  Accessing
them as ``exovista.<name>`` still works in 2.x but emits a
:class:`DeprecationWarning`; that top-level access will be removed in
exovista 3.0.  New code that needs a general-purpose low-level ExodusII API
should consider the upstream `exodusii <https://pypi.org/project/exodusii/>`_
package.  For PyVista meshes use :func:`exovista.write_exo` and
:func:`exovista.read_exo`.
"""

import glob

from .allclose import allclose
from .exoread import main as exoread
from .extension import (
    compute_edge_centers,
    compute_element_centers,
    compute_element_length,
    compute_element_volumes,
    compute_face_centers,
    compute_node_variable_values_at_element_center,
    compute_node_volumes,
    compute_volume_averaged_elem_variable,
)
from .file import ExodusIIFile, exodusii_file, write_globals
from .find_in_region import find_element_data_in_region, find_node_data_in_region
from .lineout import lineout
from .parallel_file import MFExodusIIFile, parallel_exodusii_file
from .similar import similar

__all__ = [
    # Core file operations
    "File",
    "exo_file",
    "exodusii_file",
    "ExodusIIFile",
    "write_globals",
    # Parallel file operations
    "parallel_exodusii_file",
    "MFExodusIIFile",
    # Comparison utilities
    "allclose",
    "similar",
    # Data extraction
    "lineout",
    "find_element_data_in_region",
    "find_node_data_in_region",
    # Command-line interface (``exovista-read``)
    "exoread",
    # Geometry helpers
    "compute_element_centers",
    "compute_edge_centers",
    "compute_face_centers",
    "compute_element_length",
    "compute_element_volumes",
    "compute_volume_averaged_elem_variable",
    "compute_node_volumes",
    "compute_node_variable_values_at_element_center",
]


def File(filename, *files, mode="r"):
    """Open an ExodusII file (or a parallel decomposition) for reading or writing.

    Parameters
    ----------
    filename : str
        Path of the file. In read mode it may be a glob pattern.
    *files : str
        Further files (or glob patterns) of a parallel decomposition
        (read mode only).
    mode : {"r", "w"}
        Read or write mode.

    Returns
    -------
    exodusii_file or parallel_exodusii_file
        A :class:`parallel_exodusii_file` when more than one file matches,
        otherwise an :class:`exodusii_file`.
    """
    if mode not in ("r", "w"):
        raise ValueError(f"Invalid Exodus file mode {mode!r}")

    if mode == "r":
        files = _find_files(filename, *files)
        if len(files) > 1:
            return parallel_exodusii_file(*files)
        if len(files) == 1:
            return exodusii_file(files[0], mode="r")
        raise ValueError("No files to open")

    if files:
        raise TypeError(f"Exodus writer takes 1 file but {len(files) + 1} were given")
    return exodusii_file(filename, mode="w")


exo_file = File


def _find_files(*files):
    found = []
    for file in files:
        globbed_files = glob.glob(file)
        if not globbed_files:
            raise FileNotFoundError(file)
        found.extend(globbed_files)
    return found
