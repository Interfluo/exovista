"""
ExoVista - PyVista to ExodusII Export Library.

This library extends the Sandia exodusii library to provide a simple interface
for exporting PyVista meshes to ExodusII format with user-defined element blocks
and side sets.

Key Functions:
    write_exo: Write PyVista meshes to ExodusII format
    File: Open ExodusII files for reading or writing
    allclose: Compare ExodusII files for numerical equality
    similar: Compare ExodusII files for structural similarity
"""

from .file import exodusii_file, ExodusIIFile, write_globals
from .parallel_file import parallel_exodusii_file, MFExodusIIFile
from .allclose import allclose
from .similar import similar
from .extension import *  # noqa: F403
from .lineout import lineout
from .find_in_region import find_element_data_in_region, find_node_data_in_region
from .exoread import main as exoread
from .write_pyvista import write_exo
from .read_pyvista import read_exo, read_node_fields, read_element_fields

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
    # CLI
    "exoread",
    # PyVista export / import
    "write_exo",
    "read_exo",
    "read_node_fields",
    "read_element_fields",
]

def File(filename, *files, mode="r"):

    if mode not in "rw":
        raise ValueError(f"Invalid Exodus file mode {mode!r}")

    if mode == "r":
        files = _find_files(filename, *files)
        if len(files) > 1:
            f = parallel_exodusii_file(*files)
        elif len(files) == 1:
            f = exodusii_file(files[0], mode="r")
        else:
            raise ValueError("No files to open")
    elif mode == "w":
        if files:
            raise TypeError(f"Exodus writer takes 1 file but {len(files)+1} were given")
        f = exodusii_file(filename, mode="w")

    return f


exo_file = File


def _find_files(*files):
    import glob

    found = []
    for file in files:
        globbed_files = glob.glob(file)
        if not globbed_files:
            raise FileNotFoundError(file)
        found.extend(globbed_files)
    return found
