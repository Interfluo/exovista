"""Thin helpers over :mod:`netCDF4` used by the vendored ExodusII core.

netCDF4 is a hard dependency; there is no pure-Python fallback.
"""

import numpy as np
from netCDF4 import Dataset

from .util import stringify, string_kinds
from .config import config


def library():
    return "netcdf4"


def open(filename, mode="r"):
    if mode == "w":
        return Dataset(filename, mode=mode, format="NETCDF4_CLASSIC")
    return Dataset(filename, mode=mode)


def close(*files):
    for file in files:
        try:
            if file.isopen():
                file.close()
        except Exception:
            pass


def filename(fh):
    return fh.filepath()


def get_variable(fh, name, default=None, raw=False, index=None):
    """Return the data of variable ``name`` (or ``default`` if absent).

    Parameters
    ----------
    raw : bool
        Return the :class:`netCDF4.Variable` itself instead of its data.
    index : int, slice or tuple, optional
        Read only ``var[index]`` (e.g. a single time step) instead of the whole
        variable, so large time histories are never loaded just to be sliced.
    """
    if name not in fh.variables:
        return default
    var = fh.variables[name]
    if raw:
        return var
    val = var[:] if index is None else var[index]
    val = np.ma.getdata(val)
    if isinstance(val, np.ndarray) and val.dtype.kind in string_kinds:
        val = stringify(val)
    elif isinstance(val, bytes):
        val = stringify(val)
    return val


def get_dimension(fh, name, default=None):
    if name not in fh.dimensions:
        return default
    dim = fh.dimensions[name].size
    if dim is None and name != "time_step":
        return 0
    return int(dim)


def setncattr(fh, variable, name, value):
    fh.variables[variable].setncattr(name, value)


def create_dimension(fh, name, value):
    fh.createDimension(name, value)


def create_variable(fh, id, type, shape, complevel=None):
    """Create variable ``id``; float word size follows the file's attribute.

    ``complevel`` is the deflate level (0 disables compression); ``None`` uses
    the configured default (``EXODUSII_COMPRESSION_LEVEL``).
    """
    if type is float:
        word_size = getattr(fh, "floating_point_word_size", 8)
        kind = "f" if int(word_size) == 4 else "d"
    else:
        kind = {str: "c", int: "i"}[type]
    fh.createVariable(id, kind, shape, **_compression_kwds(fh, shape, complevel))


def _compression_kwds(fh, shape, complevel=None):
    """Return netCDF4 compression keywords for createVariable, when applicable.

    Compression (HDF5 deflate) is only available for the HDF5 based NETCDF4
    formats. Scalar variables (no dimensions) cannot be chunked, so they are
    left uncompressed.
    """
    level = config.compression_level if complevel is None else complevel
    if not level or not shape:
        return {}
    # zlib/chunking is only supported by the HDF5 based NETCDF4* data models.
    if not str(getattr(fh, "data_model", "")).startswith("NETCDF4"):
        return {}
    return {"zlib": True, "complevel": int(level), "shuffle": True}


def fill_variable(fh, name, *args):
    value = args[-1]
    if len(args) == 1:
        fh.variables[name][:] = value
    elif len(args) == 2:
        i = args[0]
        fh.variables[name][i, :] = value
    elif len(args) == 3:
        i, j = args[0:2]
        fh.variables[name][i, j, :] = value
    elif len(args) == 4:
        i, j, k = args[0:3]
        fh.variables[name][i, j, k, :] = value
    else:
        raise ValueError("Unknown fill shape")
