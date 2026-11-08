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


def close(*files, strict=False):
    """Close ``files`` (open ones only); safe to call more than once.

    By default errors are ignored (read handles, cleanup after an error).
    With ``strict=True`` every file is still closed, but the first error is
    raised: HDF5 writes deferred data and metadata when a file is closed, so a
    file being written is only complete once its ``close()`` succeeded.
    """
    error = None
    for file in files:
        try:
            if file.isopen():
                file.close()
        except Exception as exc:
            if error is None:
                error = exc
    if strict and error is not None:
        raise error


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
    fh.createVariable(id, kind, shape, **_compression_kwds(fh, shape, complevel, kind))


#: Largest chunk (bytes) of a time-dependent variable; netCDF-C's default.
MAX_CHUNK_BYTES = 4 * 1024 * 1024

_ITEMSIZE = {"f": 4, "d": 8, "i": 4, "c": 1}


def _compression_kwds(fh, shape, complevel=None, kind="d"):
    """Return netCDF4 compression keywords for createVariable, when applicable.

    Compression (HDF5 deflate) is only available for the HDF5 based NETCDF4
    formats. Scalar variables (no dimensions) cannot be chunked, so they are
    left uncompressed. Variables over the time steps get one step per chunk
    (see :func:`_time_chunks`).
    """
    level = config.compression_level if complevel is None else complevel
    if not level or not shape:
        return {}
    # zlib/chunking is only supported by the HDF5 based NETCDF4* data models.
    if not str(getattr(fh, "data_model", "")).startswith("NETCDF4"):
        return {}
    kwds = {"zlib": True, "complevel": int(level), "shuffle": True}
    chunks = _time_chunks(fh, shape, _ITEMSIZE.get(kind, 8))
    if chunks is not None:
        kwds["chunksizes"] = chunks
    return kwds


def _time_chunks(fh, shape, itemsize):
    """Chunk shape ``(1, ...)`` for a ``(time_step, ...)`` variable, else None.

    A compressed chunk is decompressed whole, and netCDF-C's default chunks
    of a fixed-length time dimension span many steps, so reading one step
    would decompress all of them. One step per chunk (the layout netCDF-C
    uses for an unlimited time dimension) keeps single-step reads cheap; the
    remaining dimensions are split only when a chunk would exceed
    :data:`MAX_CHUNK_BYTES`. One-dimensional variables (the time values)
    keep the default chunking: they are read whole.
    """
    if len(shape) < 2 or shape[0] != "time_step":
        return None
    sizes = [max(1, len(fh.dimensions[name])) for name in shape[1:]]
    budget = max(1, MAX_CHUNK_BYTES // max(1, int(itemsize)))
    chunks = list(sizes)
    for i in range(len(chunks)):
        rest = int(np.prod(chunks)) // chunks[i]
        if chunks[i] * rest <= budget:
            break
        chunks[i] = max(1, min(chunks[i], budget // max(1, rest)))
    return (1, *chunks)


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
