"""PyVista writer plugin: ``mesh.save("model.exo")``.

exovista registers :func:`save_exodus` in the ``pyvista.writers`` entry-point
group for the ``.exo``, ``.e``, ``.ex2``, ``.exii`` and ``.gen`` extensions
(``.g`` is left alone: PyVista reads it as BYU geometry).  PyVista discovers
it on the first save to one of those extensions, so after
``pip install exovista`` ::

    import pyvista as pv

    mesh = pv.Sphere().delaunay_3d()
    mesh.save("sphere.exo")
    z = mesh.points[:, 2]
    mesh.save("sphere.e", times=[0.0, 1.0], node_fields={"T": [z, 2 * z]})

Every keyword argument that :meth:`pyvista.DataObject.save` does not consume
itself is forwarded to :func:`exovista.write_exo` (``surface=``,
``block_names=``, ``times=``, ``node_fields=``, ``compression_level=``, ...).

PyVista's own ``save`` parameters never reach this writer:

* ``binary`` -- ExodusII (netCDF) files are always binary; ignored.
* ``compression`` -- PyVista's VTK XML option; it is ignored for ExodusII.
  Use ``compression_level=`` (0 disables, 1-9 zlib level) instead.
* ``texture`` -- PLY only; ignored by PyVista for other formats.
* ``recompute_normals`` (``PolyData.save``) -- STL/PLY only; ignored.

If one of them is passed to :func:`save_exodus` directly (or a future
PyVista forwards it), ``binary`` and ``recompute_normals`` are ignored,
``compression`` is mapped (``None``/``False`` -> ``compression_level=0``,
``"zlib"`` -> the default level, anything else is rejected) and a
``texture`` raises :class:`ValueError`.

exovista requires PyVista >= 0.49, which always has the registry.  With an
older PyVista forced into the environment (< 0.48 has no writer registry),
importing exovista still works but ``mesh.save("x.exo")`` raises PyVista's
"Invalid file extension" ``ValueError``; call :func:`exovista.write_exo`
directly there.
"""

from __future__ import annotations

from typing import Any

__all__ = ["save_exodus"]

#: File extensions registered in ``pyvista.writers`` (see pyproject.toml).
EXTENSIONS = (".exo", ".e", ".ex2", ".exii", ".gen")


def save_exodus(dataset: Any, path: str, /, **kwargs: Any) -> None:
    """Write ``dataset`` to the ExodusII file ``path`` (PyVista writer plugin).

    This is the handler PyVista calls for ``dataset.save(path, **kwargs)``;
    it is equivalent to ``exovista.write_exo(path, dataset, **kwargs)``.

    Parameters
    ----------
    dataset : pyvista.DataSet or pyvista.MultiBlock
        The mesh to write (the ``volume`` argument of
        :func:`exovista.write_exo`).
    path : str
        Output file name.
    **kwargs
        Keyword arguments of :func:`exovista.write_exo`.  PyVista's own
        ``save`` options are handled as described in the module
        documentation.
    """
    from .write_pyvista import write_exo

    kwargs.pop("binary", None)
    kwargs.pop("recompute_normals", None)
    if kwargs.pop("texture", None) is not None:
        raise ValueError(
            "texture= is only supported when saving PLY files; ExodusII files "
            "have no texture coordinates (store the array as point data instead)"
        )
    if "compression" in kwargs:
        _apply_pyvista_compression(kwargs, kwargs.pop("compression"))
    write_exo(path, dataset, **kwargs)


def _apply_pyvista_compression(kwargs: dict, compression: Any) -> None:
    """Translate PyVista's ``compression=`` into ``compression_level=``."""
    if compression is None or compression is False:
        level = 0
    elif isinstance(compression, str) and compression.lower() == "zlib":
        level = None  # write_exo's default zlib level
    else:
        raise ValueError(
            f"compression={compression!r} is not supported for ExodusII files; "
            "netCDF supports zlib only: use compression_level=0 to disable "
            "compression or compression_level=1..9 to choose the zlib level"
        )
    explicit = kwargs.get("compression_level")
    if explicit is None:
        if level is not None:
            kwargs["compression_level"] = level
    elif level == 0 and explicit != 0:
        raise ValueError(
            f"compression={compression!r} conflicts with "
            f"compression_level={explicit!r}"
        )
