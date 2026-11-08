"""exovista: write PyVista meshes to ExodusII and read them back.

Element blocks, side sets, node sets and time-varying node, element and
global fields are written with :func:`write_exo`; :func:`read_exo` and the
``read_*`` helpers read ExodusII files (serial or parallel decompositions)
back into PyVista objects and NumPy arrays without loss of precision.

Installing exovista also registers a PyVista writer plugin
(:mod:`exovista._pyvista_plugin`), so ``mesh.save("model.exo")`` writes an
ExodusII file with :func:`write_exo` (extensions ``.exo``, ``.e``, ``.ex2``,
``.exii`` and ``.gen``) and forwards its keyword arguments.

The 1.x low-level file API (``File``, ``exodusii_file``, ``allclose``, ...)
lives in :mod:`exovista.legacy`.  Accessing it from the top-level package
still works but emits a :class:`DeprecationWarning`; it will be removed in
exovista 3.0.

(Not to be confused with NASA's ExoVista exoplanet-imaging package.)
"""

import logging as _logging
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _metadata_version

from . import legacy as _legacy
from .read_pyvista import (
    read_element_fields,
    read_exo,
    read_global_fields,
    read_node_fields,
    read_node_sets,
    read_side_sets,
)
from .write_pyvista import write_exo

try:
    __version__ = _metadata_version("exovista")
except _PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

# Library logging etiquette: never configure logging, only make sure the
# package logger does not fall back to the "last resort" stderr handler.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "write_exo",
    "read_exo",
    "read_side_sets",
    "read_node_sets",
    "read_node_fields",
    "read_element_fields",
    "read_global_fields",
    "__version__",
]

#: Names of the 1.x top-level API that now live in :mod:`exovista.legacy`.
_LEGACY_NAMES = frozenset(_legacy.__all__)


def _unbind_shadowing_submodules():
    """Let the deprecated legacy names reach ``__getattr__``.

    ``allclose``, ``similar``, ``lineout`` and ``exoread`` are both legacy
    names and submodule names.  Importing :mod:`exovista.legacy` imported
    those submodules, which binds the *module* objects as attributes of this
    package and would shadow the deprecated callables served by
    ``__getattr__``.  Drop those bindings; the submodules stay in
    ``sys.modules`` (``importlib.import_module("exovista.lineout")`` returns
    the module, and ``import exovista.lineout`` does not re-bind it).  Any
    access through the package attribute, including
    ``import exovista.lineout as m`` and ``from exovista import lineout``,
    gives the deprecated callable (with the warning), as in 1.x where these
    names were functions too.
    """
    namespace = globals()
    for name in _LEGACY_NAMES.intersection(namespace):
        del namespace[name]


_unbind_shadowing_submodules()
del _unbind_shadowing_submodules


def __getattr__(name):
    if name in _LEGACY_NAMES:
        import warnings

        warnings.warn(
            f"exovista.{name} is deprecated and will be removed in exovista 3.0; "
            f"import it from exovista.legacy instead "
            f"(from exovista.legacy import {name})",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
