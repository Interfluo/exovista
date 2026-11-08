"""The ``exovista-read`` command: print ExodusII metadata or variable values.

Run ``exovista-read --help`` for usage.  ``exovista-read model.exo`` prints a
summary of the file; ``exovista-read -n temp -i -1 model.exo`` prints the node
variable ``temp`` at the last time step.  A parallel decomposition is read by
passing a glob pattern that matches all of its parts (quote it).

This command was installed as ``exoread.py`` in exovista 1.x; it was renamed
so that it no longer collides with the script of the upstream ``exodusii``
package.
"""

import errno
import os
import sys
import glob
import argparse

from .lineout import lineout
from .file import exodusii_file
from .parallel_file import parallel_exodusii_file


class Namespace(argparse.Namespace):
    def __init__(self, **kwargs):
        self.variables = []
        super(Namespace, self).__init__(**kwargs)

    def __setattr__(self, attr, value):
        if value:
            if attr in ("globalvar", "element", "face", "edge", "node"):
                type = "d" if attr == "edge" else attr[0]
                self.variables.append(f"{type}/{value[-1]}")
        super(Namespace, self).__setattr__(attr, value)


def main(argv=None, file=None):
    """Extracts specified variable values from a given file name. By default data is
    written to stdout in tabular form. If no variables are selected, then the file
    meta data is written instead.

    Default for global variables is to write out the values for all times.

    Default for spatial variables is to write out the values for each object for the
    last time slab.
    """

    from . import __version__

    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser(prog="exovista-read", description=main.__doc__)
    p.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s (exovista {__version__})",
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "-g",
        "--global",
        action="append",
        dest="globalvar",
        help="Select a mesh global variable name to extact",
    )
    g.add_argument(
        "-e",
        "--element",
        action="append",
        help="Select a mesh element variable name to extact",
    )
    g.add_argument(
        "-f",
        "--face",
        action="append",
        help="Select a mesh face variable name to extact",
    )
    g.add_argument(
        "-d",
        "--edge",
        action="append",
        help="Select a mesh edge variable name to extact",
    )
    g.add_argument(
        "-n",
        "--node",
        action="append",
        help="Select a mesh node variable name to extact",
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "-t",
        "--time",
        type=float,
        help="Output the variable at this time.  Closest time value is chosen.",
    )
    g.add_argument(
        "-i",
        "--index",
        type=int,
        help="Output the variable at this time step index. "
        "A value of -1 means the last time step in the file. "
        "Note that exodus time steps start at one, while this starts at zero.",
    )
    g.add_argument(
        "-c",
        "--cycle",
        type=int,
        help="Output the variable at this cycle number. "
        "Numbers start at zero. A value of -1 means the last time step in the file.",
    )

    p.add_argument(
        "--object-index",
        action="store_true",
        default=False,
        help="For non-global variables, include the object index in the output.",
    )
    p.add_argument(
        "--nolabels",
        action="store_true",
        default=False,
        help="Do not write the variable names and units to the output.",
    )

    p.add_argument(
        "-L",
        "--lineout",
        metavar="{x|X|<real>}/{y|Y|<real>}/{z|Z|<real>}[/T<real tolerance>]",
        type=lineout.from_cli,
        help=lineout.__doc__,
    )

    p.add_argument("file", help="The ExodusII database file.")

    args = p.parse_args(argv, namespace=Namespace())
    try:
        f = exo_file(args.file)
        try:
            if "num_dim" not in f.files[0].dimensions:
                raise ValueError(f"{args.file}: not an ExodusII file")
            if args.variables:
                f.print(
                    *args.variables,
                    time=args.time,
                    index=args.index,
                    cycle=args.cycle,
                    lineout=args.lineout,
                    file=file,
                    labels=not args.nolabels,
                )
            else:
                f.describe(file=file)
        finally:
            f.close()
    except BrokenPipeError:
        # the reader of the output went away (``exovista-read x.exo | head``)
        if file is None or file is sys.stdout:
            _silence_stdout()
        return 1
    except (OSError, ValueError, KeyError, IndexError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        p.exit(1, f"exovista-read: error: {message}\n")
    return 0


def _silence_stdout():
    """Point stdout at devnull so that flushing it at exit cannot fail again."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        pass  # stdout has no file descriptor (e.g. captured by a test runner)


def exo_file(filename, *files):
    files = _find_files(filename, *files)
    if len(files) > 1:
        f = parallel_exodusii_file(*files)
    elif len(files) == 1:
        f = exodusii_file(files[0], mode="r")
    return f


def _find_files(*files):

    found = []
    for file in files:
        globbed_files = glob.glob(file)
        if not globbed_files:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", file)
        found.extend(globbed_files)
    return found


if __name__ == "__main__":
    sys.exit(main())
