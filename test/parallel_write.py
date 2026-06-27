#!/usr/bin/env python
import glob
import exovista
from exovista.util import working_dir


def test_exodusii_parallel_write(tmpdir, datadir, datafile):
    basefile = datafile("edges.base.exo")
    with working_dir(tmpdir):
        name = "edges"
        files = glob.glob(f"{datadir}/{name}.exo.*")
        file = exovista.exo_file(*files)
        joined = file.write(f"{name}.exo")
        base = exovista.exo_file(basefile)
        dimensions = "~four|len_line|len_string"
        assert exovista.allclose(base, joined, dimensions=dimensions, variables=None)
