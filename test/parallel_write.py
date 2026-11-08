#!/usr/bin/env python
import glob
import exovista.legacy
from exovista.util import working_dir


def test_exodusii_parallel_write(tmpdir, datadir, datafile):
    basefile = datafile("edges.base.exo")
    with working_dir(tmpdir):
        name = "edges"
        files = glob.glob(f"{datadir}/{name}.exo.*")
        file = exovista.legacy.exo_file(*files)
        joined = file.write(f"{name}.exo")
        base = exovista.legacy.exo_file(basefile)
        dimensions = "~four|len_line|len_string"
        assert exovista.legacy.allclose(
            base, joined, dimensions=dimensions, variables=None
        )
