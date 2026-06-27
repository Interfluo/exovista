"""Regression tests for joining parallel ExodusII files.

These exercise the in-repository ``noh.exo.3.*`` decomposition, so they do not
depend on any external/upstream serial reference files.
"""
import glob
import os

import exovista
from exovista.util import working_dir


def _noh_files(datadir):
    return sorted(glob.glob(os.path.join(datadir, "noh.exo.3.*")))


def test_node_set_dist_facts_consistent(datadir):
    """num_dist_facts must agree with the actual dist_facts array.

    Previously the parallel reader cached the raw ``None`` for a node set with
    no distribution factors while returning a ``np.ones`` fallback, so
    ``num_dist_facts`` reported a non-zero count even though ``dist_facts`` was
    ``None``. That inconsistency crashed the join.
    """
    exof = exovista.exo_file(*_noh_files(datadir))
    try:
        for ns in exof.node_sets():
            if ns.dist_facts is None:
                assert not ns.num_dist_facts
            else:
                assert ns.num_dist_facts == len(ns.dist_facts)
        # Repeated calls (cache hit) must return the same thing as the first.
        for set_id in exof.get_node_set_ids():
            first = exof.get_node_set_dist_facts(set_id)
            second = exof.get_node_set_dist_facts(set_id)
            assert (first is None and second is None) or (first == second).all()
    finally:
        exof.close()


def test_parallel_join_roundtrip(tmpdir, datadir):
    """Joining the decomposed files yields a readable serial file."""
    exof = exovista.exo_file(*_noh_files(datadir))
    expected = {
        "title": exof.title(),
        "num_dim": exof.num_dimensions(),
        "num_nodes": exof.num_nodes(),
        "num_elems": exof.num_elems(),
        "num_elem_blk": exof.num_elem_blk(),
        "num_node_sets": exof.num_node_sets(),
    }

    with working_dir(tmpdir.strpath):
        joined_path = exof.write("noh.exo")
        with exovista.File(joined_path) as joined:
            assert joined.title() == expected["title"]
            assert joined.num_dimensions() == expected["num_dim"]
            assert joined.num_nodes() == expected["num_nodes"]
            assert joined.num_elems() == expected["num_elems"]
            assert joined.num_elem_blk() == expected["num_elem_blk"]
            assert joined.num_node_sets() == expected["num_node_sets"]
            assert joined.get_node_set_ids().tolist() == \
                exof.get_node_set_ids().tolist()
    exof.close()
