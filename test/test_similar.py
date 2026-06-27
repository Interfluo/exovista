"""Tests for exovista.similar structural comparison."""
import os

import numpy as np
import pytest

import exovista


def _write_quad(path, block_id=1, title="t"):
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    conn = np.array([[1, 2, 3, 4]])
    with exovista.File(path, mode="w") as e:
        e.put_init(title, 2, 4, 1, 1, 0, 0)
        e.put_coord(x, y)
        e.put_element_block(block_id, "QUAD", 1, 4)
        e.put_element_conn(block_id, conn)
    return path


def test_similar_identical_files(tmp_path):
    f1 = _write_quad(os.path.join(tmp_path, "a.exo"))
    f2 = _write_quad(os.path.join(tmp_path, "b.exo"))
    assert exovista.similar(f1, f2) is True


def test_similar_detects_different_block_ids(tmp_path):
    # Identical in every respect except the element block ID. This exercises
    # the comparison that previously compared file1 against itself.
    f1 = _write_quad(os.path.join(tmp_path, "a.exo"), block_id=1)
    f2 = _write_quad(os.path.join(tmp_path, "b.exo"), block_id=7)
    with pytest.raises(ValueError):
        exovista.similar(f1, f2)


def test_similar_detects_node_count_mismatch(tmp_path):
    f1 = _write_quad(os.path.join(tmp_path, "a.exo"))

    f2 = os.path.join(tmp_path, "tri.exo")
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    conn = np.array([[1, 2, 3]])
    with exovista.File(f2, mode="w") as e:
        e.put_init("t", 2, 3, 1, 1, 0, 0)
        e.put_coord(x, y)
        e.put_element_block(1, "TRI", 1, 3)
        e.put_element_conn(1, conn)

    with pytest.raises(ValueError):
        exovista.similar(f1, f2)
