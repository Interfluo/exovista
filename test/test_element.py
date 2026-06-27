"""Tests for exovista.element geometry helpers."""
import numpy as np

from exovista.element import factory


def test_quad4_center_is_corner_average():
    # Unit square; centroid is (0.5, 0.5).
    coord = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    quad = factory("quad4", coord)
    assert np.allclose(quad.center[:2], [0.5, 0.5])


def test_quad4_center_offset_square():
    # A square translated away from the origin must report its true centroid,
    # not a value derived from the x-coordinate alone.
    coord = np.array([[2.0, 3.0], [4.0, 3.0], [4.0, 5.0], [2.0, 5.0]])
    quad = factory("quad4", coord)
    assert np.allclose(quad.center[:2], [3.0, 4.0])


def test_quad4_volume_unit_square():
    coord = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    quad = factory("quad4", coord)
    assert np.isclose(quad.volume, 1.0)


def test_hex8_center_and_volume_unit_cube():
    coord = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    hexa = factory("hex8", coord)
    assert np.allclose(hexa.center, [0.5, 0.5, 0.5])
    assert np.isclose(hexa.volume, 1.0)


def test_tri3_center():
    coord = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    tri = factory("tri3", coord)
    assert np.allclose(tri.center[:2], [1.0, 1.0])
