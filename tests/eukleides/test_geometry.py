import numpy as np

from eukleides import geometry as eg


def test_hyper_plane(horizontal_plane):
    point = np.array([0.5, 9.0, 1.0])
    assert horizontal_plane.contains(point)


def test_project(horizontal_plane):
    """
    Test the projection on the horizontal_plane, and also if the projection of random points
    on random planes belong to those planes.
    """
    point = np.array([2.0, 6.0, 2.0])
    projection = eg.project(point, horizontal_plane)
    assert horizontal_plane.contains(projection)
    assert np.array_equal(projection, np.array([0.5, 6.0, 2.0]))

    np.random.seed(0)
    for _ in range(10):
        random_plane = eg.HyperPlane(np.random.random(size=(10, )), np.random.random())
        random_point = np.random.random(size=(10, ))
        assert random_plane.contains(eg.project(random_point, random_plane))
