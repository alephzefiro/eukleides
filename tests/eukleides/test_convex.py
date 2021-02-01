import numpy as np

from eukleides import LinearConstraint, Polytope


def test_linear_constraint():
    lcon = LinearConstraint(normal=np.array([1.0, 1.0]), constant=1.0, side='leq')
    assert not lcon.contains(np.array([0.5, 9.0]))
    assert lcon.contains(np.array([0.3, 0.7]))


def test_project():
    """
    Test the projection on the horizontal_plane, and also if the projection of random points
    on random planes belong to those planes.
    """
    point = np.array([2.0, 2.0])
    poly = Polytope(
        [
            LinearConstraint(np.array([1.0, 1.0]), 1.0),
            LinearConstraint(np.array([-1.0, 1.0]), 1.0),
            LinearConstraint(np.array([1.0, -1.0]), 1.0),
            LinearConstraint(np.array([-1.0, -1.0]), 1.0),
        ]
    )
    assert np.array_equal(poly.project(point), np.array([0.5, 0.5]))

    np.random.seed(0)
    for _ in range(10):
        random_polytope = Polytope(
            [LinearConstraint(np.random.random(size=(10, )), np.random.random()) for _ in range(5)]
        )
        random_point = np.random.random(size=(10, ))
        assert random_polytope.contains(random_polytope.project(random_point))
