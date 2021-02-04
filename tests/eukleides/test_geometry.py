import numpy as np

from eukleides import HyperPlane, LinearConstraint, Polytope


def test_hyper_plane(horizontal_plane):
    point = np.array([0.5, 9.0, 1.0])
    assert horizontal_plane.contains(point)


def test_hyper_plane_project(horizontal_plane):
    """
    Test the projection on the horizontal_plane, and also if the projection of random points
    on random planes belong to those planes.
    """
    point = np.array([2.0, 6.0, 2.0])
    projection = horizontal_plane.project(point)
    assert horizontal_plane.contains(projection)
    assert np.array_equal(projection, np.array([0.5, 6.0, 2.0]))

    np.random.seed(0)
    for _ in range(10):
        random_plane = HyperPlane(np.random.random(size=(10, )), np.random.random())
        random_point = np.random.random(size=(10, ))
        assert random_plane.contains(random_plane.project(random_point))


def test_linear_constraint():
    lcon = LinearConstraint(normal=np.array([1.0, 1.0]), constant=1.0, side='leq')
    assert not lcon.contains(np.array([0.5, 9.0]))
    assert lcon.contains(np.array([0.3, 0.7]))


def test_constraint_project():
    """
    Test the projection on the horizontal_plane, and also if the projection of random points
    on random planes belong to those planes.
    """
    point = np.array([2.0, 2.0])
    constraint = LinearConstraint(np.array([1.0, 1.0]), 1.0)
    assert np.array_equal(constraint.project(point), np.array([0.5, 0.5]))

    np.random.seed(0)
    for _ in range(10):
        random_constr =  LinearConstraint(
            2 * np.random.random(size=(10, )) - 1.0,
            10.0 * np.random.random(),
            side=np.random.choice(['leq', 'geq'])
        )

        random_point = np.random.random(size=(10, ))
        assert random_constr.contains(random_constr.project(random_point))
