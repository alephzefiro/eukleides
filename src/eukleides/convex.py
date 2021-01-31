""" The function polyreg implements the polytope regression. """
from typing import List, Optional, Callable
import logging

import numpy as np

from eukleides.geometry import HyperPlane
from eukleides import gradient_updates as gu


class ConvexHull:
    """ Convex hull generated as convex combination of a finite set of points. """
    def __init__(self, points: List[np.array]):
        _dimensions = [vec.shape for vec in points]
        assert all(dim == _dimensions[0] for dim in _dimensions)
        self.points: List[np.array] = points
        self._base: Optional[np.array] = None  # pylint: disable=E1136

    @property
    def base(self) -> np.array:
        """ An array with all the points stacked, where the first index indexes the points. """
        if self._base is None:
            self._base = np.array(self.points).T
        return self._base

    @property
    def num_points(self) -> int:
        """ Number of points of the convex hull. """
        return len(self.points)


class LinearConstraint(HyperPlane):
    """
    Extends the hyperplane with an extra method to check if the desired (in)equality is satisfied.
    """
    def __init__(self, normal: np.array, constant: float = 0.0, side: str = 'leq'):
        super().__init__(normal, constant)
        assert side in {'eq', 'leq', 'geq'}
        self.side = side

    def check(self, point: np.array):
        if self.side == 'eq':
            return self.contains(point)
        scalar_prod = np.dot(self.normal, point)
        if self.side == 'leq':
            return scalar_prod <= self.constant + self.tol
        if self.side == 'geq':
            return scalar_prod >= self.constant - self.tol
        raise ValueError(f'side should be eq, leq or geq, found {self.side}')


class Polytope:
    """ Polytope of any dimension, defined as intersection of hyperplanes or half-spaces. """
    def __init__(self, constraints: List[LinearConstraint]):
        self.dim = constraints[0].normal.shape
        for constr in constraints:
            assert constr.normal.shape == self.dim
        self._constraints = constraints

    @property
    def constraints(self):
        return self._constraints

    def add_constraint(self, constraint: LinearConstraint):
        assert constraint.normal.shape == self.dim
        self._constraints.append(constraint)

    def check_all(self, point: np.array):
        return all(constr.check(point) for constr in self.constraints)


def softmax(x: np.array) -> np.array:
    expo = np.exp(x)
    return expo / np.sum(expo)


def get_convex_combination(hull: ConvexHull, lin_coefs: np.array) -> np.array:
    coefs = softmax(lin_coefs)
    return hull.base @ coefs


def calc_error(hull: ConvexHull, target: np.array, lin_coefs: np.array):
    combo = get_convex_combination(hull, lin_coefs)
    return target - combo


def calc_loss(hull: ConvexHull, target: np.array, lin_coeffs: np.array):
    err = calc_error(hull, target, lin_coeffs)
    return np.dot(err, err)


def comb_gradient(lin_coefs: np.array):
    coefs = softmax(lin_coefs)
    return np.diag(coefs) - coefs.reshape((-1, 1)) @ coefs.reshape((1, -1))


def loss_gradient(hull: ConvexHull, target: np.array, lin_coefs: np.array) -> np.array:
    return -(calc_error(hull, target, lin_coefs) @ hull.base) @ comb_gradient(lin_coefs)


# pylint: disable=too-many-arguments
def polyreg(
    hull: ConvexHull,
    target: np.array,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 10000,
    update_method: Callable = gu.euler_update,
) -> np.array:
    """
    Given the convex hull of a point, use the optimization algorithm of choice to compute the
    coefficients whose softmax determine a convex combination of the hull vertices for the given
    target point. If the target point does not lie in the convex hull, the algorithm will converge
    to the point in the convex hull that minimizes the distance from the target, namely its
    projection.
    """
    def inverse_gradient(lin_coefs):
        return - loss_gradient(hull, target, lin_coefs)

    logger = logging.getLogger('polyreg')
    logger.info(f'using {update_method.__name__}')
    lin_coefs = np.random.normal(size=hull.num_points, scale=0.001)
    prev_loss = 100000.0
    for i in range(max_iter):
        loss = calc_loss(hull, target, lin_coefs)
        vecto = loss_gradient(hull, target, lin_coefs)
        speed = np.linalg.norm(vecto)
        if i % 100 == 0:
            logger.info(f'Iter {i}: loss = {loss:.5f}, speed = {speed:.5f}')
        old_lin_coefs = lin_coefs
        lin_coefs = lin_coefs + update_method(lin_coefs, inverse_gradient, alpha=alpha)
        if loss < tol:
            logger.info('converged.')
            break
        if loss > prev_loss:
            logger.info('loss increased, reducing the learning rate.')
            alpha *= 0.9
            lin_coefs = old_lin_coefs
        else:
            prev_loss = loss
    else:
        logger.warning('did not converge.')

    return lin_coefs
