"""
Polytope regression: find a convex combination of the vertices of the polytope that determine the
given point.
"""
import logging
from typing import Callable

import numpy as np

from eukleides import ConvexHull
from eukleides import gradient_updates as gu


def softmax(x: np.ndarray) -> np.array:
    expo = np.exp(x)
    return expo / np.sum(expo)


def get_convex_combination(hull: ConvexHull, lin_coefs: np.ndarray) -> np.array:
    coefs = softmax(lin_coefs)
    return hull.base @ coefs


def calc_error(hull: ConvexHull, target: np.ndarray, lin_coefs: np.ndarray):
    combo = get_convex_combination(hull, lin_coefs)
    return target - combo


def calc_loss(hull: ConvexHull, target: np.ndarray, lin_coeffs: np.ndarray):
    err = calc_error(hull, target, lin_coeffs)
    return np.dot(err, err)


def comb_gradient(lin_coefs: np.ndarray):
    coefs = softmax(lin_coefs)
    return np.diag(coefs) - coefs.reshape((-1, 1)) @ coefs.reshape((1, -1))


def loss_gradient(hull: ConvexHull, target: np.ndarray, lin_coefs: np.ndarray) -> np.array:
    return -(calc_error(hull, target, lin_coefs) @ hull.base) @ comb_gradient(lin_coefs)


# pylint: disable=too-many-arguments
def polyreg(
    hull: ConvexHull,
    target: np.ndarray,
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
