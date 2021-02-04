"""
Definition of geometrical objects.
"""
import logging
from typing import Tuple, List, Optional

import numpy as np


class HyperPlane:
    """
    Affine space of codimension 1.

    Defined by its normal vector 'normal' and the 'constant', such that a point x belongs to the
    hyperplane iff $normal ⋅ x = constant$.
    """
    tol = 1e-8

    def __init__(self, normal: np.ndarray, constant: float = 0.0):
        self.normal = normal
        self.constant = constant

    @property
    def dim(self) -> Tuple[int, ...]:
        """ Dimension as the shape of the normal vector. To define how to treat ndarrays. """
        return self.normal.shape

    def contains(self, point: np.ndarray) -> bool:
        """ Check if the point belongs to the plane. """
        assert self.dim == point.shape, f'Dimension mismatch: {self.dim} != {point.shape}'
        scalar_product = np.dot(self.normal, point)
        return abs(scalar_product - self.constant) < self.tol

    def project(self, point: np.ndarray) -> np.ndarray:
        """
        Solve the equation (v is hyperplane normal vector and c the constant term)
        p \\cdot v - t |v|^2 + c = 0
        in t to determine the intersection of the parametric line.
        """
        t = (np.dot(point, self.normal) - self.constant) / np.dot(self.normal, self.normal)
        return point - t * self.normal


class LinearConstraint(HyperPlane):
    """
    Extends the hyperplane with an extra method to check if the desired (in)equality is satisfied.
    """
    def __init__(self, normal: np.ndarray, constant: float = 0.0, side: str = 'leq'):
        super().__init__(normal, constant)
        assert side in {'eq', 'leq', 'geq'}
        self.side = side

    def contains(self, point: np.ndarray):
        if self.side == 'eq':
            return super().contains(point)
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
        self.logger = logging.getLogger(__name__)

    @property
    def constraints(self):
        return self._constraints

    def add_constraint(self, constraint: LinearConstraint):
        assert constraint.normal.shape == self.dim
        self._constraints.append(constraint)

    def contains(self, point: np.ndarray):
        return all(constr.contains(point) for constr in self.constraints)

    def project(self, point: np.ndarray):
        """ Project a point into the polytope. """
        raise NotImplementedError('Work in progress.')



class ConvexHull:
    """ Convex hull generated as convex combination of a finite set of points. """
    def __init__(self, points: List[np.ndarray]):
        _dimensions = [vec.shape for vec in points]
        assert all(dim == _dimensions[0] for dim in _dimensions)
        self.points: List[np.ndarray] = points
        self._base: Optional[np.ndarray] = None  # pylint: disable=E1136

    @property
    def base(self) -> np.ndarray:
        """ An array with all the points stacked, where the first index indexes the points. """
        if self._base is None:
            self._base = np.array(self.points).T
        return self._base

    @property
    def num_points(self) -> int:
        """ Number of points of the convex hull. """
        return len(self.points)
