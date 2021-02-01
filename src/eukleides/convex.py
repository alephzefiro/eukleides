""" The function polyreg implements the polytope regression. """
import logging
from typing import List, Optional

import numpy as np

from eukleides.geometry import HyperPlane


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

    def project(self, point: np.ndarray) -> np.array:
        """ Project a point into the polytope. """
        for constr in self.constraints:
            if not constr.contains(point):
                prev_point = str(point)
                point = constr.project(point)
                self.logger.info(
                    f'Constraint {constr} not satisfied. Project {prev_point} to {point}'
                )
        return point
