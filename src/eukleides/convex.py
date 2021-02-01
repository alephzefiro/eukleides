""" The function polyreg implements the polytope regression. """
from typing import List, Optional, Callable

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
