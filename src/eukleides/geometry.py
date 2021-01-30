"""
Definition of basic elements of Euclidean geometry.
"""
from typing import Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class HyperPlane:
    """
    Affine space of codimension 1.

    Defined by its normal vector 'normal' and the 'constant', such that a point x belongs to the
    hyperplane iff $normal \cdot x + constant = 0$.
    """
    tol = 1e-8

    def __init__(self, normal: np.array, constant: float = 0):
        self.normal = normal
        self.constant = constant

    @property
    def dim(self) -> Tuple[int, ...]:
        """ Dimension as the shape of the normal vector. To define how to treat ndarrays. """
        return self.normal.shape

    def contains(self, point: np.array) -> bool:
        """ Check if the point belongs to the plane. """
        assert self.dim == point.shape, f'Dimension mismatch: {self.dim} != {point.shape}'
        scalar_product = np.dot(self.normal, point)
        return abs(scalar_product - self.constant) < self.tol
