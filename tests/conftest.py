""" Define the fixtures. """
import pytest
import numpy as np

from eukleides import geometry as eg


@pytest.fixture
def horizontal_plane():
    """ Horizontal plane in R^3 with equation x = 0.5 """
    return eg.HyperPlane(np.array([1.0, 0.0, 0.0]), 0.5)
