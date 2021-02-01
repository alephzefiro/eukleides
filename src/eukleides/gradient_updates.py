""" Numeric methods to implement the gradient flow for continuous optimization problems. """
from typing import Callable

import numpy as np


Gradient = Callable[[np.ndarray], np.ndarray]


def euler_update(value: np.ndarray, gradient_func: Gradient, alpha: float):
    return alpha * gradient_func(value)


def improved_euler_update(value: np.ndarray, gradient_func: Gradient, alpha: float):
    first_gradient = gradient_func(value)
    second_gradient = gradient_func(value + alpha * first_gradient)
    return 0.5 * alpha * (first_gradient + second_gradient)


def runge_kutta_update(value: np.ndarray, gradient_func: Gradient, alpha: float):
    first_gradient = gradient_func(value)
    second_gradient = gradient_func(value + 0.5 * alpha * first_gradient)
    third_gradient = gradient_func(value + 0.5 * alpha * second_gradient)
    fourth_gradient = gradient_func(value + alpha * third_gradient)
    return (alpha / 6.0) * (
        first_gradient + 2.0 * second_gradient + 2.0 * third_gradient + fourth_gradient
    )
