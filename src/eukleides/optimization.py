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


def polynomial_decrease_step(step_number: int, initial_alpha: float = 1.0, exponent: float = 0.5):
    assert exponent > 0.0
    assert initial_alpha > 0.0
    assert step_number >= 0
    return initial_alpha / (1 + step_number)**exponent


class EarlyStopper:
    """ After a given number of attempts of increasing the objective, it stops. """
    def __init__(self, max_fails: int = 3, direction: str = 'maximize'):
        assert direction in {'minimize', 'maximize'}
        self.max_fails = max_fails
        self.best_objective = None
        self.fails = None
        self.direction = direction

    def is_better(self, new_objective):
        if self.direction == 'maximize':
            return new_objective > self.best_objective
        if self.direction == 'minimize':
            return self.best_objective > new_objective
        raise ValueError(f'Optimization direction not understood: {self.direction}')

    def start(self):
        self.fails = 0
        return self

    def reset(self):
        self.fails = 0
        self.best_objective = None
        return self

    def stop(self, new_objective):
        if self.best_objective is None:
            self.best_objective = new_objective
            return False
        if self.is_better(new_objective):
            print(f'Improved by {abs(new_objective - self.best_objective)}')
            self.best_objective = new_objective
            self.fails = 0
        else:
            print(f'No improvement, attempt {self.fails} out of {self.max_fails}.')
            self.fails += 1
        if self.fails + 1 >= self.max_fails:
            print(f'Stop: {self.max_fails} failed attempts.')
            return True
        return False
