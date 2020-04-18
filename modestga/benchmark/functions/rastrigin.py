import numpy as np


def rastrigin(x):
    """Rastrigin function to be minimized.

    Global minimum y=0 at x=[0, 0, ..., 0].
    `x` can have any size.
    """
    A = 100
    n = x.size
    y = A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))
    return y
