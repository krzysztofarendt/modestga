import time

import numpy as np


def rastrigin(x, *args):
    """Rastrigin function to be minimized.

    Global minimum y=0 at x=[0, 0, ..., 0].
    `x` can have any size.
    """
    A = 100
    n = x.size
    y = A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))
    return y


def rastrigin_delayed(x, *args):
    """Rastrigin function to be minimized.

    Global minimum y=0 at x=[0, 0, ..., 0].
    `x` can have any size.
    """
    time.sleep(0.01)
    A = 100
    n = x.size
    y = A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))
    return y
