import numpy as np


def rmse(y1, y2):
    """
    Root-mean-square error.

    :param y1: 1D numpy array
    :param y2: 1D numpy array
    :return: float
    """
    assert y1.size == y2.size, "y1 and y2 must have the same size"
    n = y1.size
    e = (1 / n * np.sum((y1 - y2) ** 2)) ** 0.5
    return e