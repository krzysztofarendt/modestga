import numpy as np


def mishra_bird(x, *args):
    """Mishra's Bird constrained function with 2 parameters.

    To be used in the constrained optimization examples.

    When subject to:
        (x[0] + 5) ** 2 + (x[1] + 5) ** 2 < 25
    the global minimum is at f(-3.1302, -1.5821) = -106.7645

    Bounds: -10 <= x[0] <= 0
           -6.5 <= x[1] <= 0

    Reference:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    fx = (
        np.sin(x[1]) * np.exp((1 - np.cos(x[0])) ** 2)
        + np.cos(x[0]) * np.exp((1 - np.sin(x[1])) ** 2)
        + (x[0] - x[1]) ** 2
    )

    return fx


def mishra_bird_constr(x, *args):
    """Constraint for the Mishra's Bird function."""
    fx = (x[0] + 5) ** 2 + (x[1] + 5) ** 2 - 25

    return fx * -1
