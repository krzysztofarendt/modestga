import numpy as np


def rosenbrock_2par(x, *args):
    """Rosenbrock function with 2 parameters.

    To be used in the constrained optimization examples.

    When subject to constraints:
        (x[0] - 1) ** 3 - x[1] + 1 <= 0
        x[0] + x[1] - 2 <= 0
    the global minimum is at f(1., 1.) = 0.

    Bounds: -1.5 <= x[0] <= 1.5
            -0.5 <= x[1] <= 2.5

    Reference:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    fx = (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    return fx


def rosenbrock_constr1(x, *args):
    """First constraint for rosenbrock_2par()."""
    fx = (x[0] - 1.0) ** 3 - x[1] + 1.0
    return fx * -1.0


def rosenbrock_constr2(x, *args):
    """Second constraint for rosenbrock_2par()."""
    fx = x[0] + x[1] - 2.0
    return fx * -1.0
