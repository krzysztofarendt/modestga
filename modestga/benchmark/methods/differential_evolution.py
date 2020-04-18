from scipy.optimize import differential_evolution
import numpy as np
from modestga.ga import OptRes


def minimize(fun, bounds, x0=None, args=(), callback=None, options={}):
    """Differential Evolution from scipy.

    This is a thin wrapper over the scipy implementation::

    scipy.optimize.differential_evolution(
        func, bounds, args=(), strategy='best1bin', maxiter=1000,
        popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7,
        seed=None, callback=None, disp=False, polish=True,
        init='latinhypercube', atol=0, updating='immediate',
        workers=1, constraints=()
    )

    Default options::

        opts = {
            'generations': 1000,
            'pop_size': 100,
            'tol': 1e-6,
            'polish': False
        }

    :param fun: function to be minimized
    :param bounds: tuple, parameter bounds
    :param x0: numpy 1D array, initial parameters
    :param args: tuple, positional arguments to be passed to `fun`
    :param callback: function, called after every generation
    :param options: dict, DE options
    :return: OptRes, optimization result
    """
    pass  #TODO
