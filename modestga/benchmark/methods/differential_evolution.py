import logging

import numpy as np
from modestga.ga import OptRes
from scipy.optimize import differential_evolution


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

        options = {
            'generations': 1000,
            'pop_size': 100,
            'tol': 1e-3,
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
    np.set_printoptions(precision=3)
    log = logging.getLogger(name="minimize(DE)")

    opts = {
        "generations": 1000,
        "pop_size": 100,
        "tol": 1e-3,
        "polish": False,
        "mut_rate": (0.5, 1),
    }

    for k in options:
        if k in opts:
            log.info("Override default option: {}={}".format(k, options[k]))
            opts[k] = options[k]
        else:
            raise KeyError("Option '{}' not found".format(k))

    res_de = differential_evolution(
        fun,
        bounds,
        args,
        maxiter=opts["generations"],
        popsize=opts["pop_size"],
        tol=opts["tol"],
        mutation=opts["mut_rate"],
        polish=False,
        disp=True,
        workers=1,
    )

    res = OptRes(
        x=res_de.x,
        message=res_de.message,
        ng=res_de.nit,
        nfev=res_de.nfev,
        fx=res_de.fun,
    )

    return res


if __name__ == "__main__":
    # Example
    logging.basicConfig(level="INFO")

    from modestga.benchmark.functions import rastrigin

    N = 50
    bounds = [(-5.12, 5.12) for i in range(N)]
    res = minimize(rastrigin, bounds, x0=None)
    print(res)
