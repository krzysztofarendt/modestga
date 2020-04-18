import logging
import numpy as np
from modestga.ga import OptRes


def minimize(fun, bounds, x0=None, args=(), callback=None, options={}):
    """Monte Carlo optimization.

    :param fun: function to be minimized
    :param bounds: tuple, parameter bounds
    :param x0: numpy 1D array, initial parameters
    :param args: tuple, positional arguments to be passed to `fun`
    :param callback: function, called after every generation
    :param options: dict, GA options
    :return: OptRes, optimization result
    """
    np.set_printoptions(precision=3)
    log = logging.getLogger(name='minimize(MC)')

    def norm(x: np.array, b: np.array) -> np.array:
        return b[:, 0] + x * (b[:, 1] - b[:, 0])

    # Bounds to np.array:
    # bounds[:, 0] - lower bounds
    # bounds[:, 1] - upper bounds
    bounds = np.array(bounds)
    xinit = norm(np.random.uniform(low=0., high=1.), bounds)

    log.debug(xinit)



    # Optimization result
    best = None
    exitmsg = None
    ng = None
    nfev = None
    fx = None

    res = OptRes(
        x = None,
        message = exitmsg,
        ng = ng,
        nfev = nfev,
        fx = fx
    )

    log.info(res)

    return res


if __name__ == "__main__":
    logging.basicConfig(filename='mc.log', level='DEBUG', filemode='w')

    def fun(x, *args):
        """Noisy function to be minimized"""
        return np.sum(x ** 2) + np.random.random()
    
    bounds = [(0, 10) for i in range(10)]

    res = minimize(fun, bounds)
