import logging

import numpy as np
from modestga.ga import OptRes


def minimize(fun, bounds, x0=None, args=(), callback=None, options={}):
    """Monte Carlo optimization.

    The options have similar names to GA to avoid confusion.
    The default options are::

        options = {
            'generations': 10000, # Max. number of generations (iterations)
            'pop_size': 1000,     # Population size (number of guesses per iteration)
            'tol': 1e-3,          # Solution tolerance
            'inertia': 1000,      # Max. number of non-improving generations
        }

    :param fun: function to be minimized
    :param bounds: tuple, parameter bounds
    :param x0: numpy 1D array, initial parameters
    :param args: tuple, positional arguments to be passed to `fun`
    :param callback: function, called after every generation
    :param options: dict, MC options
    :return: OptRes, optimization result
    """
    np.set_printoptions(precision=3)
    log = logging.getLogger(name="minimize(MC)")

    def norm(x: np.array, b: np.array) -> np.array:
        """Normalizes x with respect to bounds."""
        return b[:, 0] + x * (b[:, 1] - b[:, 0])

    def random_x(b: np.array):
        """Returns random parameters within bounds `b`."""
        rx = norm(np.random.uniform(low=0.0, high=1.0, size=bounds.shape[0]), b)
        return rx

    opts = {
        "generations": 1000,  # Max. number of generations
        "pop_size": 100,  # Population size
        "tol": 1e-3,  # Solution tolerance
        "inertia": 1000,  # Max. number of non-improving generations
    }

    for k in options:
        if k in opts:
            log.info("Override default option: {}={}".format(k, options[k]))
            opts[k] = options[k]
        else:
            raise KeyError("Option '{}' not found".format(k))

    # Bounds to np.array:
    # bounds[:, 0] - lower bounds
    # bounds[:, 1] - upper bounds
    bounds = np.array(bounds)

    # Initial population
    pop_size = opts["pop_size"]
    n_params = bounds.shape[0]

    xall = np.zeros((pop_size, n_params))
    for i in range(pop_size):
        xall[i, :] = random_x(bounds)

    yall = np.full(pop_size, np.nan)

    # Add x0 if present
    if x0 is not None:
        xall[0, :] = np.array(x0)
    log.debug(f"Initial population:\n{xall}")

    # Initial evaluation
    for i in range(pop_size):
        yall[i] = fun(xall[i], *args)

    # Optimize
    tries = opts["inertia"]
    count = 1  # 1, because initial pass finished
    nfev = pop_size  # pop_size, because initial pass finished
    exitmsg = ""
    ybest = yall.min()
    xbest = xall[np.argmin(yall), :]
    ybest_prev = np.array(ybest)
    xbest_prev = np.array(xbest)

    while tries > 0 and count < opts["generations"]:

        # Generate new population
        xall = np.zeros((pop_size, n_params))
        xall[0, :] = xbest_prev
        for i in range(1, pop_size):
            xall[i, :] = random_x(bounds)

        # Evaluate
        for i in range(pop_size):
            yall[i] = fun(xall[i], *args)
            nfev += 1

        # Find current best
        ybest = yall.min()
        xbest = xall[np.argmin(yall), :]

        # Solution improved?
        if ybest_prev - ybest > opts["tol"]:
            # Yes
            ybest_prev = ybest
            xbest_prev = xbest
            log.debug(f"Improvement = {ybest_prev - ybest:.5f}")
        else:
            # No
            tries -= 1

        # Callback
        if callback is not None:
            callback(xbest, ybest, count, *args)

        # Exit message
        if tries == 0:
            exitmsg = f"Max. number of tries ({opts['inertia']}) reached"
        if count == opts["generations"]:
            exitmsg = f"Max. number of iterations ({opts['generations']}) reached"

        log.info(f"Iter.{count}: y = {ybest:.5f}")
        log.debug(f"x =\n{xbest}")

        count += 1

    # Optimization result
    res = OptRes(x=xbest, message=exitmsg, ng=count, nfev=nfev, fx=ybest)

    log.info(res)

    return res


if __name__ == "__main__":
    # Example
    logging.basicConfig(level="INFO")

    from modestga.benchmark.functions import rastrigin

    N = 50
    bounds = [(-5.12, 5.12) for i in range(N)]
    res = minimize(rastrigin, bounds, x0=None)
    print(res)
