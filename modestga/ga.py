import logging
import numpy as np
from modestga import individual
from modestga import operators
from modestga import population


def norm(x, bounds):
    """
    Normalize `x` with respect to `bounds`.

    :param x: 1D array
    :param bounds: tuple of tuples, lower and upper bounds
    :return: 1D array, values in the range [0, 1]
    """
    n = np.array(
        [(v - b[0]) / (b[1]- b[0]) for v, b in zip(x, bounds)]
    )
    if (n < 0).any() or (n > 1).any():
        msg = "x outside bounds:\n"
        msg += "x={}\n".format(x)
        msg += "bounds={}".format(bounds)
        logging.error(msg)
        raise ValueError(msg)
    return n


def denorm(x, bounds):
    """
    Denormalize `x` with respect to `bounds`.

    :param x: 1D array, values in the range [0, 1] (inclusive)
    :param bounds: tuple of tuples, lower and upper bounds
    :return: 1D array
    """
    d = np.array(
        [v * (b[1] - b[0]) + b[0] for v, b in zip(x, bounds)]
    )
    return d


def minimize(fun, bounds, x0=None, args=(), callback=None, options={}):
    """
    Minimize `fun` using Genetic Algorithm.

    `fun` must be a function of `x`, possibly followed by positional arguments.

    :param fun: function to be minimized
    :param bounds: tuple, parameter bounds
    :param x0: numpy 1D array, initial parameters
    :param args: tuple, positional arguments to be passed to `fun`
    :param callback: function, called after every generation (TODO)
    :param options: dict, GA options (TODO)
    :return: (TODO)
    """
    log = logging.getLogger(name='minimize')
    log.info('Start minimization')
    opts = {
        'generations': 100,
        'pop_size': 100,
        'mut_rate': 0.05,
        'mut_dist': None,
        'trm_size': 10,
        'tol': 1e-6
    }

    for k in options:
        if k in opts:
            log.info('Override default option: {}={}'.format(k, options[k]))
            opts[k] = options[k]
        else:
            raise KeyError("Option '{}' not found".format(k))

    log.info('Final options: {}'.format(opts))

    # Initialize population
    pop = population.Population(opts['pop_size'], bounds, fun)

    # Add user guess if present
    if x0 is not None:
        pop.ind[0] = individual.Individual(
            genes=norm(x0, bounds),
            bounds=bounds,
            fun=fun,
            args=args
        )

    # Loop over generations
    for gi in range(opts['generations']):
        pass

    return None


if __name__ == "__main__":

    logging.basicConfig(filename='ga.log', level='DEBUG', filemode='w')

    def f(x):
        return np.sum(x ** 2)

    bounds = ((0, 10), (0, 10), (0, 10))
    options = {'generations': 10, 'pop_size': 5, 'tol': 1e-6}

    opt = minimize(f, bounds, options=options)

    print(opt)
