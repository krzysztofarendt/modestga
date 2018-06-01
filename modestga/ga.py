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
    :param options: dict, GA options
    :return: OptRes, optimization result
    """
    np.set_printoptions(precision=3)

    log = logging.getLogger(name='minimize')
    log.info('Start minimization')

    opts = {
        'generations': 100,     # Max. number of generations
        'pop_size': 100,        # Population size
        'mut_rate': 0.05,       # Mutation rate
        'mut_dist': None,       # Mutation distance
        'trm_size': 10,         # Tournament size
        'tol': 1e-6,            # Solution tolerance
        'inertia': 10,          # Max. number of non-improving generations (TODO)
        'xover_ratio': 0.5      # Crossover ratio
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
        log.debug('Using initial guess x0={}'.format(x0))
        pop.ind[0] = individual.Individual(
            genes=norm(x0, bounds),
            bounds=bounds,
            fun=fun,
            args=args
        )
        log.info('Individual based on x0:\n{}'.format(pop.ind[0]))

    log.info('Initial population:\n{}'.format(pop))

    # Loop over generations
    ng = 0
    nstalled = 0
    vprev = None
    exitmsg = None

    for gi in range(opts['generations']):

        ng += 1

        # Initialize children
        children = list()
        log.debug('Initiaze new children')

        # Elitism
        children.append(pop.get_fittest())
        log.debug('Elitism, add {}'.format(children[0]))

        # Fill other slots with children
        while len(children) < len(pop.ind):
            #Cross-over
            i1, i2 = operators.tournament(pop, opts['trm_size'])
            child = operators.crossover(i1, i2, opts['xover_ratio'])

            # Mutation
            child = operators.mutation(
                child, opts['mut_rate'], opts['mut_dist']
            )

            children.append(child)
        
        # Update population with new individuals
        pop.ind = children

        log.info('Generation {}:\n{}'.format(gi, pop))

        # Tolerance check
        if vprev is None:
            vprev = pop.get_fittest().val
        elif abs(vprev - pop.get_fittest().val < opts['tol']):
            vprev = pop.get_fittest().val
            nstalled += 1
        else:
            vprev = pop.get_fittest().val
            nstalled = 0

        # Break if stalled
        if nstalled >= opts['inertia']:
            exitmsg = \
                "Solution improvement below tolerance for {} generations" \
                .format(nstalled)
            break

    if ng == opts['generations']:
        exitmsg = "Maximum number of generations ({}) reached" \
            .format(opts['generations'])

    # Optimization result
    res = OptRes(
        x = pop.get_fittest().get_estimates(),
        ng = ng,
        message = exitmsg,
        fun = pop.get_fittest().val
    )

    return res


class OptRes:
    """
    Optimization result.

    Instance attributes:
    - x - numpy 1D array, optimized parameters
    - message - str, exit message
    - ng - int, number of generations
    - fun - float, final function value
    """
    def __init__(self, x, message, ng, fun):
        self.x = x
        self.message = message
        self.ng = ng
        self.fun = fun

    def __str__(self):
        s = "Optimization result:\n"
        s += "====================\n"
        s += "x = {}\n".format(self.x)
        s += "message = {}\n".format(self.message)
        s += "ng = {}\n".format(self.ng)
        s += "fun = {}\n".format(self.fun)
        return s


if __name__ == "__main__":

    logging.basicConfig(filename='ga.log', level='DEBUG', filemode='w')

    def f(x):
        return np.sum(x ** 2)

    bounds = ((0, 10), (0, 10), (0, 10), (0, 10), (0, 10))
    options = {'tol': 1e-12, 'inertia': 25}

    res = minimize(f, bounds, options=options)

    print(res)
