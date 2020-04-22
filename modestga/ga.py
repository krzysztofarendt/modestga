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


def minimize(fun, bounds, x0=None, args=(), callback=None, options={}):
    """
    Minimize `fun` using Genetic Algorithm.

    If `x0` is given, the initial population will contain one individual
    based on `x0`. Otherwise, all individuals will be random.

    `fun` arguments: `x`, `*args`.

    `callback` arguments: `x`, `fx`, `ng`, `*args`.
    `fx` is the function value at the generation `ng`.

    The default options are::

        options = {
            'generations': 1000,    # Max. number of generations
            'pop_size': 100,        # Population size
            'mut_rate': 0.01,       # Mutation rate
            'trm_size': 10,         # Tournament size
            'tol': 1e-3,            # Solution tolerance
            'inertia': 10,          # Max. number of non-improving generations
            'xover_ratio': 0.2      # Crossover ratio
        }

    Return an optimization result object with the following attributes:
    - x - numpy 1D array, optimized parameters,
    - message - str, exit message,
    - ng - int, number of generations,
    - fx - float, final function value.

    :param fun: function to be minimized
    :param bounds: tuple, parameter bounds
    :param x0: numpy 1D array, initial parameters
    :param args: tuple, positional arguments to be passed to `fun`
    :param callback: function, called after every generation
    :param options: dict, GA options
    :return: OptRes, optimization result
    """
    np.set_printoptions(precision=3)

    log = logging.getLogger(name='minimize(GA)')
    log.info('Start minimization')

    opts = {
        'generations': 1000,    # Max. number of generations
        'pop_size': 100,        # Population size
        'mut_rate': 0.01,       # Mutation rate
        'trm_size': 20,         # Tournament size
        'tol': 1e-3,            # Solution tolerance
        'inertia': 100,         # Max. number of non-improving generations
        'xover_ratio': 0.5      # Crossover ratio
    }

    for k in options:
        if k in opts:
            opts[k] = options[k]
        else:
            raise KeyError("Option '{}' not found".format(k))

    # Initialize population
    pop = population.Population(opts['pop_size'], bounds, fun)

    # Add user guess if present
    if x0 is not None:
        x0 = np.array(x0)
        log.debug('Using initial guess x0={}'.format(x0))
        pop.ind[0] = individual.Individual(
            genes=norm(x0, bounds),
            bounds=bounds,
            fun=fun,
            args=args
        )
        log.debug('Individual based on x0:\n{}'.format(pop.ind[0]))

    log.debug('Initial population:\n{}'.format(pop))

    # Loop over generations
    ng = 0
    nstalled = 0
    vprev = None
    exitmsg = None
    scale = 0.33
    mut_rate = opts['mut_rate']

    for gi in range(opts['generations']):

        ng += 1

        # Initialize children
        children = list()

        # Elitism
        children.append(pop.get_fittest())

        # Adaptive mutation parameters
        if nstalled > (opts['inertia'] // 3):
            scale *= 0.75                                   # Search closer to current x
            mut_rate /= 1 - 1 / len(bounds)                 # Mutate more often
            mut_rate = 0.5 if mut_rate > 0.5 else mut_rate  # But not more often than 50%
        # log.info(f"scale={scale}, mut_rate={mut_rate}")

        # Fill other slots with children
        while len(children) < len(pop.ind):
            #Cross-over
            i1, i2 = operators.tournament(pop, opts['trm_size'])
            child = operators.crossover(i1, i2, opts['xover_ratio'])

            # Mutation
            child = operators.mutation(child, mut_rate, scale)

            children.append(child)

        # Update population with new individuals
        pop.ind = children

        # Tolerance check
        fittest = pop.get_fittest()
        if vprev is None:
            vprev = fittest.val
        elif abs(vprev - fittest.val < opts['tol']):
            vprev = fittest.val
            nstalled += 1
        else:
            vprev = fittest.val
            nstalled = 0

        log.info(f'ng = {gi}, nfev = {fittest.nfev}, f(x) = {fittest.val}')
        # log.debug('Generation {}:\n{}'.format(gi, pop))

        # Callback
        if callback is not None:
            x = fittest.get_estimates()
            fx = fittest.val
            callback(x, fx, ng, *args)

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
    fittest = pop.get_fittest()
    res = OptRes(
        x = fittest.get_estimates(),
        message = exitmsg,
        ng = ng,
        nfev = fittest.nfev,
        fx = fittest.val
    )

    log.info(res)

    return res


class OptRes:
    """
    Optimization result.

    Instance attributes:
    - x - numpy 1D array, optimized parameters
    - message - str, exit message
    - nfev - int, number of function evaluations
    - ng - int, number of generations
    - fx - float, final function value
    """
    def __init__(self, x, message, ng, nfev, fx):
        self.x = x
        self.message = message
        self.ng = ng
        self.nfev = nfev
        self.fx = fx

    def __str__(self):
        s = "Optimization result:\n"
        s += "====================\n"
        s += "x = {}\n".format(self.x)
        s += "message = {}\n".format(self.message)
        s += "ng = {}\n".format(self.ng)
        s += "nfev = {}\n".format(self.nfev)
        s += "fx = {}\n".format(self.fx)
        return s


if __name__ == "__main__":
    # Example
    logging.basicConfig(level='DEBUG')

    from modestga.benchmark.functions import rastrigin

    N = 50
    bounds = [(-5.12, 5.12) for i in range(N)]
    res = minimize(rastrigin, bounds, x0=None)
    print(res)
