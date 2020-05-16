import logging
import time
import multiprocessing
import os
import random
import numpy as np
import cloudpickle
from modestga import individual
from modestga import operators
from modestga import population


# logging.basicConfig(
#     level='DEBUG',
#     filemode='w',
#     format="[%(processName)s][%(levelname)s] %(message)s"
# )


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


def minimize(fun, bounds, x0=None, args=(), callback=None, options={}, workers=os.cpu_count()-1):
    """Minimizes `fun` using Genetic Algorithm.

    If `x0` is given, the initial population will contain one individual
    based on `x0`. Otherwise, all individuals will be random.

    `fun` arguments: `x`, `*args`.

    `callback` arguments: `x`, `fx`, `ng`, `*args`.
    `fx` is the function value at the generation `ng`.

    The default options are::

        options = {
            'generations': 1000,    # Max. number of generations
            'pop_size': 100 * workers,  # Population size
            'mut_rate': 0.01,       # Mutation rate
            'trm_size': 20,         # Tournament size
            'tol': 1e-3,            # Solution tolerance
            'inertia': 100,         # Max. number of non-improving generations
            'xover_ratio': 0.5      # Crossover ratio
        }

    Returns an optimization result object with the following attributes:
    - x - numpy 1D array, optimized parameters,
    - message - str, exit message,
    - ng - int, number of generations,
    - fx - float, final function value.

    :param fun: function to be minimized
    :param bounds: tuple, parameter bounds
    :param x0: numpy 1D array, initial parameters
    :param args: tuple, positional arguments to be passed to `fun` and to `callback`
    :param callback: function, called after every generation
    :param options: dict, GA options
    :param workers: int, number of processes to use
    :return: OptRes, optimization result
    """
    log = logging.getLogger(name='minimize(GA)')
    log.info('Start minimization')

    np.set_printoptions(precision=3)

    # Function evaluation counter
    nfev = 0

    # Options
    opts = {
        'generations': 1000,    # Max. number of generations
        'pop_size': 100 * workers,  # Population size
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

    # Assertions
    assert opts['trm_size'] < opts['pop_size'] // workers, \
        "Tournament size has to be smaller than population divided " + \
        "by number of workers"

    # Multiprocessing
    if workers <= 1:
        # Single process
        parallel = False
        processes = None
        pipes = None
        end_event = None
        subpop_size = None

    else:
        # Parallel processing initialization
        logging.debug(f"Using multiprocessing, workers={workers}")
        from modestga.parallel.full import parallel_pop

        parallel = True
        processes = list()
        pipes = list()
        end_event = multiprocessing.Event()
        subpop_size = opts['pop_size'] // workers

        logging.debug(f"Subpopulation size = {subpop_size}")

        for i in range(workers):
            pipe = multiprocessing.Pipe(duplex=True)
            pipe_to = pipe[0]
            pipe_from = pipe[1]
            p = multiprocessing.Process(
                target=parallel_pop,
                name=f"Subpopulation-{i}",
                args=(
                    pipe_from,
                    cloudpickle.dumps(fun),
                    args,
                    bounds,
                    subpop_size,
                    opts['trm_size'],
                    opts['xover_ratio'],
                    opts['mut_rate'],
                    end_event
                )
            )
            p.start()
            processes.append(p)
            pipes.append(pipe_to)

    # Initialize population
    pop = population.Population(opts['pop_size'], bounds, fun, evaluate=True)
    nfev += len(pop.ind)

    # Add user guess if present
    if x0 is not None:
        x0 = np.array(x0)
        pop.ind[0] = individual.Individual(
            genes=norm(x0, bounds),
            bounds=bounds,
            fun=fun,
            args=args
        )
        nfev += 1

    # Loop over generations
    ng = 0
    nstalled = 0
    vprev = None
    exitmsg = None
    scale = 0.33
    mut_rate = opts['mut_rate']

    for gi in range(opts['generations']):
        ng += 1

        # Adaptive mutation parameters
        if nstalled > (opts['inertia'] // 3):
            scale *= 0.75                                   # Search closer to current x
            mut_rate /= 1 - 1 / len(bounds)                 # Mutate more often
            mut_rate = 0.5 if mut_rate > 0.5 else mut_rate  # But not more often than 50%

        # Fill other slots with children
        if not parallel:
            # Single process
    
            # Initialize children
            children = list()

            # Elitism
            children.append(pop.get_fittest())

            while len(children) < opts['pop_size']:
                #Cross-over
                i1, i2 = operators.tournament(pop, opts['trm_size'])
                child = operators.crossover(i1, i2, opts['xover_ratio'])

                # Mutation
                child = operators.mutation(child, mut_rate, scale)

                # Evaluate f(x)
                child.evaluate()
                nfev += 1

                # Add to children
                children.append(child)

            # Update population with new individuals
            pop.ind = children
        else:
            # Parallel processing
            data_to = list()
            data_from = list()

            # Divide genes among subpopulation
            all_genes = pop.get_genes()
            all_fx = pop.get_fx()
            subpop_genes = list()
            subpop_fx = list()
            for i in range(workers):
                subpop_genes.append(list())
                subpop_fx.append(list())
                for j in range(subpop_size):
                    subpop_genes[i].append(all_genes[i * subpop_size + j])
                    subpop_fx[i].append(all_fx[i * subpop_size + j])
            
            # Send data to workers
            for i in range(workers):
                data_to.append(dict())
                data_to[i]['scale'] = scale
                data_to[i]['genes'] = subpop_genes[i]
                data_to[i]['fx'] = subpop_fx[i]
                pipes[i].send(data_to[i])

            # Receive data from workers
            while len(data_from) < workers:
                for i in range(workers):
                    if pipes[i].poll(0.001):
                        data_from.append(pipes[i].recv())

            # Extract genes and function values
            new_genes = list()  # List (1 per subpop) of lists (1 per ind) of arrays
            new_fx = list()     # List (1 per subpop) of lists (1 per ind) of floats
            for d in data_from:
                new_genes.extend(d['genes'])
                new_fx.extend(d['fx'])
            nfev += len(new_fx)

            # Aggregate individuals of all subpopulations
            new_ind = list()
            for i in range(len(new_genes)):
                new_ind.append(
                    individual.Individual(
                        new_genes[i], bounds, fun, args=args, val=new_fx[i]
                    )
                )

            # Elitism (replace random individual)
            new_ind[random.randint(0, len(new_ind) - 1)] = pop.get_fittest()

            # Put individuals in a random order
            # (otherwise subpopulation don't exchange genes)
            random.shuffle(new_ind)

            # Update individuals
            pop.ind = new_ind

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

        log.info(f'ng = {gi}, nfev = {nfev}, f(x) = {fittest.val}')

        # User callback function
        if callback is not None:
            x = fittest.get_estimates()
            fx = fittest.val
            callback(x, fx, ng, *args)

        # Break if successful, i.e. f(x) = 0
        if fittest.val < opts['tol']:
            exitmsg = \
                "Solution found, f(x) < 0 + tol" \
                .format(nstalled)
            break

        # Break if stalled
        if nstalled >= opts['inertia']:
            exitmsg = \
                "Solution improvement below tolerance for {} generations" \
                .format(nstalled)
            break

    if ng == opts['generations']:
        exitmsg = "Maximum number of generations ({}) reached" \
            .format(opts['generations'])

    # Send message to subprocesses that the optimization is finished
    if parallel:
        end_event.set()
        for proc, pipe in zip(processes, pipes):
            log.debug(f"Closing {pipe} and joining {proc}")
            pipe.close()
            proc.join()

    # Optimization result
    fittest = pop.get_fittest()
    res = OptRes(
        x = fittest.get_estimates(),
        message = exitmsg,
        ng = ng,
        nfev = nfev,
        fx = fittest.val
    )

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
    from modestga.benchmark.functions import rastrigin
    fun = rastrigin
    bounds = [(-5.12, 5.12) for i in range(128)]
    options = {
        'generations': 10,
        'pop_size': 500,
        'tol': 1e-3
    }
    def callback(x, fx, ng, *args):
        """Callback function called after each generation"""
        print(f"\nCallback example:\nx=\n{x}\nf(x)={fx}\n")

    t0 = time.perf_counter()
    res = minimize(fun, bounds, callback=callback, options=options, workers=4)
    print(res)
    print(f"Time: {time.perf_counter() - t0}")
