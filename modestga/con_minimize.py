import logging
import numpy as np
from modestga import minimize


def con_minimize(fun, bounds, constr=(), pscale=1e3, x0=None, args=(),
               callback=None, options={}, workers=None):
    """Constrained minimization of `fun` using Genetic Algorithm.

    This function is a wrapper over modetga.minimize().
    The constraints are defined as a tuple of functions
    (`fcon1(x, *args)`, `fcon2(x, *args)`, `...`).
    The algorithm searches for a solution minimizing
    `fun(x, *args)` and satisfying the conditions
    (`fcon1(x, *args) >= 0`, `fcon2(x, *args) >= 0`, `...`).

    `callback` arguments: `x`, `fx`, `ng`, `*args`.
    `fx` is the function value at the generation `ng`.

    Returns an optimization result object with the following attributes:
    - x - numpy 1D array, optimized parameters,
    - message - str, exit message,
    - ng - int, number of generations,
    - fx - float, final function value.

    :param fun: function to be minimized
    :param bounds: tuple, parameter bounds
    :param constr: tuple, functions defining constraints
    :param pscale: float, penalty scale (the higher, to more important constraints are)
    :param x0: numpy 1D array, initial parameters
    :param args: tuple, positional arguments to be passed to `fun` and to `fcon`
    :param callback: function, called after every generation
    :param options: dict, GA options
    :param workers: int, number of processes to use (will use all CPUs if None)
    :return: OptRes, optimization result
    """
    # Wrap cost function with constraints
    def fun_soft_con(x, *augmented_args):
        # Unpack constraints and arguments
        fcore = augmented_args[0]   # Function to be minimized
        fcons = augmented_args[1]   # Constraints
        user_args = augmented_args[2:]  # Arguments

        # Initialize penalty
        penalty = 0.

        # Update penalty
        # (the more negative fcon() is, the higher penalty)
        for f in fcons:
            penalty += np.max([f(x, *user_args) * -1 * pscale, 0.]) ** 2

        return fcore(x, *user_args) + penalty

    # Run minimization
    augmented_args = (fun, constr, *args)

    res = minimize(
        fun=fun_soft_con,
        bounds=bounds,
        x0=x0,
        args=augmented_args,
        callback=callback,
        options=options,
        workers=workers)

    # Extend result with contraint violation info
    res.constr = [fcon(res.x, *args) for fcon in constr]

    return res


if __name__ == "__main__":

    # Example of constrained minimization

    # Set up logging
    logging.basicConfig(
        level='DEBUG',
        filemode='w',
        format="[%(processName)s][%(levelname)s] %(message)s"
    )

    # Set up cost function
    from modestga.benchmark.functions import rastrigin
    fun = rastrigin

    # Set up bounds (rectangular)
    bounds = [(-5.12, 5.12) for i in range(8)]

    # Constraints (algorithm will try to keep their outputs positive)
    def fcon1(x, *args):
        return x[1] - 1.

    def fcon2(x, *args):
        return x[0] - 2.

    constr = (fcon1, fcon2)

    # Genetic Algorithm options
    options = {
        'generations': 1000,
        'pop_size': 500,
        'tol': 1e-3
    }

    # Run minimization
    res = con_minimize(fun, bounds, constr=constr, options=options)

    print(res)
