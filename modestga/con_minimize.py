import logging

import numpy as np

import modestga


def con_minimize(
    fun, bounds, constr=(), x0=None, args=(), callback=None, options={}, workers=None
):
    """Constrained minimization of `fun` using Genetic Algorithm.

    This function is a wrapper over modestga.minimize().
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
        fcore = augmented_args[0]  # Function to be minimized
        fcons = augmented_args[1]  # Constraints
        user_args = augmented_args[2:]  # Arguments

        # Evaluate core function
        ycore = fcore(x, *user_args)

        # Initialize penalty
        penalty = 0.0

        # Update penalty
        # (the more negative fcon() is, the higher penalty)
        for f in fcons:
            ycon = np.max([f(x, *user_args) * -1.0, 0.0])
            pscale = ycore / (ycon + 1e-6)
            penalty += ycon * pscale

        return ycore + penalty

    # Run minimization
    augmented_args = (fun, constr, *args)

    res = modestga.minimize(
        fun=fun_soft_con,
        bounds=bounds,
        x0=x0,
        args=augmented_args,
        callback=callback,
        options=options,
        workers=workers,
    )

    # Extend result with contraint violation info
    res.constr = [fcon(res.x, *args) for fcon in constr]

    return res
