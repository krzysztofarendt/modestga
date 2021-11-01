import logging

import numpy as np


def bounds_tuples_to_array(t):
    """
    Convert a list of tuples with bounds into an array with the first
    dimension for lower bounds and the second for upper bounds.
    """
    a = np.array([[b[0] for b in t], [b[1] for b in t]])
    return a


def bounds_array_to_tuples(a):
    """
    Inversed conversion to `bounds_tuples_to_array()`.
    """
    t = tuple([(lo, hi) for lo, hi in zip(a[0], a[1])])
    return t


class Individual:

    # Total number of instances
    count = 0

    def __init__(self, genes, bounds, fun, args=(), val=None):
        """
        :param genes: 1D array, floats between 0 and 1 (inclusive)
        :param bounds: tuple of tuples, parameter bounds (min, max)
        :param fun: function to be minimized
        :param args: arguments to pass to fun
        :param val: Function value or None
        """
        # Set name
        self.id = "Ind#{}".format(Individual.count)
        self.log = logging.getLogger(name=self.id)
        Individual.count += 1

        # Copy arguments
        self.gen = np.array(genes)
        self.fun = fun
        self.args = args
        self.bnd = bounds_tuples_to_array(bounds)

        # Evaluate and save score
        if val is None:
            self.val = self.evaluate()
        else:
            self.val = val

    def set_genes(self, g):
        self.gen = g

    def get_estimates(self):
        return self.bnd[0] + self.gen * (self.bnd[1] - self.bnd[0])

    def evaluate(self):
        """
        Evaluate cost function.

        Instead of calling this method,
        you may read the instance attribute `val`.
        """
        try:
            self.val = self.fun(self.get_estimates(), *self.args)
        except Exception as e:
            print(e)
            self.log.warning(str(e))
            self.val = 1e8  # Very large value
        return self.val

    def copy(self):
        ind = Individual(
            self.gen, bounds_array_to_tuples(self.bnd), self.fun, self.args, self.val
        )
        return ind

    def __str__(self):
        s = "{}: {} -> {}".format(self.id, self.gen, self.val)
        return s
