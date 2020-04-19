import logging
import numpy as np


def bounds_tuples_to_array(t):
    """
    Convert a list of tuples with bounds into an array with the first
    dimension for lower bounds and the second for upper bounds.
    """
    a = np.array([
        [b[0] for b in t],
        [b[1] for b in t]
    ])
    return a


def bounds_array_to_tuples(a):
    """
    Inversed conversion to `bounds_tuples_to_array()`.
    """
    t = tuple([(lo, hi) for lo, hi in zip(a[0], a[1])])
    return t


class Individual():

    # Total number of instances
    count = 0

    # Total number of function evaluations
    nfev = 0

    def __init__(self, genes, bounds, fun, args=()):
        """
        :param genes: 1D array, floats between 0 and 1 (inclusive)
        :param bounds: tuple of tuples, parameter bounds (min, max)
        :param fun: function to be minimized
        """
        # Set name
        self.id = 'Ind#{}'.format(Individual.count)
        self.log = logging.getLogger(name=self.id)
        Individual.count += 1

        # Copy arguments
        self.gen = np.array(genes)
        self.fun = fun
        self.args = args
        self.bnd = bounds_tuples_to_array(bounds)

        # Evaluate and save score
        self.val = self.evaluate()

        # self.log.debug("Instantiated with genes {}".format(self.gen))

    def set_genes(self, g):
        self.gen = g
        self.val = self.evaluate()

    def get_estimates(self):
        return self.bnd[0] + self.gen * (self.bnd[1] - self.bnd[0])

    def evaluate(self):
        """
        Evaluate cost function.

        Instead of calling this method,
        you may read the instance attribute `val`.
        """
        Individual.nfev += 1
        return self.fun(self.get_estimates(), *self.args)

    def copy(self):
        ind = Individual(
            self.gen,
            bounds_array_to_tuples(self.bnd),
            self.fun
        )
        return ind

    def __str__(self):
        s = '{}: {} -> {}'.format(self.id, self.gen, self.val)
        return s
