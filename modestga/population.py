import logging
import numpy as np
from modestga.individual import Individual


class Population():

    def __init__(self, size, bounds, fun):
        self.log = logging.getLogger('Population')

        self.ind = list()

        for i in range(size):
            self.ind.append(Individual(
                genes=np.random.rand(len(bounds)),
                bounds=bounds,
                fun=fun)
            )

    def get_fittest(self):
        fittest = None
        min_fun = None

        for i in self.ind:
            if min_fun is None:
                min_fun = i.val
                fittest = i
            elif i.val < min_fun:
                min_fun = i.val
                fittest = i

        return fittest

    def __str__(self):
        s = ''
        for i in self.ind:
            s += '{}\n'.format(i)
        return s
