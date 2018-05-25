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

        self.log.debug('Instantiated:\n{}'.format(self))
    
    def __str__(self):
        s = ''
        for i in self.ind:
            s += '{}\n'.format(i)
        return s
