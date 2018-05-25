import logging
import numpy as np
from modestga.individual import Individual


class Population():

    def __init__(self, size, bounds, fun):

        self.ind = list()

        for i in range(size):
            self.ind.append(Individual(
                genes=np.random.rand(len(bounds)),
                bounds=bounds,
                fun=fun
            ))
