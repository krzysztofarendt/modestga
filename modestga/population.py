import logging
import numpy as np
from modestga.individual import Individual


class Population():

    def __init__(self, size, bounds, fun, args=(), genes=None):
        self.log = logging.getLogger('Population')

        self.ind = list()

        if genes is None:
            genes = np.random.rand(len(bounds))

        for i in range(size):
            self.ind.append(Individual(
                genes=np.random.rand(len(bounds)),
                bounds=bounds,
                fun=fun,
                args=args)
            )

    def set_genes(self, genes, evaluate=True):
        for i, g in enumerate(genes):
            self.ind[i].gen = g
            if evaluate is True:
                self.ind[i].evaluate()

    def get_genes(self):
        genes = [x.gen for x in self.ind]
        return genes

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
