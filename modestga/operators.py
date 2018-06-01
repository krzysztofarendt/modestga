import logging
import random
import numpy as np


def crossover(ind1, ind2, uniform=0.5):
    log = logging.getLogger('crossover')
    log.debug("{} x {}".format(ind1.id, ind2.id))
    child = ind1.copy()

    for i in range(ind2.gen.size):
        if random.random() > uniform:
            child.gen[i] = ind2.gen[i]

    return child


def tournament(pop, size):
    log = logging.getLogger('tournament')

    # Form groups
    group1 = np.random.choice(pop.ind, size=size, replace=False)
    group2 = np.random.choice(pop.ind, size=size, replace=False)

    # Pick and return fittest from each group
    fit1 = np.argmin([x.val for x in group1])
    fit2 = np.argmin([x.val for x in group2])

    i1 = group1[fit1]
    i2 = group2[fit2]

    log.debug("{}, {}".format(i1.id, i2.id))

    return i1, i2


def mutation(ind, rate):
    """
    Mutate genes. `rate` controls how many genes are mutated.

    :param ind: Individual
    :param rate: float (0-1), mutation rate
    :return: mutated Individual (copy)
    """
    log = logging.getLogger('mutation')
    log.debug('{}'.format(ind.id))

    mut = np.random.rand(ind.gen.size)
    mut = np.where(mut < rate, random.random(), ind.gen)

    mutind = ind.copy()
    mutind.set_genes(mut)
    return mutind
