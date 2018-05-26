import logging
import random
import numpy as np


def crossover(ind1, ind2, uniform=0.5):
    child = ind1.copy()

    for i in range(ind2.gen.size):
        if random.random() > uniform:
            child.gen[i] = ind2.gen[i]

    return child


def tournament(pop, size):
    # Form groups
    group1 = np.random.choice(pop.ind, size=size, replace=False)
    group2 = np.random.choice(pop.ind, size=size, replace=False)

    # Pick and return fittest from each group
    fit1 = np.argmin([x.val for x in group1])
    fit2 = np.argmin([x.val for x in group2])

    return group1[fit1], group2[fit2]

def mutation(rate, dist=1):
    """
    :param rate: float (0-1), mutation rate
    :param dist: float (0-1), mutation distance
    """
    pass
