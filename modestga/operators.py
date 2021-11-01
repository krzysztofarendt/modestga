import logging
import random

import numpy as np


def crossover(ind1, ind2, uniform=0.5):
    # log = logging.getLogger('crossover')
    child = ind1.copy()

    rand = np.random.random(size=child.gen.size)
    child.gen = np.where(rand > uniform, child.gen, ind2.gen)

    return child


def tournament(pop, size):
    # log = logging.getLogger('tournament')

    assert size * 2 < len(
        pop.ind
    ), f"Tournament size ({size}) has to be lower than population ({len(pop.ind)}) // 2"

    # Form groups
    g1g2 = np.random.choice(pop.ind, size=size * 2, replace=False)
    group1 = g1g2[:size]
    group2 = g1g2[size:]

    # Pick and return fittest from each group
    fit1 = np.argmin([x.val for x in group1])
    fit2 = np.argmin([x.val for x in group2])

    i1 = group1[fit1]
    i2 = group2[fit2]

    # log.debug("{}, {}".format(i1.id, i2.id))

    return i1, i2


def mutation(ind, rate, scale):
    """
    Mutate genes. `rate` controls how many genes are mutated.

    :param ind: Individual
    :param rate: float (0-1), mutation rate
    :param scale: standard deviation of the normal distribution
    :return: mutated Individual (copy)
    """
    # log = logging.getLogger('mutation')

    # Draw random value to be compared with rate
    mut = np.random.rand(ind.gen.size)

    # Draw new random genes from a normal distribution
    mut_gen = np.random.normal(loc=ind.gen, scale=scale)
    # To 0-1 range
    mut_gen = np.where(mut_gen > 1.0, 1.0, mut_gen)
    mut_gen = np.where(mut_gen < 0.0, 0.0, mut_gen)
    # Substitute genes where mut < rate
    new_gen = np.where(mut < rate, mut_gen, ind.gen)

    # log.debug(f"Old genes: {ind.gen}")
    # log.debug(f"Candidate genes {mut_gen}")
    # log.debug(f"New genes {new_gen}")

    mutind = ind.copy()
    mutind.set_genes(new_gen)

    return mutind
