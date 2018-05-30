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


def mutation(ind, rate, dist=None):
    """
    Mutate genes within the distance `dist`. `rate` controls
    how many genes are mutated. Mutated genes cannot exceed
    the bounds (0-1). `dist` is adviced to be either very small,
    or `None`. The equation used to calculated a new gene is:

    if dist is not None:
        new = old + dist * random
    else:
        new = random

    In the first case, `new` is trimmed to stay within 0 and 1.
    In result, the boundaries are "sticky" (more probable to 
    reach the closer boundary than stay in the middle).

    :param ind: Individual
    :param rate: float (0-1), mutation rate
    :param dist: float (0-1) or None, mutation distance
    :return: mutated Individual (copy)
    """
    mut = np.random.rand(ind.gen.size)

    if dist is not None:
        mut = np.where(
            mut < rate,
            np.maximum(
                np.minimum(
                    ind.gen + dist * random.random(),
                    1.
                ),
                0.
            ),
            ind.gen
        )
    else:
        mut = np.where(mut < rate, random.random(), ind.gen)

    mutind = ind.copy()
    mutind.gen = mut
    return mutind
