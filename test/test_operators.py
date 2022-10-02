import numpy as np

from modestga import operators
from modestga import population


def cost_fun(x, *args):
    return np.sum(np.array(x) ** 2)


def test_operators():
    # Crossover
    pop = population.Population(
        size=2, bounds=[(0, 5) for x in range(100)], fun=cost_fun
    )
    ind1 = pop.ind[0]
    ind2 = pop.ind[1]
    child = operators.crossover(ind1, ind2, uniform=0.5)
    assert (
        child.gen == ind1.gen
    ).sum() > 30, "Given uniformity=0.5, too few genes from ind1"
    assert (
        child.gen == ind2.gen
    ).sum() > 30, "Given uniformity=0.5, too few genes from ind2"

    # Mutation
    pop = population.Population(
        size=1, bounds=[(0, 5) for x in range(100)], fun=cost_fun
    )
    ind = pop.ind[0]
    mut1 = operators.mutation(ind, rate=1.0, scale=0.1)  # mutate all randomly
    mut2 = operators.mutation(ind, rate=0.0, scale=0.1)  # mutate none
    assert (mut1.gen != ind.gen).all()
    assert (mut2.gen == ind.gen).all()

    # Tournament
    popsize = 50
    pop = population.Population(
        size=popsize, bounds=[(0, 5) for x in range(5)], fun=cost_fun
    )

    try:
        t1, t2 = operators.tournament(pop, popsize)
    except AssertionError as e:
        # Tournament size has to be lower than population/2
        # This exception is fine
        print(f"\nAssertionError caught (it's OK): '{e}'")
        pass

    t1, t2 = operators.tournament(pop, 10)
    assert t1.id != t2.id, (
        "Small tournament size, so two different individuals "
        + "should be returned (at least with the current random seed)"
    )

    t1, t2 = operators.tournament(pop, 1)
    assert t1.id != t2.id, (
        "Small tournament size, so two different individuals "
        + "should be returned (at least with the current random seed)"
    )
