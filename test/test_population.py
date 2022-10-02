import numpy as np

from modestga import population


def cost_fun(x, *args):
    return np.sum(np.array(x) ** 2)


def test_population():
    pop = population.Population(
        size=20, bounds=[(0, 10), (0, 10), (0, 10)], fun=cost_fun
    )

    assert len(pop.ind) == 20

    # Get fittest
    fittest = pop.get_fittest()
    for i in pop.ind:
        assert fittest.val <= i.val

    fval = cost_fun(pop.get_fittest().get_estimates())
    assert fittest.val == fval
