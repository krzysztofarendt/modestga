import numpy as np

from modestga import individual


def cost_fun(x, *args):
    return np.sum(np.array(x) ** 2)


def test_individual():
    ind = individual.Individual(
        genes=[0.0, 0.25, 0.5, 0.75, 1.0],
        bounds=[(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)],
        fun=cost_fun,
    )

    # Test calculation of estimates from genes
    est = ind.get_estimates()
    assert (np.abs(np.array([0, 2.5, 5.0, 7.5, 10.0]) - est)).sum() < 1e-8

    # Test copy
    ind_copy = ind.copy()
    assert (ind.gen == ind_copy.gen).all(), "Genes equality failed"
    assert (ind.bnd == ind_copy.bnd).all(), "Bounds equality failed"

    # Test evaluate function
    assert np.abs(ind.evaluate() - 187.5) < 1e-8
