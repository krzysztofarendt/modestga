import logging
import unittest
import os
import shutil
import tempfile
import random
import numpy as np
from modestga import metrics
from modestga import individual
from modestga import population
from modestga import operators
from modestga import ga

logging.basicConfig(filename="test.log", level="DEBUG", filemode="w")

class TestModestga(unittest.TestCase):

    def setUp(self):
        # Random seed
        random.seed(123)
        np.random.seed(123)
        # Temporary directory
        self.tempdir = tempfile.mkdtemp()
        print("Temp dir created: {}".format(self.tempdir))
        # Cost function
        def f(x):
            return np.sum(x ** 2)
        self.fun = f

    def tearDown(self):
        shutil.rmtree(self.tempdir)
        assert not os.path.exists(self.tempdir)
        print("Temp dir removed: {}".format(self.tempdir))

    def test_metrics(self):
        y1 = np.arange(10)
        y2 = y1 + np.random.rand(y1.size)
        e = metrics.rmse(y1, y2)  # True: 0.586
        self.assertLess(0.586 - e, 1e-3)

    def test_individual(self):
        ind = individual.Individual(
            genes=[0., 0.25, 0.5, 0.75, 1.],
            bounds=[(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)],
            fun=self.fun)

        # Test calculation of estimates from genes
        est = ind.get_estimates()
        self.assertTrue((np.abs(np.array([0, 2.5, 5., 7.5, 10.]) - est)).sum() < 1e-8)

        # Test copy
        ind_copy = ind.copy()
        self.assertTrue((ind.gen == ind_copy.gen).all(), "Genes equality failed")
        self.assertTrue((ind.bnd == ind_copy.bnd).all(), "Bounds equality failed")

        # Test individual count
        self.assertEqual(ind.count, 2)

        # Test evaluate function
        self.assertLess(np.abs(ind.evaluate() - 187.5), 1e-8)

    def test_population(self):
        pop = population.Population(
            size=20, bounds=[(0, 10), (0, 10), (0, 10)], fun=self.fun)

        self.assertEqual(len(pop.ind), 20)

        # Get fittest
        fittest = pop.get_fittest()
        for i in pop.ind:
            self.assertTrue(fittest.val <= i.val)


    def test_operators(self):
        # Crossover
        pop = population.Population(
            size=2, bounds=[(0, 5) for x in range(100)], fun=self.fun
        )
        ind1 = pop.ind[0]
        ind2 = pop.ind[1]
        child = operators.crossover(ind1, ind2, uniform=0.5)
        self.assertTrue((child.gen == ind1.gen).sum() > 30,
            "Given uniformity=0.5, too few genes from ind1")
        self.assertTrue((child.gen == ind2.gen).sum() > 30,
            "Given uniformity=0.5, too few genes from ind2")

        # Mutation
        pop = population.Population(
            size=1, bounds=[(0, 5) for x in range(10)], fun=self.fun
        )
        ind = pop.ind[0]
        mut1 = operators.mutation(ind, rate=0.5, dist=None) # mutate some randomly
        mut2 = operators.mutation(ind, rate=0.5, dist=0.01) # mutate some a bit
        mut3 = operators.mutation(ind, rate=1.0, dist=None) # mutate all randomly
        mut4 = operators.mutation(ind, rate=0.0, dist=None) # mutate none
        self.assertTrue((mut1.gen == ind.gen).sum() > 0)
        self.assertTrue((np.abs(mut2.gen - ind.gen) <= 0.01).all())
        self.assertTrue((mut3.gen != ind.gen).all())
        self.assertFalse((np.abs(mut3.gen - ind.gen) <= 0.01).all())
        self.assertTrue((mut4.gen == ind.gen).all())

        # Tournament
        popsize = 50
        pop = population.Population(
            size=popsize, bounds=[(0, 5) for x in range(5)], fun=self.fun
        )

        t1, t2 = operators.tournament(pop, popsize)
        self.assertEqual(t1.id, t2.id,
            "Tournament size equal to population size, "
            "yet two different individuals selected")

        t1, t2 = operators.tournament(pop, 10)
        self.assertNotEqual(t1.id, t2.id,
            "Small tournament size, so two different individuals "
            "should be returned (at least with the current random seed)")

    def test_ga(self):
        # Test norm(), denorm()
        x = [0, 2.5, 7.5, 10]
        bounds = tuple([(0, 10) for i in x])
        n = ga.norm(x, bounds)
        d = ga.denorm(n, bounds)
        self.assertTrue(((d - x) < 1e-12).all())
        self.assertTrue(((n - np.array([0, 0.25, 0.75, 1.0])) < 1e-12).all())

        # 

if __name__ == "__main__":
    unittest.main(verbosity=2)
