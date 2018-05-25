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

logging.basicConfig(filename="test.log", level="DEBUG", filemode="w")

class TestEditor(unittest.TestCase):

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

    def test_population(self):
        pop = population.Population(
            size=20,
            bounds=[(0, 10), (0, 10), (0, 10)],
            fun=self.fun)

        self.assertEqual(len(pop.ind), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)