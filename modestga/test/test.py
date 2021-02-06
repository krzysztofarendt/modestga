import logging
import unittest
import os
import shutil
import tempfile
import random
import numpy as np
from modestga import individual
from modestga import population
from modestga import operators
from modestga import ga


class TestModestga(unittest.TestCase):

    def setUp(self):
        # Random seed
        random.seed(123)
        np.random.seed(123)
        # Cost function
        def f(x):
            return np.sum(np.array(x) ** 2)
        self.fun = f

    def tearDown(self):
        pass

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

        fval = self.fun(pop.get_fittest().get_estimates())
        self.assertTrue(fittest.val == fval)

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
            size=1, bounds=[(0, 5) for x in range(100)], fun=self.fun
        )
        ind = pop.ind[0]
        mut1 = operators.mutation(ind, rate=1.0, scale=0.1) # mutate all randomly
        mut2 = operators.mutation(ind, rate=0.0, scale=0.1) # mutate none
        self.assertTrue((mut1.gen != ind.gen).all())
        self.assertTrue((mut2.gen == ind.gen).all())

        # Tournament
        popsize = 50
        pop = population.Population(
            size=popsize, bounds=[(0, 5) for x in range(5)], fun=self.fun
        )

        try:
            t1, t2 = operators.tournament(pop, popsize)
        except AssertionError as e:
            # Tournament size has to be lower than population/2
            # This exception is fine
            print(f"\nAssertionError caught (it's OK): '{e}'")
            pass

        t1, t2 = operators.tournament(pop, 10)
        self.assertNotEqual(t1.id, t2.id,
            "Small tournament size, so two different individuals "
            "should be returned (at least with the current random seed)")

        t1, t2 = operators.tournament(pop, 1)
        self.assertNotEqual(t1.id, t2.id,
            "Small tournament size, so two different individuals "
            "should be returned (at least with the current random seed)")

    def test_ga(self):
        # Test norm()
        x = [0, 2.5, 7.5, 10]
        bounds = tuple([(0, 10) for i in x])
        n = ga.norm(x, bounds)
        self.assertTrue((np.abs(n - np.array([0, 0.25, 0.75, 1.0])) < 1e-12).all())

        # Test x0 and elitism
        options = {'generations': 10, 'mut_rate': 0.25}
        opt1 = ga.minimize(self.fun, bounds, options=options)
        self.assertEqual(opt1.fx, self.fun(opt1.x))

        options = {'generations': 50, 'mut_rate': 0.01}
        x0 = opt1.x
        opt2 = ga.minimize(self.fun, bounds, x0=x0, options=options)
        self.assertEqual(opt2.fx, self.fun(opt2.x))
        self.assertLessEqual(opt2.fx, opt1.fx)

        # Test small population size
        options = {'generations': 2, 'pop_size': 4, 'trm_size': 1}
        opt3 = ga.minimize(self.fun,
                           bounds,
                           x0=x0,
                           options=options,
                           workers=1)

        # Test pop_size and trm_size re-adjusting
        options = {'generations': 2, 'pop_size': 8}
        opt4 = ga.minimize(self.fun,
                           bounds,
                           x0=x0,
                           options=options,
                           workers=2)

        # Test convergence
        options['generations'] = 100
        options['mut_rate'] = 0.01
        opt = ga.minimize(self.fun, bounds)
        self.assertLess(opt.fx, 0.1)

        # Test callback
        options = {'generations': 5, 'pop_size': 8, 'trm_size': 1}
        def cb(x, fx, ng):
            print("Generation #{}".format(ng))
            print("x = {}".format(x))
            print("f(x) = {}".format(fx))
            global fx_last
            global x_last
            fx_last = fx
            x_last = x
        opt = ga.minimize(self.fun, bounds, callback=cb, options=options, workers=2)
        self.assertEqual(fx_last, opt.fx)
        self.assertTrue((np.abs(x_last - opt.x) < 1e-10).all())

    def test_ga_1param(self):
        x0 = [5]
        bounds = tuple([(0, 10) for i in x0])
        options = {'generations': 20, 'pop_size': 4, 'trm_size': 1}
        res = ga.minimize(self.fun,
                          bounds,
                          x0=x0,
                          options=options,
                          workers=1)

    def test_args_passing(self):
        x0 = tuple(np.random.random(5))
        bounds = tuple([(-1, 5) for i in x0])
        args = ['arg0_ok', 'arg1_ok']
        options = {'generations': 3}

        def fun_args_wrapper(x, *args):
            arg0 = args[0]
            arg1 = args[1]
            self.assertEqual(arg0, 'arg0_ok')
            self.assertEqual(arg1, 'arg1_ok')
            return self.fun(x)

        res = ga.minimize(fun_args_wrapper,
                          bounds,
                          x0=x0,
                          args=args,
                          options=options,
                          workers=1)

    def test_args_passing_2workers(self):
        x0 = tuple(np.random.random(5))
        bounds = tuple([(-1, 5) for i in x0])
        args = ['arg0_ok', 'arg1_ok']
        options = {'generations': 3}

        def fun_args_wrapper(x, *args):
            arg0 = args[0]
            arg1 = args[1]
            self.assertEqual(arg0, 'arg0_ok')
            self.assertEqual(arg1, 'arg1_ok')
            return self.fun(x)

        res = ga.minimize(fun_args_wrapper,
                          bounds,
                          x0=x0,
                          args=args,
                          options=options,
                          workers=2)

    def test_exotic_pickling(self):
        x0 = tuple(np.random.random(8))
        bounds = tuple([(-1, 1) for i in x0])
        options = {'generations': 3}

        class FunWrapper:
            def __init__(self, fun_to_apply):
                self.fun_to_apply = fun_to_apply

        args = [FunWrapper(np.fft.fft)]

        def fun_to_pickle(x, *args):
            """Some exotic function to be pickled."""
            fun_wrapper = args[0]
            x = fun_wrapper.fun_to_apply(x)
            return self.fun(x)

        res = ga.minimize(fun_to_pickle,
                          bounds,
                          x0=x0,
                          args=args,
                          options=options,
                          workers=2)


if __name__ == "__main__":
    logging.basicConfig(filename="test.log", level="DEBUG", filemode="w")
    unittest.main(verbosity=2)
