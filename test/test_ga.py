import numpy as np

from modestga import ga
from modestga import minimize


def cost_fun(x, *args):
    return np.sum(np.array(x) ** 2)


def test_ga():
    # Test norm()
    x = [0, 2.5, 7.5, 10]
    bounds = tuple([(0, 10) for i in x])
    n = ga.norm(x, bounds)
    assert (np.abs(n - np.array([0, 0.25, 0.75, 1.0])) < 1e-12).all()

    # Test x0 and elitism
    options = {"generations": 3, "mut_rate": 0.25}
    opt1 = minimize(cost_fun, bounds, options=options)
    assert opt1.fx == cost_fun(opt1.x)

    options = {"generations": 10, "mut_rate": 0.01}
    x0 = opt1.x
    opt2 = minimize(cost_fun, bounds, x0=x0, options=options)
    assert opt2.fx == cost_fun(opt2.x)
    assert opt2.fx <= opt1.fx

    # Test small population size
    options = {"generations": 3, "pop_size": 4, "trm_size": 1}
    opt3 = minimize(cost_fun, bounds, x0=x0, options=options, workers=1)

    # Test pop_size and trm_size re-adjusting
    options = {"generations": 3, "pop_size": 8}
    opt4 = minimize(cost_fun, bounds, x0=x0, options=options, workers=2)

    # Test convergence
    options["generations"] = 100
    options["mut_rate"] = 0.01
    opt = minimize(cost_fun, bounds)
    assert opt.fx < 0.1

    # Test callback
    options = {"generations": 3, "pop_size": 8, "trm_size": 1}

    def cb(x, fx, ng):
        print("Generation #{}".format(ng))
        print("x = {}".format(x))
        print("f(x) = {}".format(fx))
        global fx_last
        global x_last
        fx_last = fx
        x_last = x

    # Single process
    opt = minimize(cost_fun, bounds, callback=cb, options=options, workers=1)
    assert fx_last == opt.fx
    assert (np.abs(x_last - opt.x) < 1e-10).all()

    # Two subprocesses
    opt = minimize(cost_fun, bounds, callback=cb, options=options, workers=2)
    assert fx_last == opt.fx
    assert (np.abs(x_last - opt.x) < 1e-10).all()


def test_ga_1param():
    x0 = [5]
    bounds = tuple([(0, 10) for i in x0])
    options = {"generations": 3, "pop_size": 4, "trm_size": 1}
    res = minimize(cost_fun, bounds, x0=x0, options=options, workers=1)

    # pop_size has to be larger for workers=2
    options = {"generations": 3, "pop_size": 8, "trm_size": 1}
    res = minimize(cost_fun, bounds, x0=x0, options=options, workers=2)
