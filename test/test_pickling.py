import numpy as np

from modestga import minimize


def cost_fun(x, *args):
    return np.sum(np.array(x) ** 2)


def test_exotic_pickling():
    x0 = tuple(np.random.random(8))
    bounds = tuple([(-1, 1) for i in x0])
    options = {"generations": 3}

    class FunWrapper:
        def __init__(self, fun_to_apply):
            self.fun_to_apply = fun_to_apply

    args = [FunWrapper(np.fft.fft)]

    def fun_to_pickle(x, *args):
        """Some exotic function to be pickled."""
        fun_wrapper = args[0]
        x = fun_wrapper.fun_to_apply(x)
        return cost_fun(x)

    res = minimize(fun_to_pickle, bounds, x0=x0, args=args, options=options, workers=2)
