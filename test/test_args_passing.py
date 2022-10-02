import numpy as np

from modestga import ga
from modestga import minimize


def cost_fun(x, *args):
    return np.sum(np.array(x) ** 2)


def test_args_passing():
    x0 = tuple(np.random.random(5))
    bounds = tuple([(-1, 5) for i in x0])
    args = ["arg0_ok", "arg1_ok"]
    options = {"generations": 3}

    def fun_args_wrapper(x, *args):
        arg0 = args[0]
        arg1 = args[1]
        assert arg0 == "arg0_ok"
        assert arg1 == "arg1_ok"
        return cost_fun(x)

    res = minimize(
        fun_args_wrapper, bounds, x0=x0, args=args, options=options, workers=1
    )


def test_args_passing_2workers():
    x0 = tuple(np.random.random(5))
    bounds = tuple([(-1, 5) for i in x0])
    args = ["arg0_ok", "arg1_ok"]
    options = {"generations": 3}

    def fun_args_wrapper(x, *args):
        arg0 = args[0]
        arg1 = args[1]
        assert arg0 == "arg0_ok"
        assert arg1 == "arg1_ok"
        return cost_fun(x)

    res = minimize(
        fun_args_wrapper, bounds, x0=x0, args=args, options=options, workers=2
    )
