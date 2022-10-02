import logging

import numpy as np

from modestga import minimize


def cost_fun(x, *args):
    return np.sum(np.array(x) ** 2)


def test_exception_handling():
    x0 = [5]
    bounds = tuple([(0, 10) for i in x0])
    options = {"generations": 2, "pop_size": 4, "trm_size": 1}

    def fun_exception(x, *args):
        raise Exception("Dummy Exception")
        return cost_fun(x, *args)

    res = minimize(fun_exception, bounds, x0=x0, options=options, workers=1)


if __name__ == "__main__":
    logging.basicConfig(filename="exception_handling.log", level="DEBUG", filemode="w")
    test_exception_handling()  # -> check out the log file
