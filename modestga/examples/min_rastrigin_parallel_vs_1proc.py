"""Rastrigin function minimization example.

Parallel optimization method vs. 1 process method.
This test will print a summary similar to the ones below:

    1. Result for constant population size (irrespectively of the number of workers):

    SUMMARY:
    ========
                    fx     nfev      time     time*fx
    workers
    1        53.229780  10480.0  4.234151  225.382907
    2        58.947938  10500.0  1.494908   88.121749
    4        68.272754  10500.0  0.741022   50.591583
    8        80.214267  10420.0  0.678856   54.453974

    2. Result for the population size adjusted to the number of workers:

    SUMMARY:
    ========
                     fx     nfev      time    time*fx
    workers
    1        186.162248   2080.0  0.317001  59.013659
    2        110.891146   4200.0  0.432722  47.985088
    4         68.240490   8400.0  0.520978  35.551787
    8         48.816336  16800.0  1.033473  50.450364

It means, that for the constant population size the multiprocessing approach
reduces the time it takes to go through 1 generation, but the more processes
are used, the slower is the pace of reducing the cost function per generation.

On the other hand, for the adjusted population size (= 100 * workers),
the final cost function gets lower with increasing number of workers.
At the same time the total time increases, but more slowly than the number
of function evaluation increases (due to multiprocessing).

In both cases, the optimum number of workers on my machine was 4.
"""
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modestga import minimize
from modestga.benchmark.functions import rastrigin as fun

logging.basicConfig(filename="ga.log", level="INFO", filemode="w")


# Test settings
n_tests = 10
workers_list = [1, 2, 4, 8]
generations = 20
adjusted_pop_size = True  # True or False

# Number of dimensions
N = 20


def callback(x, fx, ng, *args):
    """Callback function called after each generation"""
    # Print progress
    print("Generation #{}".format(ng))
    print("    x = {}".format(x))
    print("    fx = {}".format(fx))
    # Save to solution history
    x_hist = args[0]
    fx_hist = args[1]
    x_hist.append(x)
    fx_hist.append(fx)


bounds = [(-5.12, 5.12) for i in range(N)]

x_hist = list()
fx_hist = list()
args = (x_hist, fx_hist)

options = {
    "generations": generations,
    "trm_size": 20,
}
if adjusted_pop_size is False:
    options["pop_size"] = 500

result_df = pd.DataFrame(
        index=pd.Index(workers_list, name="workers"),
        columns=["fx", "nfev", "time", "time*fx"])

for workers in workers_list:
    test_time = []
    test_fx = []
    test_nfev = []

    for n in range(n_tests):
        t0 = time.perf_counter()
        res = minimize(fun, bounds, args=args, callback=callback, options=options, workers=workers)
        test_time.append(time.perf_counter() - t0)
        test_fx.append(res.fx)
        test_nfev.append(res.nfev)

    result_df.loc[workers, "fx"] = np.mean(test_fx)
    result_df.loc[workers, "nfev"] = np.mean(test_nfev)
    result_df.loc[workers, "time"] = np.mean(test_time)
    result_df.loc[workers, "time*fx"] = np.mean(test_time) * np.mean(test_fx)


# Print optimization results
print("SUMMARY:")
print("========")
print(result_df)
