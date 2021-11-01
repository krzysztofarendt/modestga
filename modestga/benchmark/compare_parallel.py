import logging
import time

import numpy as np
import yaml
from modestga import ga
from modestga.benchmark.functions import rastrigin
from modestga.parallel import simple

logging.basicConfig(level="INFO")


if __name__ == "__main__":
    # Test name
    tname = "parallel_results_1"  # Change name and parameters below

    # Number of tests
    n_tests = 5

    # Number of dimensions
    n_dims = 128

    # Workers
    n_workers = [1, 2, 3, 4, 5, 6, 7, 8]

    # Functions to test
    functions = [
        ga.minimize
        # simple.minimize
    ]

    # Options
    options = {
        "generations": 1000,  # Max. number of generations
        "pop_size": 1000,  # Population size
        "tol": 1e-3,  # Solution tolerance
    }

    # Results
    res = dict()

    for i in range(n_tests):
        k1 = f"run={i}"
        res[k1] = dict()

        for nw in n_workers:
            k2 = f"nworkers={nw}"
            res[k1][k2] = dict()

            for fun in functions:
                fname = fun.__module__.split(".")[-1]
                k3 = f"method={fname}"
                res[k1][k2][k3] = dict()

                opts = options.copy()

                bounds = [(-5.12, 5.12) for i in range(n_dims)]
                t0 = time.perf_counter()
                out = fun(rastrigin, bounds, x0=None, options=opts, workers=nw)
                elapsed_t = time.perf_counter() - t0

                res[k1][k2][k3]["f(x)"] = float(out.fx)
                res[k1][k2][k3]["x"] = out.x.tolist()
                res[k1][k2][k3]["nfev"] = int(out.nfev)
                res[k1][k2][k3]["ng"] = int(out.ng)
                res[k1][k2][k3]["message"] = str(out.message)
                res[k1][k2][k3]["time"] = elapsed_t

                with open(f"modestga/benchmark/results/{tname}.yaml", "w") as yaml_file:
                    yaml.dump(res, yaml_file)

    print("Finished!")
