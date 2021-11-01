import logging
import time

import numpy as np
import yaml
from modestga import ga
from modestga.benchmark.functions import rastrigin
from modestga.benchmark.methods import differential_evolution
from modestga.benchmark.methods import monte_carlo

logging.basicConfig(level="INFO")


if __name__ == "__main__":
    # Test name
    tname = "test_3"  # Change name and parameters below

    # Number of tests
    n_tests = 5

    # Number of dimensions
    n_dims = [2, 4, 8, 16, 32, 64, 128]

    # Functions to test
    functions = [ga.minimize, differential_evolution.minimize, monte_carlo.minimize]

    # Options
    options = {
        "generations": 1000,  # Max. number of generations
        "pop_size": 100,  # Population size
        "tol": 1e-3,  # Solution tolerance
    }

    mutation = {
        "ga": 0.0025,  # Initial mutation, it's adaptive
        "differential_evolution": (0, 0.5),
        "monte_carlo": None,
    }

    # Results
    res = dict()

    for i in range(n_tests):
        k1 = f"run={i}"
        res[k1] = dict()

        for nd in n_dims:
            k2 = f"ndim={nd}"
            res[k1][k2] = dict()

            for fun in functions:
                fname = fun.__module__.split(".")[-1]
                k3 = f"method={fname}"
                res[k1][k2][k3] = dict()

                mut = mutation[fname]
                print(mut, fname)
                opts = options.copy()
                if mut is not None:
                    opts["mut_rate"] = mut

                bounds = [(-5.12, 5.12) for i in range(nd)]
                t0 = time.perf_counter()
                out = fun(rastrigin, bounds, x0=None, options=opts)
                elapsed_t = time.perf_counter() - t0

                res[k1][k2][k3]["f(x)"] = float(out.fx)
                res[k1][k2][k3]["x"] = out.x.tolist()
                res[k1][k2][k3]["nfev"] = int(out.nfev)
                res[k1][k2][k3]["ng"] = int(out.ng)
                res[k1][k2][k3]["message"] = str(out.message)
                res[k1][k2][k3]["time"] = elapsed_t
                res[k1][k2][k3]["mut"] = f"{mut}"

                with open(f"modestga/benchmark/results/{tname}.yaml", "w") as yaml_file:
                    yaml.dump(res, yaml_file)

    print("Finished!")
