"""Inter-method comparison."""
import logging
import os
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
    tname = "method_comparison"  # Change name and parameters below

    # Output file
    outpath = f"modestga/benchmark/results/{tname}.yaml"
    if os.path.exists(outpath):
        os.remove(outpath)

    # Number of tests
    n_tests = 5

    # Number of dimensions
    n_dims = [1, 2, 4, 8, 16, 32, 64]

    # Functions to test
    functions = [ga.minimize, differential_evolution.minimize, monte_carlo.minimize]

    # Options
    options = {
        "generations": 1000,  # Max. number of generations
        "pop_size": 50,  # Population size
        "tol": 1e-3,  # Solution tolerance
    }

    mutation = {
        "ga": [0.0025, 0.01, 0.05],  # Initial mutation, it's adaptive
        "differential_evolution": [(0, 0.5), (0, 1.0), (0, 1.9)],
        "monte_carlo": (None, None, None),
    }
    m_muts = 3

    # Results
    sim_num = -1
    res = dict()

    for nrun in range(n_tests):
        for nd in n_dims:
            for fun in functions:
                for m in range(m_muts):
                    sim_num += 1
                    res[sim_num] = dict()

                    method_name = fun.__module__.split(".")[-1]
                    mut = mutation[method_name][m]
                    opts = options.copy()
                    if mut is not None:
                        opts["mut_rate"] = mut

                    bounds = [(-5.12, 5.12) for i in range(nd)]
                    t0 = time.perf_counter()
                    out = fun(rastrigin, bounds, x0=None, options=opts)
                    elapsed_t = time.perf_counter() - t0

                    res[sim_num]["method"] = method_name
                    res[sim_num]["run"] = nrun
                    res[sim_num]["ndim"] = nd
                    res[sim_num]["mut"] = f"{mut}"
                    res[sim_num]["f(x)"] = float(out.fx)
                    res[sim_num]["nfev"] = int(out.nfev)
                    res[sim_num]["ng"] = int(out.ng)
                    res[sim_num]["time"] = elapsed_t
                    # res[sim_num]["x"] = out.x.tolist()  # List of parameters
                    # res[sim_num]["message"] = str(out.message)  # Output message

                    with open(
                        f"modestga/benchmark/results/{tname}.yaml", "a"
                    ) as yaml_file:
                        yaml.dump(res, yaml_file)

    print("Finished!")
