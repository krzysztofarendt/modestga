"""Example of constrained optimization of the Rosenbrock function.

Global minimum at f(1., 1.) = 0.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from modestga import con_minimize
from modestga.benchmark.functions.rosenbrock import rosenbrock_2par
from modestga.benchmark.functions.rosenbrock import rosenbrock_constr1
from modestga.benchmark.functions.rosenbrock import rosenbrock_constr2

# Set up logging
logging.basicConfig(
    level="DEBUG", filemode="w", format="[%(processName)s][%(levelname)s] %(message)s"
)

# Run minimization
res = con_minimize(
    fun=rosenbrock_2par,
    bounds=[(-1.5, 1.5), (-0.5, 2.5)],
    constr=[rosenbrock_constr1, rosenbrock_constr2],
    workers=1,
    options={"tol": 1e-6},
)

print(res)
