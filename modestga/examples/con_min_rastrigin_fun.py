"""Example of constrained minimization of the rastrigin function."""
import logging

from modestga import con_minimize

# Set up logging
logging.basicConfig(
    level="DEBUG", filemode="w", format="[%(processName)s][%(levelname)s] %(message)s"
)

# Set up cost function
from modestga.benchmark.functions import rastrigin

fun = rastrigin

# Set up bounds (rectangular)
bounds = [(-5.12, 5.12) for i in range(8)]

# Constraints (algorithm will try to keep their outputs positive)
def fcon1(x, *args):
    return x[1] - 1.0


def fcon2(x, *args):
    return x[0] - 2.0


constr = (fcon1, fcon2)

# Genetic Algorithm options
options = {"generations": 1000, "pop_size": 500, "tol": 1e-3}

# Run minimization
res = con_minimize(fun, bounds, constr=constr, options=options)

print(res)
