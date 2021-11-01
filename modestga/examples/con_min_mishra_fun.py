"""Example of constrained optimization of the Mishra's Bird function.

Global minimum at f(-3.1302, -1.5821) = -106.7645
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from modestga import con_minimize
from modestga.benchmark.functions.mishra_bird import mishra_bird
from modestga.benchmark.functions.mishra_bird import mishra_bird_constr

# Set up logging
logging.basicConfig(
    level="DEBUG", filemode="w", format="[%(processName)s][%(levelname)s] %(message)s"
)

# Run minimization
res = con_minimize(
    fun=mishra_bird,
    bounds=[(-10.0, 0.0), (-6.5, 0.0)],
    constr=[mishra_bird_constr],
    workers=1,
)

print(res)
