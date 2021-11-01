"""Rastrigin function minimization example.

The optimization result is presented as an interactive matplotlib
chart, so make sure you have an X11 server running if you're on Linux.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from modestga import minimize
from modestga.benchmark.functions import rastrigin as fun

logging.basicConfig(filename="ga.log", level="INFO", filemode="w")

# Number of dimensions
N = 50


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
    "generations": 500,
}

res = minimize(fun, bounds, args=args, callback=callback, options=options)

# Print optimization result
print(res)

if __name__ == "__main__":
    # Plot solution history
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x_hist, alpha=0.1, color="k")
    ax[0].set_title("x")
    ax[1].plot(fx_hist, color="k")
    ax[1].set_title("f(x) = RASTRIGIN FUNC.")
    ax[1].set_xlabel("Generation")

    plt.show()
