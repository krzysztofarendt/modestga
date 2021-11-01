"""Noisy function minimization example.

The optimization result is presented as an interactive matplotlib
chart, so make sure you have an X11 server running if you're on Linux.
"""
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from modestga import minimize

logging.basicConfig(filename="ga.log", level="INFO", filemode="w")


def fun(x, *args):
    """Noisy function to be minimized"""
    return np.sum(x ** 2) + random.random()


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


bounds = [(0, 10) for i in range(10)]

x_hist = list()
fx_hist = list()
args = (x_hist, fx_hist)

res = minimize(fun, bounds, args=args, callback=callback, workers=1)

# Print optimization result
print(res)

if __name__ == "__main__":
    # Plot solution history
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x_hist)
    ax[0].set_title("x")
    ax[1].plot(fx_hist)
    ax[1].set_title("f(x) = np.sum(x ** 2) + random.random()")
    ax[1].set_xlabel("Generation")

    plt.show()
