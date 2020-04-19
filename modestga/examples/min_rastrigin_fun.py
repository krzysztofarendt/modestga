import logging
import numpy as np
import matplotlib.pyplot as plt
from modestga import minimize


logging.basicConfig(filename='ga.log', level='INFO', filemode='w')

# Number of dimensions
N = 50

def fun(x, *args):
    """Rastrigin function to be minimized.

    Global minimum y=0 at x=[0, 0, ..., 0].
    """
    A = 100
    global N
    y = A * N + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))
    return y

# print(fun(np.array([0, 0.3])))

def callback(x, fx, ng, *args):
    """Callback function called after each generation"""

    # Print progress
    print('Generation #{}'.format(ng))
    print('    x = {}'.format(x))
    print('    fx = {}'.format(fx))

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
    'tol': 1e-8,
    'pop_size': 200,
    'trm_size': 40,
    'mut_rate': 0.01,
    'generations': 1000,
    'inertia': 50,
    'xover_ratio': 0.2
}

res = minimize(fun, bounds, args=args, callback=callback, options=options)

# Print optimization result
print(res)

# Plot solution history

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(x_hist, alpha=0.1, color='k')
ax[0].set_title('x')
ax[1].plot(fx_hist, color='k')
ax[1].set_title('f(x) = RASTRIGIN FUNC.')
ax[1].set_xlabel('Generation')

plt.show()
