# modestga
Genetic Algorithm with a `scipy`-like interface:

```
minimize(fun, bounds, x0=None, args=(), callback=None, options={})
```

Main features:
- parallel (two parallel modes available: full and simple),
- adaptive mutation,
- pure Python, so easy to adapt to own needs.

## Installation
```
pip install modestga
```
To get the latest version, clone the repository and install using `setup.py`:
```
git clone https://github.com/krzysztofarendt/modestga
cd modestga
pip install -e .
```

## Test
Run example (50-dimensional [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function) minimization):
```python
>>> from modestga.examples import min_rastrigin_fun
```

## Example
```python
import logging
import random
import numpy as np
from modestga import minimize

# Set up logging if needed
logging.basicConfig(filename='ga.log', level='INFO', filemode='w')

# Define function to be minimized (can be noisy)
def fun(x, *args):
    return np.sum(x ** 2) + random.random()

# Specify parameter bounds (here: 100 parameters allowed to vary from 0 to 10)
bounds = [(0, 10) for i in range(100)]

# Overwrite default evolution options
options = {
    'generations': 1000,    # Max. number of generations
    'pop_size': 500,        # Population size
    'mut_rate': 0.01,       # Initial mutation rate (adaptive mutation)
    'trm_size': 20,         # Tournament size
    'tol': 1e-3             # Solution tolerance
}

# Minimize
# (it uses all available CPUs by default)
res = minimize(fun, bounds, options=options)

# Print optimization result
print(res)

# Final parameters
x = res.x

# Final function value
fx = res.fx

# Number of function evaluations
nfev = res.nfev
```

## Benchmarks

### modestga vs. Differential Evolution (Scipy) vs. Monte Carlo
The algorithm has been benchmarked against [Differential Evolution (SciPy)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) and naive Monte Carlo (`modestga.benchmark.methods.monte_carlo`) using the [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function). Fig. 1 shows mean results from five runs for each case. The main parameters were as follows:
- population = 100,
- maximum number of generations = 1000,
- tolerance = 1e-3,
- mutation rate - three scenarios for GA and DE,
- workers (CPUs) = 1.

The Monte Carlo method did not take into account the tolerance and was simply stopped at 1000 iteration.

Note that unlike in `modestga`, in Differentian Evolution the population size is multiplied by the number of parameters. For exact meaning of the mutation parameter in Differential Evolution please refer to the SciPy documention.

<p align="center">
<img src="modestga/benchmark/results/comparison.png" align="center">
<div align="center">Figure 1: Comparison results</div>
</p>

Summary:
- in almost all considered mutation cases `modestga` achieves similar or better result in significantly shorter time for large-scale problems (N > 32),
- `modestga` is slower for small-scale problems, especially when the cost function evaluation is fast, as in this case,
- the increasing time in Differential Evolution is due to the increasing size of population (it's multiplied by the number of parameters), but larger population seems to be inefective in solving the minimization problem.

### Number of CPUs vs. computing time
To be added soon...