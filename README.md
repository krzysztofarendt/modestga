# modestga
Genetic Algorithm with a `scipy`-like interface:

```
minimize(fun, bounds, x0=None, args=(), callback=None, options={})
```

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

## Benchmark
The algorithm has been benchmarked against [Differential Evolution (SciPy)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) and naive Monte Carlo (`modestga.benchmark.methods.monte_carlo`) using the [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function). The below chart shows mean results from five runs for each case. The main parameters were as follows:
- population = 100,
- maximum number of generations = 1000,
- tolerance = 1e-3,
- mutation rate - three scenarios for GA and DE.

The Monte Carlo method did not take into account the tolerance and was simply stopped at 1000 iteration.

Note that, unlike in `modestga`, in Differentian Evolution the population size is multiplied by the number of parameters. For exact meaning of the mutation parameter in Differential Evolution please refer to the SciPy documention.

![Comparison](modestga/benchmark/results/comparison.png)

Summary:
- in almost all considered mutation cases `modestga` achieves similar or better result in significantly shorter time for large-scale problems (N > 32),
- the increasing time in Differential Evolution is due to the increasing size of population (it's multiplied by the number of parameters), but larger population seems to be inefective in solving the minimization problem,
- `modestga` is slower for small-scale problems, especially when the cost function evaluation is fast, as in this case.

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

# Specify parameter bounds (here: 100 parameters)
# Lower bounds <- 0
# Upper bounds <- 10
bounds = [(0, 10) for i in range(100)]

# Overwrite default evolution options
# (keep in mind that the defaults are quite good though!)
options = {
    'tol': 1e-6,
    'pop_size': 100,
    'trm_size': 50,
    'mut_rate': 0.01
}

# Minimize
res = minimize(fun, bounds, options=options)

# Print optimization result
print(res)

# Final parameters
x = res.x

# Final function value
fx = res.fx
```
