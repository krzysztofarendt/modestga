# modestga
Genetic Algorithm with a `scipy`-like interface:

```
minimize(fun, bounds, x0=None, args=(), callback=None, options={})
```

Example:

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
options = {
    'tol': 1e-6,
    'pop_size': 100,
    'trm_size': 50,
    'mut_rate': 0.1
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


Implementation based on [ModestPy](https://github.com/sdu-cfei/modest-py).
