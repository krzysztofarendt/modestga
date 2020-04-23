import yaml
import numpy as np
import pandas as pd


# Load data to a dataframe
with open('modestga/benchmark/results/test_1.yaml') as f:
    d = yaml.load(f)

df = pd.DataFrame(columns=['run', 'ndim', 'method', 'f(x)', 'nfev', 'ng', 'time', 'mut'])

for k1 in d.keys():
    for k2 in d[k1].keys():
        for k3 in d[k1][k2].keys():
            s = pd.Series({
                'run': k1.split('=')[-1],
                'ndim': k2.split('=')[-1],
                'method': k3.split('=')[-1],
                'f(x)': d[k1][k2][k3]['f(x)'],
                'nfev': d[k1][k2][k3]['nfev'],
                'ng': d[k1][k2][k3]['ng'],
                'time': d[k1][k2][k3]['time']
            })
            df = df.append(s, ignore_index=True)

print(df)

# Calculate average results
print(
    df.groupby(['method', 'ndim']).mean()
)
