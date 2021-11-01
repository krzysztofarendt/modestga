import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_df(path):
    # Load data to a dataframe
    with open(path) as f:
        d = yaml.safe_load(f)

    df = pd.DataFrame(
        columns=["run", "nworkers", "method", "f(x)", "nfev", "ng", "time"]
    )

    for k1 in d.keys():
        for k2 in d[k1].keys():
            for k3 in d[k1][k2].keys():
                s = pd.Series(
                    {
                        "run": int(k1.split("=")[-1]),
                        "nworkers": int(k2.split("=")[-1]),
                        "method": k3.split("=")[-1],
                        "f(x)": float(d[k1][k2][k3]["f(x)"]),
                        "nfev": float(d[k1][k2][k3]["nfev"]),
                        "ng": float(d[k1][k2][k3]["ng"]),
                        "time": float(d[k1][k2][k3]["time"]),
                    }
                )
                df = df.append(s, ignore_index=True)

    return df


df = load_df("./modestga/benchmark/results/parallel_results.yaml")

# Calculate average results
m = df.groupby(["method", "nworkers"]).median()

# Plot comparison
fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=90)

ax.plot(m.loc["ga"]["time"])
ax.set_xlabel("CPUs")
ax.set_ylabel("Time [s]")
plt.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.95)
fig.savefig("./modestga/benchmark/results/comparison_parallel.png")
