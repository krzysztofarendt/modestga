"""Analyze results of inter-method comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_df(path):
    # Load data to a dataframe
    with open(path) as f:
        d = yaml.safe_load(f)

    df = pd.DataFrame(
        columns=["run", "ndim", "method", "f(x)", "nfev", "ng", "time"],
    )

    for nsim in d.keys():
        s = pd.Series(d[nsim])
        df = df.append(s, ignore_index=True)

    df["run"] = df["run"].astype(int)
    df["ndim"] = df["ndim"].astype(int)
    df["f(x)"] = df["f(x)"].astype(float)
    df["nfev"] = df["nfev"].astype(int)
    df["ng"] = df["ng"].astype(int)
    df["time"] = df["time"].astype(float)

    return df


yaml_path = "./modestga/benchmark/results/method_comparison.yaml"
df = load_df(yaml_path)

# Calculate average results
summary = df.groupby(["method", "mut", "ndim"]).mean()

# Plot results
fig, axes = plt.subplots(3, 3, sharey=False, sharex=True, figsize=(8, 8), dpi=80)

ax = axes[0, 0]
ax.set_xticks([2, 16, 32, 64, 128])
ax.set_ylim((0, 11000))
ax.set_yscale("symlog")
ax.set_ylabel("Final function value $f(x)$")
ax.set_title("Genetic Algorithm\n(modestga)")
ax.plot(
    summary.loc[("ga", "0.0025", slice(None)), ["f(x)"]].droplevel([0, 1]),
    label="mut=0.0025",
    c="r",
)
ax.plot(
    summary.loc[("ga", "0.01", slice(None)), ["f(x)"]].droplevel([0, 1]),
    label="mut=0.01",
    c="b",
)
ax.plot(
    summary.loc[("ga", "0.05", slice(None)), ["f(x)"]].droplevel([0, 1]),
    label="mut=0.05",
    c="g",
)
ax.legend(loc="upper left")

ax = axes[0, 1]
ax.set_ylim((0, 11000))
ax.set_yscale("symlog")
ax.set_title("Differential Evolution\n(scipy)")
ax.plot(
    summary.loc[
        ("differential_evolution", "(0, 0.5)", slice(None)), ["f(x)"]
    ].droplevel([0, 1]),
    label="mut=(0, 0.5)",
    c="r",
)
ax.plot(
    summary.loc[
        ("differential_evolution", "(0, 1.0)", slice(None)), ["f(x)"]
    ].droplevel([0, 1]),
    label="mut=(0, 1.0)",
    c="b",
)
ax.plot(
    summary.loc[
        ("differential_evolution", "(0, 1.9)", slice(None)), ["f(x)"]
    ].droplevel([0, 1]),
    label="mut=(0, 1.9)",
    c="g",
)
ax.legend(loc="upper left")

ax = axes[0, 2]
ax.set_ylim((0, 11000))
ax.set_yscale("symlog")
ax.set_title("Monte Carlo\n(modestga/baseline)")
ax.plot(
    summary.loc[("monte_carlo", "None", slice(None)), ["f(x)"]].droplevel([0, 1]), c="r"
)

ax = axes[1, 0]
ax.set_ylim((1e3, 1e8))
ax.set_yscale("symlog")
ax.set_ylabel("Function evaluations $n_{ev}$")
ax.plot(
    summary.loc[("ga", "0.0025", slice(None)), ["nfev"]].droplevel([0, 1]),
    label="mut=0.0025",
    c="r",
)
ax.plot(
    summary.loc[("ga", "0.01", slice(None)), ["nfev"]].droplevel([0, 1]),
    label="mut=0.01",
    c="b",
)
ax.plot(
    summary.loc[("ga", "0.05", slice(None)), ["nfev"]].droplevel([0, 1]),
    label="mut=0.05",
    c="g",
)
ax.legend(loc="upper left")

ax = axes[1, 1]
ax.set_ylim((1e3, 1e8))
ax.set_yscale("symlog")
ax.plot(
    summary.loc[
        ("differential_evolution", "(0, 0.5)", slice(None)), ["nfev"]
    ].droplevel([0, 1]),
    label="mut=(0, 0.5)",
    c="r",
)
ax.plot(
    summary.loc[
        ("differential_evolution", "(0, 1.0)", slice(None)), ["nfev"]
    ].droplevel([0, 1]),
    label="mut=(0, 1.0)",
    c="b",
)
ax.plot(
    summary.loc[
        ("differential_evolution", "(0, 1.9)", slice(None)), ["nfev"]
    ].droplevel([0, 1]),
    label="mut=(0, 1.9)",
    c="g",
)
ax.legend(loc="upper left")

ax = axes[1, 2]
ax.set_ylim((1e3, 1e8))
ax.set_yscale("symlog")
ax.plot(
    summary.loc[("monte_carlo", "None", slice(None)), ["nfev"]].droplevel([0, 1]), c="r"
)

ax = axes[2, 0]
ax.set_ylim((0, 8000))
ax.set_yscale("symlog")
ax.set_xlabel("Number of parameters $N$")
ax.set_ylabel("Computing time $t$ [s]")
ax.plot(
    summary.loc[("ga", "0.0025", slice(None)), ["time"]].droplevel([0, 1]),
    label="mut=0.0025",
    c="r",
)
ax.plot(
    summary.loc[("ga", "0.01", slice(None)), ["time"]].droplevel([0, 1]),
    label="mut=0.01",
    c="b",
)
ax.plot(
    summary.loc[("ga", "0.05", slice(None)), ["time"]].droplevel([0, 1]),
    label="mut=0.05",
    c="g",
)
ax.legend(loc="upper left")

ax = axes[2, 1]
ax.set_ylim((0, 8000))
ax.set_yscale("symlog")
ax.set_xlabel("Number of parameters $N$")
ax.plot(
    summary.loc[
        ("differential_evolution", "(0, 0.5)", slice(None)), ["time"]
    ].droplevel([0, 1]),
    label="mut=(0, 0.5)",
    c="r",
)
ax.plot(
    summary.loc[
        ("differential_evolution", "(0, 1.0)", slice(None)), ["time"]
    ].droplevel([0, 1]),
    label="mut=(0, 1.0)",
    c="b",
)
ax.plot(
    summary.loc[
        ("differential_evolution", "(0, 1.9)", slice(None)), ["time"]
    ].droplevel([0, 1]),
    label="mut=(0, 1.9)",
    c="g",
)
ax.legend(loc="upper left")

ax = axes[2, 2]
ax.set_ylim((0, 8000))
ax.set_yscale("symlog")
ax.set_xlabel("Number of parameters $N$")
ax.plot(
    summary.loc[("monte_carlo", "None", slice(None)), ["time"]].droplevel([0, 1]), c="r"
)

plt.tight_layout()
plt.savefig("./modestga/benchmark/results/comparison.png")
