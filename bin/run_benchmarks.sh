#!/bin/bash

echo "This script will run all benchmark simulations.
This might take few hours to complete!"

read -r -p "Do you want to continue? [y/N] " response

case "$response" in
    [yY][eE][sS]|[yY])
        echo "Ok, let's go!"
        source venv/bin/activate

        # Run benchmark 2 first, because it's faster
        # (single vs. multi CPU)
        python ./modestga/benchmark/benchmark_2_run.py
        python ./modestga/benchmark/benchmark_2_plot.py

        # Benchmark 1 is an inter-method comparison
        # (modestga, differential evolution, monte carlo)
        python ./modestga/benchmark/benchmark_1_run.py
        python ./modestga/benchmark/benchmark_1_plot.py
        ;;
esac
