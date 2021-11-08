import os
from pathlib import Path

import numpy as np
import pandas as pd

from setup_reader import read_setup
from modestga import ga
from modestga.benchmark.functions import rastrigin
from modestga.benchmark.methods import differential_evolution
from modestga.benchmark.methods import monte_carlo

if __name__ == "__main__":

    # Make sure the script is run from modestga/benchmark
    cur_dir_name = Path(os.getcwd()).name
    assert (
        cur_dir_name == "benchmark"
    ), "This script has to be run from modestga/benchmark directory"

    # Read setup
    setup = read_setup()

    # Get benchmark names
    benchmarks = setup.keys()

    # Iterate over benchmarks:
    for bmk in benchmarks:
        print(bmk)
