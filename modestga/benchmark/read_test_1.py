import yaml
import numpy as np


with open('modestga/benchmark/results/test_1.yaml') as f:
    d = yaml.load(f)

print(d)
