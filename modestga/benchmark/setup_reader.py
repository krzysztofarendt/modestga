import yaml


def read_setup(path="setup.yaml"):
    with open(path, "r") as f:
        dct = yaml.safe_load(f)
    return dct
