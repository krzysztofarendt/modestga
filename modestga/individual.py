import logging
import random
import pandas as pd
import numpy as np
import copy


class Individudal():

    def __init__(self, x, pop, genes, cfun):
        self.x = x
        self.pop = pop
        self.genes = genes
        self.cfun = cfun

    def calculate(self):
        pass

    def reset(self):
        pass

    def set_gene(self, name, value):
        pass

    def get_gene(self, name):
        pass

    def get_estimates(self):
        pass

    def get_error(self):
        pass

    def copy(self):
        pass

    @staticmethod
    def _get_random_genes(self):
        pass