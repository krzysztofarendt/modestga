import unittest
import os
import shutil
import tempfile
import random
import numpy as np
from modestga import metrics


class TestEditor(unittest.TestCase):

    def setUp(self):
        # Random seed
        random.seed(123)
        np.random.seed(123)
        # Temporary directory
        self.tempdir = tempfile.mkdtemp()
        print("Temp dir created: {}".format(self.tempdir))

    def tearDown(self):
        shutil.rmtree(self.tempdir)
        assert not os.path.exists(self.tempdir)
        print("Temp dir removed: {}".format(self.tempdir))

    def test_metrics(self):
        y1 = np.arange(10)
        y2 = y1 + np.random.rand(y1.size)
        e = metrics.rmse(y1, y2)  # True: 0.586
        self.assertLess(0.586 - e, 1e-3)

    def test_individual(self):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
