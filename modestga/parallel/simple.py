"""This parallel mode is DEPRECATED. Use modestga.minimize() instead."""
import logging
import os
import queue
import time
from multiprocessing import Process
from multiprocessing import Queue

import cloudpickle
import modestga
import numpy as np
import pandas as pd
from modestga.benchmark.functions import rastrigin
from modestga.ga import OptRes

logging.basicConfig(level="INFO", format="[%(processName)s][%(levelname)s] %(message)s")


class SimpleParallel:
    """Runs multiple minimization processes in parallel and returns all results.

    This is the most simple implementation in which there is no communication
    between processes. Instead, the best final solution is returned.
    """

    def __init__(
        self,
        fun,
        bounds,
        x0=None,
        args=(),
        callback=None,
        options={},
        workers=os.cpu_count() - 1,
    ):

        self.pickled_fun = cloudpickle.dumps(fun)
        self.bounds = bounds
        self.x0 = x0
        self.pickled_args = cloudpickle.dumps(args)
        self.pickled_callback = cloudpickle.dumps(callback)
        self.options = options
        self.workers = workers
        self.queue = Queue()

    def run(self):
        processes = list()
        all_results = list()
        for i in range(self.workers):
            pid = i
            p = Process(
                target=SimpleParallel._optimization_process,
                args=(
                    self.pickled_fun,
                    self.bounds,
                    self.x0,
                    self.pickled_args,
                    self.pickled_callback,
                    self.options,
                    self.queue,
                    pid,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for the results
        n_results = 0

        while n_results < self.workers:
            try:
                res = self.queue.get(True, timeout=5)
                all_results.append(res)
                n_results = len(all_results)
            except queue.Empty:
                time.sleep(0.05)
                continue

        # Join processes
        for p in processes:
            p.join()

        # All results to dataframe
        df = pd.DataFrame(all_results).sort_values("fx")

        logging.info(f"Results from all processes:\n{df}")

        return df

    @staticmethod
    def _optimization_process(
        pickled_fun, bounds, x0, pickled_args, pickled_callback, options, queue, pid
    ):
        fun = cloudpickle.loads(pickled_fun)
        args = cloudpickle.loads(pickled_args)
        callback = cloudpickle.loads(pickled_callback)

        result = modestga.minimize(fun, bounds, x0, args, callback, options, workers=1)

        result_dict = {
            "pid": pid,
            "x": result.x,
            "message": result.message,
            "ng": result.ng,
            "nfev": result.nfev,
            "fx": result.fx,
        }

        queue.put(result_dict, block=True, timeout=5)


def minimize(
    fun, bounds, x0=None, args=(), callback=None, options={}, workers=os.cpu_count() - 1
):
    """Wrapper over SimpleParallel providing same interface as modestga.minimize()"""

    pmin = SimpleParallel(fun, bounds, x0, args, callback, options, workers)
    res = pmin.run()
    best_res = res.iloc[0]  # Beacuse df is sorted by fx in ascending order

    res = OptRes(
        x=best_res["x"],
        message=best_res["message"],
        ng=best_res["ng"],
        nfev=best_res["nfev"],
        fx=best_res["fx"],
    )
    return res


if __name__ == "__main__":
    # Example
    from modestga.benchmark.functions import rastrigin

    fun = rastrigin
    bounds = [(-5.12, 5.12) for i in range(64)]
    options = {"generations": 100, "pop_size": 100, "tol": 1e-3}

    def callback(x, fx, ng, *args):
        """Callback function called after each generation"""
        # print(f"\nCallback example:\nx=\n{x}\nf(x)={fx}\n")
        pass

    res = minimize(fun, bounds, callback=callback, options=options)
    print(res)
